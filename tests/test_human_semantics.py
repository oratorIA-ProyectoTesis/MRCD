"""U5a human semantics/proposal integrity, synthetic HTTP and signal evidence only."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest
from fastapi.testclient import TestClient

from core.human_events import HUMAN_ANNOTATION_SCHEMA, HUMAN_EVENT_SCHEMA, validate_human_event
from core.cell_evidence import cell_evidence
from core.cell_engine import CellAnalysisEngine
from core.features_v2 import build_cell_features
from core.targets import project_targets
from core import evaluation
from app.api import create_app
from app.store import Store
from tests.test_cells_product import prep, analysis, FakeCellInference
from tests.test_cells_v2 import raw_signals


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setenv("MRCD_DATA", str(tmp_path / "data"))
    store = Store(f"sqlite:///{tmp_path/'app.db'}")
    users = {}
    for name, role in (("admin","admin"),("ana","annotator"),("beto","annotator"),("carla","adjudicator"),("rita","researcher")):
        users[name] = store.create("user", dict(name=name,role=role,token_sha256=hashlib.sha256(name.encode()).hexdigest()),status="active")["id"]
    rec = store.create("recording",dict(sha256="a"*64,speaker_id="synthetic-s",media=dict(analysis=analysis(1000),original=dict(has_video=False)),segments=[dict(id="s0",core_start_ms=0,core_end_ms=1000)]),status="ready")
    output = CellAnalysisEngine(FakeCellInference()).analyze(prep(1000),analysis=analysis(1000),recording_id=rec["id"],segments=rec["data"]["segments"])
    run = store.create("run",dict(system="mrcd",result=output),parent=rec["id"],status="succeeded")
    client=TestClient(create_app(store))
    def call(who,method,url,**kw):
        return client.request(method,"/api"+url,headers={"Authorization":f"Bearer {who}",**kw.pop("headers",{})},**kw)
    return store,call,users,rec,run


def task(api, who="ana", mode="assisted"):
    _,call,users,rec,run=api
    response=call("rita","POST","/review/tasks",json=dict(recording_id=rec["id"],start_ms=0,end_ms=1000,context_ms=0,mode=mode,split="dev",assignee=users[who],run_id=run["id"]))
    assert response.status_code==200,response.text
    tid=response.json()["id"]
    view=call(who,"GET",f"/review/tasks/{tid}")
    assert view.status_code==200,view.text
    return tid,view.json()


def mark(suggestion, **values):
    return dict(event_id="human-1",start_ms=490,end_ms=510,label=suggestion["label"],decision="event",type_known=True,is_disfluent=None,
                suggestion_id=suggestion["event_id"],proposal_ref=suggestion["proposal_ref"],**values)


def confirmation(event):
    return dict(type="confirm_human_event",event_id=event["event_id"],reviewed_event={k:event.get(k) for k in ("start_ms","end_ms","label","decision","uncertainty","type_known","is_disfluent")})


def patch(api,tid,view,event,*,confirmed=True,who="ana",contextual=False):
    _,call,_,_,_=api
    return call(who,"PATCH",f"/annotations/{view['annotation']['id']}",headers={"If-Match":str(view["annotation"]["rev"])},json=dict(semantic_schema=HUMAN_ANNOTATION_SCHEMA,events=[] if contextual else [event],contextual=[event] if contextual else [],coverage=[[0,1000]],actions=[confirmation(event)] if confirmed else []))


def test_proposal_copy_never_implies_known_type_or_context_and_blind_stays_prediction_free(api):
    store,call,_,_,_=api
    tid,view=task(api)
    suggestion=view["suggestions"][0]
    assert suggestion["type_known"] is False and suggestion["is_disfluent"] is None
    event=mark(suggestion); event.update(source_kind="human_verified_model_suggestion",provenance=dict(actor="invented"))
    response=patch(api,tid,view,event,confirmed=False)
    assert response.status_code==200,response.text
    stored=response.json()["data"]["events"][0]
    assert stored["type_known"] is False and stored["is_disfluent"] is None
    assert stored["source_kind"]=="model_proposal_copy" and stored["human_event_schema"]==HUMAN_EVENT_SCHEMA
    assert stored["provenance"]["actor_id"]!="invented"
    assert [stored["start_ms"],stored["end_ms"]]==[490,510]
    _,blind=task(api,"beto","blind")
    assert blind["suggestions"]==[] and "proposal_ref" not in json.dumps(blind) and "run_id" not in json.dumps(blind["task"])


@pytest.mark.parametrize("fault",["spoof_id","wrong_run","stale_run","changed_digest","stale_confirmation"])
def test_spoofed_or_stale_proposal_or_confirmation_rejects_atomically(api,fault):
    store,call,_,_,run=api
    tid,view=task(api); event=mark(view["suggestions"][0]); body=confirmation(event)
    if fault=="spoof_id": event["suggestion_id"]="invented"
    elif fault=="wrong_run": event["proposal_ref"]["run_id"]="other-run"
    elif fault=="stale_run": store.update(run["id"],{**run["data"],"note":"changed after task opened"})
    elif fault=="changed_digest": event["proposal_ref"]["event_sha256"]="f"*64
    else: event["end_ms"]=520
    before=deepcopy(store.get(view["annotation"]["id"])); before_task=deepcopy(store.get(tid))
    response=call("ana","PATCH",f"/annotations/{before['id']}",headers={"If-Match":str(before["rev"])},json=dict(semantic_schema=HUMAN_ANNOTATION_SCHEMA,events=[event],coverage=[[0,1000]],actions=[body]))
    assert response.status_code in (409,422),response.text
    assert store.get(before["id"])==before and store.get(tid)==before_task


@pytest.mark.parametrize("context",[None,False,True])
def test_confirm_type_context_independently_through_adjudication_and_freeze(api,context):
    store,call,_,_,_=api
    tids=[]; saved=[]
    for who in ("ana","beto"):
        tid,view=task(api,who); event=mark(view["suggestions"][0]); event["is_disfluent"]=context
        response=patch(api,tid,view,event,who=who)
        assert response.status_code==200,response.text
        human=response.json()["data"]["events"][0]; saved.append(human)
        assert human["type_known"] is True and human["is_disfluent"] is context and human["label"]=="filler_word"
        assert human["contextual_state"]==({True:"disfluent",False:"functional"}.get(context))
        assert human["provenance"]["proposal_bounds"]==[0,1000]
        targets=project_targets(1000,[human],[[0,1000]])
        assert targets["multi_hot"][:,0].tolist()==[True,True]
        assert targets["context_mask"][:,0].tolist()==[context is not None]*2
        assert call(who,"POST",f"/review/tasks/{tid}/submit").status_code==200
        tids.append(tid)
    cid=call("admin","POST","/adjudication-cases",json=dict(task_a=tids[0],task_b=tids[1],adjudicator=api[2]["carla"])).json()["id"]
    view=call("carla","GET",f"/review/adjudication/{cid}").json()
    event=deepcopy(saved[0])
    response=call("carla","POST","/adjudications",json=dict(semantic_schema=HUMAN_ANNOTATION_SCHEMA,case_id=cid,case_revision=view["case"]["rev"],events=[event],contextual=[],actions=[confirmation(event)],reasons=dict(review="Explicit synthetic human review"),based_on={k:view["tracks"][k]["rev"] for k in "AB"}))
    assert response.status_code==200,response.text
    frozen=call("admin","POST","/dataset-snapshots",json=dict(name="semantic-test",splits=["dev"]))
    assert frozen.status_code==200,frozen.text
    final=frozen.json()["data"]["items"][0]["events"][0]
    assert [final[k] for k in ("start_ms","end_ms","label","type_known","is_disfluent")]==[490,510,"filler_word",True,context]
    assert final["provenance"]["source_annotations"] and final["provenance"]["proposal_bounds"]==[0,1000]
    with pytest.raises(ValueError,match="task-aware"): evaluation.reference_view(frozen.json()["data"],{})


def test_contextual_functional_phenotype_and_explicit_uncertainty_survive(api):
    tid,view=task(api); event=mark(view["suggestions"][0]); event.update(expression="eh",is_disfluent=False)
    response=patch(api,tid,view,event,contextual=True)
    assert response.status_code==200,response.text
    stored=response.json()["data"]["contextual"][0]
    assert stored["label"]=="filler_word" and stored["type_known"] and stored["is_disfluent"] is False
    validate_human_event(stored)
    for reason in ("class","boundaries","audio"):
        human=dict(stored,decision="uncertain",uncertainty=reason,is_disfluent=None,contextual_state=None)
        validate_human_event(human)
        assert project_targets(1000,[human],[[0,1000]])["multi_hot"][:,0].all()


def test_acquired_effective_evidence_and_unmeasured_quality_are_distinct():
    raw=raw_signals(3000)
    raw.update(k_times=np.arange(30)/10,k_vectors=np.ones((30,12)),k_detected=np.ones(30,bool))
    features=build_cell_features(raw,[],3000)
    caps=dict(modalities=dict(ac=True,pros=True,ling=False,kin=False))
    evidence=cell_evidence(features,mode="audio_text",capabilities=caps)
    center=evidence[3]
    assert center["acquired"]["kin"]["available"] is True
    assert center["effective"]["kin"]["available"] is False and "mode_disabled" in center["effective"]["kin"]["reasons"]
    assert center["quality"]["kin"]["value"] is None and center["quality"]["kin"]["reason"]=="quality_not_assessed"
    assert center["effective"]["ac"]["available"] is True
    assert center["acquired"]["ling"]["available"] is False and center["acquired"]["ling"]["reasons"]
    json.dumps(evidence,allow_nan=False)


def test_ui_proposal_copy_is_unconfirmed_and_confirmation_is_explicit():
    source=(Path(__file__).parents[1]/"app/static/app.js").read_text()
    code="function humanProposal(s, id) {"+source.split("function humanProposal(s, id) {",1)[1].split("// END HUMAN SEMANTICS",1)[0]
    script=code+"\nconst p=humanProposal({event_id:'m1',label:'filler_word',start_ms:490,end_ms:510,proposal_ref:{run_id:'r'}},'h1');console.log(JSON.stringify({p,a:humanConfirmation({...p,type_known:true,is_disfluent:null})}));"
    out=json.loads(subprocess.run(["node","-e",script],capture_output=True,text=True,check=True).stdout)
    assert out["p"]["type_known"] is False and out["p"]["is_disfluent"] is None
    assert out["a"]["type"]=="confirm_human_event" and out["a"]["reviewed_event"]["type_known"] is True


def test_versioned_negative_only_snapshot_keeps_schema_and_stays_guarded(api):
    _,call,_,_,_=api
    tid,view=task(api,mode="blind")
    response=call("ana","PATCH",f"/annotations/{view['annotation']['id']}",headers={"If-Match":str(view["annotation"]["rev"])},json=dict(semantic_schema=HUMAN_ANNOTATION_SCHEMA,events=[],contextual=[],coverage=[[0,1000]],actions=[]))
    assert response.status_code==200,response.text
    assert call("ana","POST",f"/review/tasks/{tid}/submit").status_code==200
    snapshot=call("admin","POST","/dataset-snapshots",json=dict(name="negative-only",include="submitted",splits=["dev"]))
    assert snapshot.status_code==200,snapshot.text
    snap=snapshot.json()
    assert snap["data"]["items"][0]["semantic_schema"]==HUMAN_ANNOTATION_SCHEMA
    with pytest.raises(ValueError,match="task-aware"): evaluation.reference_view(snap["data"],{})
    for format in ("csv","eaf"):
        assert call("rita","GET",f"/exports/{snap['id']}?format={format}").status_code==422


def test_explicit_class_boundary_audio_uncertainty_preserves_human_type_over_http(api):
    for reason in ("class","boundaries","audio"):
        tid,view=task(api)
        event=mark(view["suggestions"][0]); event.update(decision="uncertain",uncertainty=reason)
        response=patch(api,tid,view,event)
        assert response.status_code==200,response.text
        saved=response.json()["data"]["events"][0]
        assert saved["type_known"] is True and saved["uncertainty"]==reason and saved["decision"]=="uncertain"
        assert saved["is_disfluent"] is None


def test_historical_annotation_get_is_not_rewritten_as_versioned_evidence(api):
    store,call,_,_,_=api
    tid,view=task(api,mode="blind")
    legacy=dict(events=[dict(event_id="old",start_ms=100,end_ms=200,label="filler_word",decision="event")],contextual=[],coverage=[])
    store.update(view["annotation"]["id"],legacy)
    before=deepcopy(store.get(view["annotation"]["id"]))
    response=call("ana","GET",f"/review/tasks/{tid}")
    assert response.json()["annotation"]==before and store.get(before["id"])==before
    assert "type_known" not in response.json()["annotation"]["data"]["events"][0]


def test_quality_effective_masks_include_training_column_counts_without_changing_acquisition():
    raw=raw_signals(1000); f=build_cell_features(raw,[],1000)
    widths=dict(ac=5,pros=8,ling=12,kin=12)
    stats={key:dict(mean=[0.]*width,std=[1.]*width,count=[0]*width) for key,width in widths.items()}
    report=cell_evidence(f,mode="audio_text",capabilities=dict(modalities=dict(ac=True,pros=True,ling=False,kin=False)),stats=stats)
    assert report[0]["acquired"]["ac"]["available"] is True
    assert report[0]["effective"]["ac"]["available"] is False
    assert "no_trained_measurements" in report[0]["effective"]["ac"]["reasons"]
    assert report[0]["quality"]["ac"]["value"] is None


def test_actual_ui_copy_and_confirm_action_payload_roundtrips_to_server(api):
    tid,view=task(api); s=view["suggestions"][0]
    source=(Path(__file__).parents[1]/"app/static/app.js").read_text()
    helpers="function humanProposal(s, id) {"+source.split("function humanProposal(s, id) {",1)[1].split("// END HUMAN SEMANTICS",1)[0]
    suggestion="function suggestion(s, act) {"+source.split("function suggestion(s, act) {",1)[1].split("\n  view.onclick",1)[0]
    save="function saveDraft() {"+source.split("function saveDraft() {",1)[1].split("\n  function removeKey",1)[0]
    stubs="""let state={semantic_schema:'mrcd-human-annotation-v1',events:[],contextual:[],actions:[]},locked=false,editing=null,isCase=false,fromSuggestion=null;
const s=INPUT; let sel={start_ms:490,end_ms:510};
const uid=()=> 'ui-human', commit=(fn,action)=>{fn();if(action)state.actions.push(action)}, fillForm=()=>{},say=()=>{},draw=()=>{},toast=()=>{},resetForm=()=>{},renderSel=()=>{};
let kind='cls:filler_word'; const outsideRegion=()=>false,editingEvent=()=>null,chosen=()=>kind;
const values={'f-note':{value:''},'f-known':{checked:true},'f-context':{value:'null'},'f-text':{value:'eh'},'f-spk':{value:''},'f-expr':{value:'eh'},'f-dis':{value:'false'},'f-func':{value:'functional'},'f-phenotype':{value:'filler_word'}};
const $=id=>values[id.slice(1)];
""".replace("INPUT",json.dumps(s))
    script=helpers+suggestion+save+stubs+"\nsuggestion(s,'accept');const copy=JSON.parse(JSON.stringify(state));state={semantic_schema:'mrcd-human-annotation-v1',events:[],contextual:[],actions:[]};fromSuggestion=s;saveDraft();const confirmed=JSON.parse(JSON.stringify(state));state={semantic_schema:'mrcd-human-annotation-v1',events:[],contextual:[{...confirmed.events[0],expression:'eh'}],actions:[]};fromSuggestion=null;editing={type:'ctx',index:0};kind='legit';saveDraft();console.log(JSON.stringify({copy,confirmed,contextual:state}));"
    out=json.loads(subprocess.run(["node","-e",script],capture_output=True,text=True,check=True).stdout)
    assert out["copy"]["events"][0]["type_known"] is False and out["copy"]["events"][0]["is_disfluent"] is None
    assert out["confirmed"]["actions"][0]["type"]=="confirm_human_event"
    assert out["contextual"]["contextual"][0]["proposal_ref"]==s["proposal_ref"]
    assert out["contextual"]["contextual"][0]["suggestion_id"]==s["event_id"]
    _,call,_,_,_=api
    response=call("ana","PATCH",f"/annotations/{view['annotation']['id']}",headers={"If-Match":str(view["annotation"]["rev"])},json=dict(**out["confirmed"],coverage=[[0,1000]]))
    assert response.status_code==200,response.text
    assert response.json()["data"]["events"][0]["type_known"] is True
    subprocess.run(["node","--check",str(Path(__file__).parents[1]/"app/static/app.js")],capture_output=True,text=True,check=True)


def test_legacy_client_cannot_claim_new_context_or_verified_provenance(api):
    store,call,_,_,_=api
    tid,view=task(api,mode="blind")
    before=deepcopy(store.get(view["annotation"]["id"]))
    response=call("ana","PATCH",f"/annotations/{before['id']}",headers={"If-Match":str(before["rev"])},json=dict(events=[dict(event_id="legacy",start_ms=100,end_ms=200,label="filler_word",is_disfluent=True,source_kind="human_verified_model_suggestion",provenance=dict(confirmed=True,actor_id="fake"))]))
    assert response.status_code==422
    assert store.get(before["id"])==before


def test_manual_mark_in_assisted_task_records_exposure_not_independence(api):
    tid,view=task(api)
    event=mark(view["suggestions"][0]); event.pop("suggestion_id");event.pop("proposal_ref")
    response=patch(api,tid,view,event)
    assert response.status_code==200,response.text
    saved=response.json()["data"]["events"][0]
    assert saved["source_kind"]=="human_assisted_manual" and saved["provenance"]["mode"]=="assisted"


def test_legacy_unbound_suggestion_link_cannot_be_certified_or_crash_new_adjudication(api):
    store,call,users,_,_=api; tids=[]
    event=dict(event_id="old-human",start_ms=100,end_ms=200,label="filler_word",decision="event",suggestion_id="old-unbound-model",proposal_ref=None,provenance=None)
    for who in ("ana","beto"):
        tid,view=task(api,who,mode="blind")
        store.update(view["annotation"]["id"],dict(events=[event],contextual=[],coverage=[[0,1000]]),status="submitted")
        store.update(tid,status="submitted");tids.append(tid)
    cid=call("admin","POST","/adjudication-cases",json=dict(task_a=tids[0],task_b=tids[1],adjudicator=users["carla"])).json()["id"]
    view=call("carla","GET",f"/review/adjudication/{cid}").json();before=deepcopy(store.get(cid))
    current=dict(event,type_known=True,is_disfluent=None)
    response=call("carla","POST","/adjudications",json=dict(semantic_schema=HUMAN_ANNOTATION_SCHEMA,case_id=cid,case_revision=before["rev"],events=[current],actions=[confirmation(current)],based_on={k:view["tracks"][k]["rev"] for k in "AB"}))
    assert response.status_code==422 and "unverified" in response.text
    assert store.get(cid)==before and store.find("adjudication")==[]
