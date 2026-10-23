"""Independent U5a HTTP contract probes; synthetic SQLite and in-process client only."""
from copy import deepcopy
import json

from tests.test_human_semantics import api, task, mark, patch, confirmation
from core.human_events import HUMAN_ANNOTATION_SCHEMA, HUMAN_EVENT_SCHEMA
from core import evaluation
from core.cell_decode import decode_cells, validate_cell_result


def make_case(api, *, versioned=False):
    store, call, users, _, _ = api
    tids = []
    for who in ("ana", "beto"):
        tid, view = task(api, who, mode="blind")
        body = dict(events=[], contextual=[], coverage=[[0, 1000]], actions=[])
        if versioned:
            body["semantic_schema"] = HUMAN_ANNOTATION_SCHEMA
        response = call(who, "PATCH", f"/annotations/{view['annotation']['id']}",
                        headers={"If-Match": str(view["annotation"]["rev"])}, json=body)
        assert response.status_code == 200, response.text
        assert call(who, "POST", f"/review/tasks/{tid}/submit").status_code == 200
        tids.append(tid)
    response = call("admin", "POST", "/adjudication-cases", json=dict(
        task_a=tids[0], task_b=tids[1], adjudicator=users["carla"]))
    assert response.status_code == 200, response.text
    cid = response.json()["id"]
    view = call("carla", "GET", f"/review/adjudication/{cid}").json()
    return cid, dict(case_id=cid, case_revision=view["case"]["rev"], events=[], contextual=[],
                     reasons=dict(review="Synthetic independent contract review"),
                     based_on={side: view["tracks"][side]["rev"] for side in "AB"})


def test_unversioned_adjudication_rejects_forged_new_human_provenance(api):
    store, call, _, _, _ = api
    cid, body = make_case(api)
    before = deepcopy(store.get(cid))
    fake_ref = dict(run_id="foreign-run", run_revision=99, event_id="foreign-event", event_sha256="f" * 64)
    event = dict(event_id="forged-human", start_ms=490, end_ms=510, label="filler_word", decision="event",
                 type_known=True, is_disfluent=True, contextual_state="disfluent", human_event_schema=HUMAN_EVENT_SCHEMA,
                 suggestion_id="foreign-event", proposal_ref=fake_ref, source_kind="human_verified_model_suggestion",
                 provenance=dict(confirmed=True, actor_id="forged-actor", proposal_ref=fake_ref, proposal_bounds=[0, 1000]))
    body["events"] = [event]
    response = call("carla", "POST", "/adjudications", json=body)
    if response.status_code == 200:
        frozen = call("admin", "POST", "/dataset-snapshots", json=dict(name="forged-probe", splits=["dev"]))
        assert frozen.status_code == 200, frozen.text
        print("FORGED_FROZEN_EVENT=" + json.dumps(frozen.json()["data"]["items"][0]["events"][0], sort_keys=True))
    assert response.status_code in (409, 422), response.text
    assert store.get(cid) == before and store.find("adjudication") == []


def test_negative_only_versioned_sources_cannot_adjudicate_through_legacy(api):
    store, call, _, _, _ = api
    cid, body = make_case(api, versioned=True)
    before = deepcopy(store.get(cid))
    response = call("carla", "POST", "/adjudications", json=body)
    if response.status_code == 200:
        frozen = call("admin", "POST", "/dataset-snapshots", json=dict(name="downgrade-probe", splits=["dev"]))
        assert frozen.status_code == 200, frozen.text
        item = frozen.json()["data"]["items"][0]
        print("DOWNGRADED_ITEM=" + json.dumps(item, sort_keys=True))
        evaluation.reference_view(frozen.json()["data"], {})
        print("LEGACY_EVALUATION_ACCEPTED_VERSIONED_NEGATIVE_SOURCES=True")
    assert response.status_code in (409, 422), response.text
    assert store.get(cid) == before


def test_new_semantic_annotation_cannot_be_downgraded_by_unversioned_patch(api):
    store, call, _, _, _ = api
    tid, view = task(api, mode="blind")
    original = dict(event_id="known-phenotype", start_ms=490, end_ms=510, label="filler_word", decision="event",
                    type_known=True, is_disfluent=None)
    saved = patch(api, tid, view, original)
    assert saved.status_code == 200, saved.text
    before = deepcopy(store.get(view["annotation"]["id"]))
    legacy = {key: original[key] for key in ("event_id", "start_ms", "end_ms", "label", "decision")}
    response = call("ana", "PATCH", f"/annotations/{before['id']}", headers={"If-Match": str(before["rev"])},
                    json=dict(events=[legacy], contextual=[], coverage=[[0, 1000]], actions=[]))
    if response.status_code == 200:
        assert call("ana", "POST", f"/review/tasks/{tid}/submit").status_code == 200
        frozen = call("admin", "POST", "/dataset-snapshots", json=dict(name="patch-downgrade-probe", include="submitted", splits=["dev"]))
        assert frozen.status_code == 200, frozen.text
        print("PATCH_DOWNGRADED_ITEM=" + json.dumps(frozen.json()["data"]["items"][0], sort_keys=True))
        evaluation.reference_view(frozen.json()["data"], {})
        print("LEGACY_EVALUATION_ACCEPTED_PATCH_DOWNGRADE=True")
    assert response.status_code in (409, 422), response.text
    assert store.get(before["id"]) == before


def test_reused_confirmation_cannot_rebind_proposal_under_previous_actor(api):
    store, call, users, _, run = api
    tid, view = task(api)
    event = mark(view["suggestions"][0])
    saved = patch(api, tid, view, event)
    assert saved.status_code == 200, saved.text
    before = deepcopy(saved.json())
    changed = deepcopy(run["data"])
    for cell in changed["result"]["cells"]:
        cell["type_probabilities"][0] = 0.1
        cell["type_decisions"][0] = "negative"
    changed["result"]["events"] = decode_cells(changed["result"]["cells"], recording_id=run["parent"])
    validate_cell_result(changed["result"])
    store.update(run["id"], changed)
    refreshed = call("ana", "GET", f"/review/tasks/{tid}").json()
    assert refreshed["suggestions"][0]["label"] != event["label"]
    payload = deepcopy(before["data"])
    payload["events"][0]["proposal_ref"] = refreshed["suggestions"][0]["proposal_ref"]
    # No new confirmation action: only substitute the new source reference while keeping the old action.
    response = call("admin", "PATCH", f"/annotations/{before['id']}",
                    headers={"If-Match": str(before["rev"])}, json=payload)
    if response.status_code == 200:
        actual = response.json()["data"]["events"][0]
        print("REBOUND_EVENT=" + json.dumps(actual, sort_keys=True))
        assert actual["provenance"]["actor_id"] == users["ana"]
        assert actual["provenance"]["proposal_ref"] != actual["proposal_ref"]
    assert response.status_code in (409, 422), response.text
    assert store.get(before["id"]) == before


"""Independent read-only U5a review: synthetic JS action extraction and HTTP only."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest
from tests.test_human_semantics import api, task, patch, mark, confirmation
from core.human_events import HUMAN_ANNOTATION_SCHEMA
from core.cell_evidence import cell_evidence
from core.cell_normalization import apply_cell_stats
from core.features_v2 import build_cell_features
from tests.test_cells_v2 import raw_signals

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT/'app/static/app.js').read_text()

def js_actions(script, initial):
    helpers = 'function humanProposal(s, id) {' + SOURCE.split('function humanProposal(s, id) {',1)[1].split('// END HUMAN SEMANTICS',1)[0]
    form = 'function fillForm(v = {}) {' + SOURCE.split('function fillForm(v = {}) {',1)[1].split('\n  function resetForm',1)[0]
    load = 'function load(key) {' + SOURCE.split('function load(key) {',1)[1].split('\n\n  function saveDraft',1)[0]
    save = 'function saveDraft() {' + SOURCE.split('function saveDraft() {',1)[1].split('\n  function removeKey',1)[0]
    onclick = 'view.onclick = e => {' + SOURCE.split('view.onclick = e => {',1)[1].split('\n  view.onchange',1)[0]
    prelude = """
let state=INITIAL, locked=false, editing=null, isCase=false, fromSuggestion=null, sel={};
let selected=''; let counter=0;
const values={}; for(const key of ['text','unc','prob','expr','dis','context','phenotype','func','spk','note'])values['#f-'+key]={value:''}; values['#f-known']={checked:false};
const choices=['cls:filler_word','cls:repetition','uncertain','not_evaluable','legit'].map(value=>({value,get checked(){return selected===value},set checked(x){if(x)selected=value;else if(selected===value)selected=''}}));
const view={querySelectorAll:()=>choices};
const $=id=>values[id];
const uid=()=> 'ui-'+(++counter), commit=(fn,action)=>{fn();if(action)state.actions.push(action)};
const showFields=()=>{},say=()=>{},draw=()=>{},toast=()=>{},resetForm=()=>{},renderSel=()=>{},fmt=x=>String(x);
const outsideRegion=()=>false,editingEvent=()=>editing?.type==='event'?state.events.find(e=>e.event_id===editing.id):null,chosen=()=>selected;
const reasons={}; function renderDiff(){};renderDiff.rows=[];
""".replace('INITIAL',json.dumps(initial))
    out=subprocess.run(['node','-e',prelude+helpers+form+load+save+onclick+'\n'+script+'\nconsole.log(JSON.stringify(state));'],capture_output=True,text=True,check=True)
    return json.loads(out.stdout)


def test_edit_existing_proposal_to_functional_keeps_binding(api):
    _,call,_,_,_=api
    tid,view=task(api)
    event=mark(view['suggestions'][0]);event.update(uncertainty=None)
    first=patch(api,tid,view,event)
    assert first.status_code==200,first.text
    saved=first.json()
    original=saved['data']['events'][0]
    state=js_actions("load('e:human-1');selected='legit';values['#f-expr'].value='eh';values['#f-dis'].value='false';saveDraft();",saved['data'])
    result=call('ana','PATCH',f"/annotations/{saved['id']}",headers={'If-Match':str(saved['rev'])},json={**state,'semantic_schema':HUMAN_ANNOTATION_SCHEMA})
    assert result.status_code==200,result.text
    final=result.json()['data']['contextual'][0]
    assert final['label']==original['label'] and final['type_known'] is True and final['is_disfluent'] is False
    assert [final['start_ms'],final['end_ms']]==[490,510]
    assert final.get('proposal_ref')==original['proposal_ref'],json.dumps(final,indent=2)
    assert final.get('suggestion_id')==original['suggestion_id']
    assert final['provenance'].get('proposal_bounds')==original['provenance']['proposal_bounds']


@pytest.mark.parametrize('decision,label,uncertainty',[('uncertain','filler_word','boundaries'),('not_evaluable',None,'audio')])
def test_take_ab_preserves_existing_uncertainty_decision(api,decision,label,uncertainty):
    store,call,users,_,_=api
    tids=[]
    for who in ('ana','beto'):
        tid,view=task(api,who,mode='blind')
        event=dict(event_id='source-human',start_ms=490,end_ms=510,label=label,decision=decision,uncertainty=uncertainty,type_known=label is not None,is_disfluent=None)
        saved=patch(api,tid,view,event,who=who)
        assert saved.status_code==200,saved.text
        assert call(who,'POST',f'/review/tasks/{tid}/submit').status_code==200
        tids.append(tid)
    create=call('admin','POST','/adjudication-cases',json=dict(task_a=tids[0],task_b=tids[1],adjudicator=users['carla']))
    assert create.status_code==200,create.text
    cid=create.json()['id']; view=call('carla','GET',f'/review/adjudication/{cid}').json()
    src=view['tracks']['A']['events'][0]
    script="isCase=true;renderDiff.rows=[[SOURCE,null]];view.onclick({target:{closest:()=>({dataset:{take:'0:A'}})}});values['#f-known'].checked=SOURCE.type_known;saveDraft();".replace('SOURCE',json.dumps(src))
    state=js_actions(script,dict(events=[],contextual=[],actions=[]))
    adjud=call('carla','POST','/adjudications',json=dict(semantic_schema=HUMAN_ANNOTATION_SCHEMA,case_id=cid,case_revision=view['case']['rev'],events=state['events'],contextual=state['contextual'],actions=state['actions'],reasons=dict(review='Synthetic review retains source decision'),based_on={k:view['tracks'][k]['rev'] for k in 'AB'}))
    assert adjud.status_code==200,adjud.text
    final=store.get(adjud.json()['id'])['data']['events'][0]
    assert final['decision']==decision and final['uncertainty']==uncertainty, json.dumps(dict(source=src,final=final),indent=2)


def test_legacy_contextual_draft_remains_savable_without_bulk_reannotation(api):
    store,call,_,_,_=api
    tid,view=task(api,mode='blind')
    legacy=dict(events=[],contextual=[dict(start_ms=100,end_ms=200,expression='este',is_disfluent=False,function='demonstrative')],coverage=[[0,1000]],actions=[])
    old=call('ana','PATCH',f"/annotations/{view['annotation']['id']}",headers={'If-Match':str(view['annotation']['rev'])},json=legacy)
    assert old.status_code==200,old.text
    loaded=call('ana','GET',f'/review/tasks/{tid}').json()['annotation']
    before=deepcopy(store.get(loaded['id']))
    # Exact current UI save envelope after an unrelated edit such as reviewed coverage.
    response=call('ana','PATCH',f"/annotations/{loaded['id']}",headers={'If-Match':str(loaded['rev'])},json=js_save_payload(loaded['data']))
    if response.status_code!=200:
        assert store.get(loaded['id'])==before
    assert response.status_code==200,response.text


def test_effective_unavailable_has_reason_when_acquired_columns_are_untrained():
    f=build_cell_features(raw_signals(1000),[],1000)
    f['ac_mask'][:,:,0]=False
    f['ac_mask'][:,:,2:]=False
    widths=dict(ac=5,pros=8,ling=12,kin=12)
    stats={key:dict(mean=[0.]*width,std=[1.]*width,count=[0]*width) for key,width in widths.items()}
    stats['ac']['count'][0]=20
    caps=dict(modalities=dict(ac=True,pros=False,ling=False,kin=False))
    normalized=apply_cell_stats(f,stats,mode='audio_text',modalities=caps['modalities'])
    report=cell_evidence(f,mode='audio_text',capabilities=caps,stats=stats)
    assert report[0]['acquired']['ac']['available'] is True
    assert report[0]['effective']['ac']['available'] is False and not normalized['ac_mask'][0].any()
    assert report[0]['effective']['ac']['reasons'],json.dumps(report[0],indent=2)


def js_save_payload(state, *, client_op='ui-unrelated-coverage'):
    """Execute the real async autosave function, not a hand-built version envelope."""
    save = 'async function save() {' + SOURCE.split('async function save() {', 1)[1].split('\n  const pending =', 1)[0]
    helpers = 'function humanProposal(s, id) {' + SOURCE.split('function humanProposal(s, id) {', 1)[1].split('// END HUMAN SEMANTICS', 1)[0]
    stubs = """
let state=INITIAL, saving=false, again=false, retry=null, op=CLIENT_OP, rev=1;
const ann={id:'synthetic'}, pendingKey='p', local={del:()=>{}}, status=()=>{}, tel=()=>{}, say=()=>{}, route=()=>{}, $=()=>({});
let captured; const api=async (url, request)=>{captured=request.body;return {rev:2}};
""".replace('INITIAL', json.dumps(state)).replace('CLIENT_OP', json.dumps(client_op))
    script = stubs + helpers + save + "\nsave().then(()=>console.log(JSON.stringify(captured)));"
    return json.loads(subprocess.run(['node','-e',script],capture_output=True,text=True,check=True).stdout)


def test_legacy_actual_ui_edit_and_autosave_does_not_implicitly_confirm(api):
    store,call,_,_,_=api;tid,view=task(api,mode='blind')
    legacy=dict(events=[],contextual=[dict(start_ms=100,end_ms=200,expression='este',is_disfluent=False,function='demonstrative')],coverage=[[0,1000]],actions=[])
    old=call('ana','PATCH',f"/annotations/{view['annotation']['id']}",headers={'If-Match':str(view['annotation']['rev'])},json=legacy).json()
    state=js_actions("load('c:0');values['#f-note'].value='historical edit';saveDraft();",old['data'])
    response=call('ana','PATCH',f"/annotations/{old['id']}",headers={'If-Match':str(old['rev'])},json=js_save_payload(state))
    assert response.status_code==200,response.text
    saved=response.json()['data'];assert not saved.get('semantic_schema')
    item=saved['contextual'][0];assert item['note']=='historical edit' and item['expression']=='este' and item['is_disfluent'] is False
    assert item.get('type_known') is None and not any(a.get('type')=='confirm_human_event' for a in saved['actions'])


def test_explicit_legacy_upgrade_preserves_original_revision_and_never_invents_knowledge(api):
    store,call,_,_,_=api;tid,view=task(api,mode='blind')
    legacy=dict(events=[],contextual=[dict(start_ms=100,end_ms=200,expression='este',is_disfluent=False,function='demonstrative')],coverage=[[0,1000]],actions=[])
    old=call('ana','PATCH',f"/annotations/{view['annotation']['id']}",headers={'If-Match':str(view['annotation']['rev'])},json=legacy).json()
    body=dict(expected_annotation_revision=old['rev'],confirm_upgrade=True,reason='Explicit synthetic legacy transition')
    stale=call('ana','POST',f"/annotations/{old['id']}/upgrade-semantics",json={**body,'expected_annotation_revision':old['rev']-1})
    assert stale.status_code==409,stale.text
    assert store.get(old['id'])==old
    response=call('ana','POST',f"/annotations/{old['id']}/upgrade-semantics",json=body)
    assert response.status_code==200,response.text
    upgraded=response.json();data=upgraded['data'];item=data['contextual'][0]
    assert data['semantic_schema']==HUMAN_ANNOTATION_SCHEMA
    assert item['event_id'] and item['legacy_original']==old['data']['contextual'][0]
    assert [item['start_ms'],item['end_ms']]==[100,200] and item['expression']=='este'
    assert item['type_known'] is False and item['is_disfluent'] is None and item['provenance']['confirmed'] is False
    assert data['semantic_transition']['source_revision']==old['rev']
    assert any(h['rev']==old['rev'] and h['data']==old['data'] for h in store.history(old['id']))
    # New actions are now explicit, bound to this upgraded revision and do not delete raw history.
    state=js_actions("load('c:0');values['#f-phenotype'].value='filler_word';values['#f-known'].checked=true;values['#f-dis'].value='false';saveDraft();",data)
    saved=call('ana','PATCH',f"/annotations/{old['id']}",headers={'If-Match':str(upgraded['rev'])},json=js_save_payload(state))
    assert saved.status_code==200,saved.text
    final=saved.json()['data']['contextual'][0]
    assert final['type_known'] is True and final['is_disfluent'] is False and final['legacy_original']==item['legacy_original']


@pytest.mark.parametrize('claim',[
    {'contextual_state':'functional'}, {'type_known':False}, {'human_event_schema':'unknown-v99'},
    {'proposal_ref':{'run_id':'foreign'}}, {'source_kind':'human_verified_model_suggestion'},
    {'provenance':{'confirmed':True,'actor_id':'forged-actor'}},
])
def test_unversioned_residual_claims_reject_atomically_on_patch_and_adjudication(api,claim):
    store,call,_,_,_=api
    tid,view=task(api,mode='blind');old=deepcopy(store.get(view['annotation']['id']))
    event=dict(event_id='legacy',start_ms=100,end_ms=200,label='filler_word',**claim)
    response=call('ana','PATCH',f"/annotations/{old['id']}",headers={'If-Match':str(old['rev'])},json=dict(events=[event]))
    assert response.status_code==422,response.text
    assert store.get(old['id'])==old
    cid,body=make_case(api);before=deepcopy(store.get(cid));body['events']=[event]
    response=call('carla','POST','/adjudications',json=body)
    assert response.status_code==422,response.text
    assert store.get(cid)==before and not store.find('adjudication')


def test_source_reconfirmation_changes_actor_only_with_new_source_attestation(api):
    store,call,users,_,run=api;tid,view=task(api)
    event=mark(view['suggestions'][0]);saved=patch(api,tid,view,event).json()
    changed=deepcopy(run['data']);changed['synthetic_revision']='renewed source'
    store.update(run['id'],changed)
    current=call('ana','GET',f'/review/tasks/{tid}').json()['suggestions'][0]
    body=deepcopy(saved['data']);body.pop('client_op',None)
    body['events'][0]['proposal_ref']=current['proposal_ref']
    action=confirmation(body['events'][0]);action['reviewed_proposal']={key:body['events'][0].get(key) for key in ('suggestion_id','proposal_ref')}
    body['actions'].append(action)
    response=call('admin','PATCH',f"/annotations/{saved['id']}",headers={'If-Match':str(saved['rev'])},json=body)
    assert response.status_code==200,response.text
    final=response.json()['data']['events'][0]
    assert final['provenance']['actor_id']==users['admin'] and final['provenance']['proposal_ref']==current['proposal_ref']
    assert final['provenance']['proposal_ref']!=saved['data']['events'][0]['provenance']['proposal_ref']


@pytest.mark.parametrize('revision',[True,1.0])
def test_proposal_identity_revision_is_strict_not_python_numeric_equality(api,revision):
    store,call,_,_,_=api;tid,view=task(api);event=mark(view['suggestions'][0]);event['proposal_ref']['run_revision']=revision
    before=deepcopy(store.get(view['annotation']['id']))
    response=patch(api,tid,view,event)
    assert response.status_code==422,response.text
    assert store.get(before['id'])==before


def test_explicit_upgrade_can_confirm_adjudicate_freeze_without_losing_originals(api):
    store,call,users,_,_=api;tids=[];saved=[]
    for who in ('ana','beto'):
        tid,view=task(api,who,mode='blind')
        old=call(who,'PATCH',f"/annotations/{view['annotation']['id']}",headers={'If-Match':str(view['annotation']['rev'])},json=dict(events=[],contextual=[dict(start_ms=100,end_ms=200,expression='este',is_disfluent=False)],coverage=[[0,1000]])).json()
        upgraded=call(who,'POST',f"/annotations/{old['id']}/upgrade-semantics",json=dict(expected_annotation_revision=old['rev'],confirm_upgrade=True,reason='Preserve a synthetic historical source')).json()
        event=deepcopy(upgraded['data']['contextual'][0]);event.update(label='filler_word',type_known=True,is_disfluent=False)
        body={**upgraded['data'],'contextual':[event],'actions':[*upgraded['data']['actions'],confirmation(event)]}
        response=call(who,'PATCH',f"/annotations/{old['id']}",headers={'If-Match':str(upgraded['rev'])},json=body)
        assert response.status_code==200,response.text
        saved.append(response.json()['data']['contextual'][0])
        assert call(who,'POST',f'/review/tasks/{tid}/submit').status_code==200
        tids.append(tid)
    cid=call('admin','POST','/adjudication-cases',json=dict(task_a=tids[0],task_b=tids[1],adjudicator=users['carla'])).json()['id']
    view=call('carla','GET',f'/review/adjudication/{cid}').json();event=saved[0]
    body=dict(semantic_schema=HUMAN_ANNOTATION_SCHEMA,case_id=cid,case_revision=view['case']['rev'],events=[],contextual=[event],actions=[confirmation(event)],based_on={k:view['tracks'][k]['rev'] for k in 'AB'})
    response=call('carla','POST','/adjudications',json=body)
    assert response.status_code==200,response.text
    frozen=call('admin','POST','/dataset-snapshots',json=dict(name='upgraded-synthetic',splits=['dev']))
    assert frozen.status_code==200,frozen.text
    final=frozen.json()['data']['items'][0]['contextual'][0]
    assert final['legacy_original']==event['legacy_original'] and final['type_known'] is True and final['is_disfluent'] is False


def test_legacy_snapshot_cannot_certify_retained_new_claims_without_schema(api):
    store,call,_,_,_=api;tid,view=task(api,mode='blind')
    event=dict(event_id='injected',start_ms=100,end_ms=200,label='filler_word',decision='event',type_known=True,is_disfluent=True)
    store.update(view['annotation']['id'],dict(events=[event],contextual=[],coverage=[[0,1000]]),status='submitted')
    store.update(tid,status='submitted');before=deepcopy(store.get(view['annotation']['id']))
    response=call('admin','POST','/dataset-snapshots',json=dict(name='bad-retained-markers',include='submitted',splits=['dev']))
    assert response.status_code==422,response.text
    assert store.get(before['id'])==before and not store.find('snapshot')


@pytest.mark.parametrize('data',[
    {'events':None}, {'contextual':{}}, {'events':[None]}, {'actions':None}, {'actions':[1]},
])
def test_semantic_marker_guard_rejects_malformed_import_containers_with_clear_error(data):
    from core.human_events import annotation_semantics
    with pytest.raises(ValueError,match='events|contextual|actions'):
        annotation_semantics(data)
