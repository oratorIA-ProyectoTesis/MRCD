"""Read-only synthetic UI remediation probes; run only through canonical U5a wrapper."""
import ast
import json
from copy import deepcopy
from pathlib import Path

from tests.test_human_semantics import api, task
from tests.test_human_semantics_review import SOURCE, js_actions, js_save_payload
from core.human_events import HUMAN_ANNOTATION_SCHEMA


def legacy_annotation(api, who='ana', *, contextual=False):
    _, call, _, _, _ = api
    tid, view = task(api, who, mode='blind')
    item = dict(start_ms=100, end_ms=300, expression='este', is_disfluent=False) if contextual else dict(
        event_id='old-event', start_ms=100, end_ms=300, label='filler_word', decision='event')
    body = dict(events=[] if contextual else [item], contextual=[item] if contextual else [], coverage=[[0, 1000]], actions=[])
    result = call(who, 'PATCH', f"/annotations/{view['annotation']['id']}", headers={'If-Match':str(view['annotation']['rev'])}, json=body)
    assert result.status_code == 200, result.text
    return tid, result.json()


def upgrade(api, annotation, who='ana'):
    result = api[1](who, 'POST', f"/annotations/{annotation['id']}/upgrade-semantics", json=dict(
        expected_annotation_revision=annotation['rev'], confirm_upgrade=True, reason='Explicit synthetic UI review transition'))
    assert result.status_code == 200, result.text
    return result.json()


def save(api, annotation, state, who='ana'):
    return api[1](who, 'PATCH', f"/annotations/{annotation['id']}", headers={'If-Match':str(annotation['rev'])}, json=js_save_payload(state, client_op=f"lineage-edit-{annotation['rev']}"))




def test_historical_context_to_event_retains_usable_ui_identity(api):
    _, annotation = legacy_annotation(api, contextual=True)
    state = js_actions("load('c:0');selected='cls:filler_word';saveDraft();", annotation['data'])
    response = save(api, annotation, state)
    assert response.status_code == 200, response.text
    event = response.json()['data']['events'][0]
    assert isinstance(event.get('event_id'), str) and event['event_id'], event
    # The list's actual edit key must still resolve after saving/reloading.
    js_actions("load('e:' + state.events[0].event_id);", response.json()['data'])


def test_historical_drag_stays_in_historical_family_and_saves(api):
    _, annotation = legacy_annotation(api)
    invalidation = 'function invalidateHuman(mark) {' + SOURCE.split('function invalidateHuman(mark) {', 1)[1].split("\n  regions.on('region-created'", 1)[0]
    handler = "regions.on('region-updated', r => {" + SOURCE.split("regions.on('region-updated', r => {", 1)[1].split("\n  regions.on('region-clicked'", 1)[0]
    script = "const handlers={};const regions={on:(name,fn)=>handlers[name]=fn};const G=x=>Math.round(x*1000);" + invalidation + handler
    state = js_actions(script + ";\nhandlers['region-updated']({id:'old-event',start:0.11,end:0.31});", annotation['data'])
    response = save(api, annotation, state)
    assert response.status_code == 200, response.text
    assert not response.json()['data'].get('semantic_schema')
    assert response.json()['data']['events'][0]['start_ms'] == 110


def test_upgraded_mark_can_split_without_losing_immutable_original(api):
    _, annotation = legacy_annotation(api)
    upgraded = upgrade(api, annotation)
    original = deepcopy(upgraded['data']['events'][0]['legacy_original'])
    invalidation = 'function invalidateHuman(mark) {' + SOURCE.split('function invalidateHuman(mark) {', 1)[1].split("\n  regions.on('region-created'", 1)[0]
    split = "$('#split-ev').onclick = () => {" + SOURCE.split("$('#split-ev').onclick = () => {", 1)[1].split("\n  $('#merge-ev').onclick", 1)[0]
    script = "values['#split-ev']={};const now=()=>200;" + invalidation + split
    state = js_actions(script + ";\nload('e:'+state.events[0].event_id);values['#split-ev'].onclick();", upgraded['data'])
    response = save(api, upgraded, state)
    assert response.status_code == 200, response.text
    events = response.json()['data']['events']
    assert len(events) == 2 and all(event.get('legacy_original') == original for event in events)


def test_actual_take_ab_of_upgraded_mark_keeps_original_through_freeze(api):
    store, call, users, _, _ = api
    tids = []
    for who in ('ana', 'beto'):
        tid, annotation = legacy_annotation(api, who)
        upgraded = upgrade(api, annotation, who)
        assert call(who, 'POST', f'/review/tasks/{tid}/submit').status_code == 200
        tids.append(tid)
    created = call('admin', 'POST', '/adjudication-cases', json=dict(task_a=tids[0], task_b=tids[1], adjudicator=users['carla']))
    assert created.status_code == 200, created.text
    cid = created.json()['id']
    view = call('carla', 'GET', f'/review/adjudication/{cid}').json()
    source = view['tracks']['A']['events'][0]
    script = "isCase=true;renderDiff.rows=[[INPUT,null]];view.onclick({target:{closest:()=>({dataset:{take:'0:A'}})}});saveDraft();".replace('INPUT', json.dumps(source))
    state = js_actions(script, dict(events=[], contextual=[], actions=[]))
    result = call('carla', 'POST', '/adjudications', json=dict(semantic_schema=HUMAN_ANNOTATION_SCHEMA,
        case_id=cid, case_revision=view['case']['rev'], events=state['events'], contextual=state['contextual'], actions=state['actions'],
        based_on={side:view['tracks'][side]['rev'] for side in 'AB'}, reasons=dict(review='Synthetic actual UI source preservation')))
    assert result.status_code == 200, result.text
    frozen = call('admin', 'POST', '/dataset-snapshots', json=dict(name='actual-upgraded-ui', splits=['dev']))
    assert frozen.status_code == 200, frozen.text
    final = frozen.json()['data']['items'][0]['events'][0]
    assert final.get('legacy_original') == source['legacy_original'], final


import pytest
from tests.test_human_semantics import confirmation


def actual_mutation(state, operation):
    invalidation = 'function invalidateHuman(mark) {' + SOURCE.split('function invalidateHuman(mark) {',1)[1].split("\n  regions.on('region-created'",1)[0]
    if operation == 'drag':
        handler = "regions.on('region-updated', r => {" + SOURCE.split("regions.on('region-updated', r => {",1)[1].split("\n  regions.on('region-clicked'",1)[0]
        code="const handlers={};const regions={on:(name,fn)=>handlers[name]=fn};const G=x=>Math.round(x*1000);"+invalidation+handler+";\nhandlers['region-updated']({id:state.events[0].event_id,start:0.11,end:0.31});"
    else:
        split="$('#split-ev').onclick = () => {"+SOURCE.split("$('#split-ev').onclick = () => {",1)[1].split("\n  $('#merge-ev').onclick",1)[0]
        merge="$('#merge-ev').onclick = () => {"+SOURCE.split("$('#merge-ev').onclick = () => {",1)[1].split("\n  $('#play').onclick",1)[0]
        code="values['#split-ev']={};values['#merge-ev']={};const now=()=>200;"+invalidation+split+";\n"+merge+";\nload('e:'+state.events[0].event_id);values['#split-ev'].onclick();"
        if operation=='merge':code+="load('e:'+state.events[0].event_id);values['#merge-ev'].onclick();"
    return js_actions(code,state)


@pytest.mark.parametrize('versioned',[False,True])
@pytest.mark.parametrize('operation',['drag','split','merge'])
def test_actual_mutations_keep_family_and_revoke_only_versioned_confirmations(api,versioned,operation):
    _,annotation=legacy_annotation(api)
    if versioned:
        annotation=upgrade(api,annotation)
        data=deepcopy(annotation['data']);event=data['events'][0];event.update(type_known=True,is_disfluent=True)
        data['actions'].append(confirmation(event))
        result=save(api,annotation,data);assert result.status_code==200,result.text
        annotation=result.json()
    state=actual_mutation(annotation['data'],operation)
    result=save(api,annotation,state);assert result.status_code==200,result.text
    final=result.json()['data']
    assert bool(final.get('semantic_schema')) is versioned
    assert any(a.get('type')=='revoke_human_confirmation' for a in final['actions']) is versioned
    assert all(e.get('type_known') is (False if versioned else None) and e.get('is_disfluent') is None for e in final['events'])
    if versioned:assert all(not e['provenance']['confirmed'] for e in final['events'])


@pytest.mark.parametrize('existing_id',[None,'retained-context-id'])
@pytest.mark.parametrize('versioned',[False,True])
def test_context_to_event_identity_and_original_field_preservation(api,existing_id,versioned):
    store,_,_,_,_=api;_,annotation=legacy_annotation(api,contextual=True)
    if existing_id:
        data=deepcopy(annotation['data']);data['contextual'][0]['event_id']=existing_id
        result=save(api,annotation,data);assert result.status_code==200,result.text;annotation=result.json()
    if versioned:annotation=upgrade(api,annotation)
    source=deepcopy(annotation['data']['contextual'][0]);state=js_actions("load('c:0');selected='cls:filler_word';values['#f-note'].value='retained note';saveDraft();",annotation['data'])
    result=save(api,annotation,state);assert result.status_code==200,result.text
    event=result.json()['data']['events'][0]
    assert event['event_id'] and (not source.get('event_id') or event['event_id']==source['event_id'])
    assert event['expression']==source['expression'] and event['note']=='retained note'
    if versioned:assert event['legacy_original']==source['legacy_original'] and event['source_lineage']


@pytest.mark.parametrize('tamper',['foreign','stale','hash','original','missing','bool_revision'])
def test_child_lineage_rejects_forged_foreign_or_stale_parent_atomically(api,tamper):
    store,call,_,_,_=api;_,ann=legacy_annotation(api);ann=upgrade(api,ann)
    data=deepcopy(ann['data']);child=deepcopy(data['events'][0]);child.update(event_id='new-child',parent_event_id=child['event_id'],start_ms=200)
    assert isinstance(child.get('source_lineage'),dict),'server must issue independently verified parent lineage'
    if tamper=='foreign':
        _,other=legacy_annotation(api,'beto');other=upgrade(api,other,'beto');child['source_lineage']=other['data']['events'][0]['source_lineage']
    elif tamper=='stale':
        current=deepcopy(ann['data']);current['events'][0]['end_ms']=310
        result=save(api,ann,current);assert result.status_code==200,result.text;ann=result.json()
    elif tamper=='hash':child['source_lineage']['event_sha256']='f'*64
    elif tamper=='original':child['legacy_original']['start_ms']=99
    elif tamper=='missing':child.pop('source_lineage')
    else:child['source_lineage']['annotation_revision']=True
    data['events']=[child];before=deepcopy(store.get(ann['id']))
    result=save(api,ann,data)
    assert result.status_code in (409,422),result.text
    assert store.get(ann['id'])==before


def test_split_child_then_actual_ui_autosave_rebases_lineage_without_confirmation(api):
    _,ann=legacy_annotation(api);ann=upgrade(api,ann)
    first=save(api,ann,actual_mutation(ann['data'],'split'));assert first.status_code==200,first.text
    ann=first.json();state=actual_mutation(ann['data'],'drag');second=save(api,ann,state)
    assert second.status_code==200,second.text
    assert len(second.json()['data']['events'])==2 and all(not e['provenance']['confirmed'] for e in second.json()['data']['events'])


def test_existing_original_cannot_be_silently_dropped_on_edit(api):
    store,_,_,_,_=api;_,ann=legacy_annotation(api);ann=upgrade(api,ann)
    state=deepcopy(ann['data']);state['events'][0].pop('legacy_original');state['events'][0].pop('source_lineage')
    before=deepcopy(store.get(ann['id']));result=save(api,ann,state)
    assert result.status_code==409,result.text
    assert store.get(ann['id'])==before


def test_actual_autosave_refreshes_only_references_for_next_pending_edit(api):
    _,ann=legacy_annotation(api);ann=upgrade(api,ann)
    state=actual_mutation(ann['data'],'split');before=deepcopy(state)
    result=save(api,ann,state);assert result.status_code==200,result.text
    saved=result.json();state=js_actions('refreshHumanLineages(state, INPUT);'.replace('INPUT',json.dumps(saved['data'])),state)
    assert all(e['source_lineage']['annotation_revision']==saved['rev'] for e in state['events'])
    assert [e['provenance'] for e in state['events']]==[e['provenance'] for e in before['events']]
    state['events'][1]['end_ms']=290
    again=save(api,saved,state);assert again.status_code==200,again.text
    assert all(not e['provenance']['confirmed'] for e in again.json()['data']['events'])
