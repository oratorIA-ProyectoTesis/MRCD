"""Independent synthetic residual check; canonical U5a wrapper execution only."""
from copy import deepcopy
import json
import subprocess

import pytest

from tests.test_human_semantics import api
from tests.test_human_lineage import legacy_annotation, upgrade
from tests.test_human_semantics_review import SOURCE, js_save_payload


def actual_history_action(initial, operation, *, undo_stack=None, server_saved=None):
    """Real commit, mutation, reference refresh and Undo; DOM/time/transport are stubs."""
    helpers = 'function humanProposal(s, id) {' + SOURCE.split('function humanProposal(s, id) {', 1)[1].split('// END HUMAN SEMANTICS', 1)[0]
    commit = 'function commit(mutate, action) {' + SOURCE.split('function commit(mutate, action) {', 1)[1].split('\n\n  function coverage()', 1)[0]
    invalidate = 'function invalidateHuman(mark) {' + SOURCE.split('function invalidateHuman(mark) {', 1)[1].split("\n  regions.on('region-created'", 1)[0]
    drag = "regions.on('region-updated', r => {" + SOURCE.split("regions.on('region-updated', r => {", 1)[1].split("\n  regions.on('region-clicked'", 1)[0]
    split = "$('#split-ev').onclick = () => {" + SOURCE.split("$('#split-ev').onclick = () => {", 1)[1].split("\n  $('#merge-ev').onclick", 1)[0]
    remove = 'function removeKey(key) {' + SOURCE.split('function removeKey(key) {',1)[1].split('\n  function markIn()',1)[0]
    merge = "$('#merge-ev').onclick = () => {" + SOURCE.split("$('#merge-ev').onclick = () => {",1)[1].split("\n  $('#play').onclick",1)[0]
    undo_handler = "$('#undo').onclick = () => {" + SOURCE.split("$('#undo').onclick = () => {", 1)[1].split("\n  $('#submit').onclick", 1)[0]
    prelude = """
let state=INITIAL, undo=UNDO_STACK, locked=false, isCase=false, counter=0, editing=null;
const handlers={}, values={'#split-ev':{},'#merge-ev':{},'#undo':{}}, $=key=>values[key];
const regions={on:(name,fn)=>handlers[name]=fn}, G=x=>Math.round(x*1000);
const uid=()=> 'independent-child-'+(++counter), now=()=>200;
const editingEvent=()=>state.events[0], activity=()=>{}, draw=()=>{}, scheduleSave=()=>{}, resetForm=()=>{}, say=()=>{}, toast=()=>{};
""".replace('INITIAL', json.dumps(initial)).replace('UNDO_STACK', json.dumps(undo_stack or []))
    if operation == 'undo':
        action = 'refreshHumanLineages(state, SERVER); values["#undo"].onclick();'.replace('SERVER', json.dumps(server_saved))
    elif operation == 'drag':
        action = "handlers['region-updated']({id:state.events[0].event_id,start:0.11,end:0.31});"
    elif operation == 'delete':
        action = "removeKey('e:'+state.events[0].event_id);"
    elif operation == 'merge':
        action = "values['#merge-ev'].onclick();"
    else:
        action = "values['#split-ev'].onclick();"
    code = '\n;\n'.join([prelude, helpers, commit, invalidate, drag, split, remove, merge, undo_handler, action,
                            'console.log(JSON.stringify({state,undo}));'])
    return json.loads(subprocess.run(['node', '-e', code], check=True, text=True, capture_output=True).stdout)


@pytest.mark.parametrize('operation', ['drag', 'split'])
def test_actual_undo_after_successful_autosave_retains_savable_original(api, operation):
    store, call, _, _, _ = api
    _, ann = legacy_annotation(api)
    ann = upgrade(api, ann)
    original = deepcopy(ann['data'])
    changed = actual_history_action(ann['data'], operation)
    assert len(changed['undo']) == 1 and json.loads(changed['undo'][0]) == original
    first = call('ana', 'PATCH', f"/annotations/{ann['id']}", headers={'If-Match': str(ann['rev'])},
                 json=js_save_payload(changed['state'], client_op='independent-change'))
    assert first.status_code == 200, first.text
    saved = first.json()
    undone = actual_history_action(changed['state'], 'undo', undo_stack=changed['undo'], server_saved=saved['data'])
    assert undone['state']['events'] == original['events']
    assert undone['state']['events'][0]['legacy_original'] == original['events'][0]['legacy_original']
    assert undone['state']['events'][0]['provenance']['confirmed'] is False
    before = deepcopy(store.get(ann['id']))
    result = call('ana', 'PATCH', f"/annotations/{ann['id']}", headers={'If-Match': str(saved['rev'])},
                  json=js_save_payload(undone['state'], client_op='independent-undo'))
    if result.status_code != 200:
        assert store.get(ann['id']) == before
    assert result.status_code == 200, result.text
    final = result.json()['data']['events'][0]
    assert final['legacy_original'] == original['events'][0]['legacy_original']
    assert final['provenance']['confirmed'] is False
    assert final['type_known'] is False and final['is_disfluent'] is None


@pytest.mark.parametrize('operation',['delete','merge'])
def test_actual_undo_with_source_absent_after_save(api,operation):
    store,call,_,_,_=api;_,ann=legacy_annotation(api)
    if operation=='merge':
        data=deepcopy(ann['data']);second=deepcopy(data['events'][0]);second.update(event_id='second-origin',start_ms=400,end_ms=600);data['events'].append(second)
        response=call('ana','PATCH',f"/annotations/{ann['id']}",headers={'If-Match':str(ann['rev'])},json={**data,'client_op':'seed-second'})
        assert response.status_code==200,response.text;ann=response.json()
    ann=upgrade(api,ann);original=deepcopy(ann['data']);changed=actual_history_action(original,operation)
    first=call('ana','PATCH',f"/annotations/{ann['id']}",headers={'If-Match':str(ann['rev'])},json=js_save_payload(changed['state'],client_op='remove-source'))
    assert first.status_code==200,first.text;saved=first.json()
    undone=actual_history_action(changed['state'],'undo',undo_stack=changed['undo'],server_saved=saved['data'])
    assert undone['state']['events']==original['events']
    response=call('ana','PATCH',f"/annotations/{ann['id']}",headers={'If-Match':str(saved['rev'])},json=js_save_payload(undone['state'],client_op='restore-source'))
    assert response.status_code==200,response.text
    assert len(response.json()['data']['events'])==len(original['events'])
    assert all(not e['provenance']['confirmed'] and not e['type_known'] and e['is_disfluent'] is None for e in response.json()['data']['events'])


from tests.test_human_semantics import confirmation
from tests.test_human_semantics_review import js_actions
from app.store import Store


def deleted_source(api, *, confirmed=False):
    store,call,_,_,_=api;_,ann=legacy_annotation(api);ann=upgrade(api,ann)
    if confirmed:
        body=deepcopy(ann['data']);event=body['events'][0];event.update(type_known=True,is_disfluent=True)
        action=confirmation(event);action['confirmation_id']='prior-human-confirmation';body['actions'].append(action)
        result=call('ana','PATCH',f"/annotations/{ann['id']}",headers={'If-Match':str(ann['rev'])},json={**body,'client_op':'confirm-source'})
        assert result.status_code==200,result.text;ann=result.json()
    changed=actual_history_action(ann['data'],'delete')
    result=call('ana','PATCH',f"/annotations/{ann['id']}",headers={'If-Match':str(ann['rev'])},json=js_save_payload(changed['state'],client_op='delete-source'))
    assert result.status_code==200,result.text;deleted=result.json()
    assert len(deleted['data'].get('lineage_restore',[]))==1,'deleted origin must have a registered revision-bound restoration handle'
    body=deepcopy(ann['data']);body['events'][0]['source_lineage']=deepcopy(deleted['data']['lineage_restore'][0]['source_lineage']);body['client_op']='restore-source'
    return ann,deleted,body,changed


@pytest.mark.parametrize('tamper',['hash','foreign','unregistered','stale','original','revision_bool','source_revision'])
def test_restore_registry_rejects_forged_unregistered_foreign_stale_atomically(api,tamper):
    store,call,_,_,_=api;original,ann,body,_=deleted_source(api)
    ref=body['events'][0]['source_lineage']
    if tamper=='hash':ref['event_sha256']='f'*64
    elif tamper=='foreign':
        _,other,foreign,_=deleted_source(api)
        body['events'][0]['source_lineage']=foreign['events'][0]['source_lineage']
    elif tamper=='unregistered':ref['event_id']='nonregistered-event'
    elif tamper=='stale':ref['current_revision']-=1
    elif tamper=='original':body['events'][0]['legacy_original']['start_ms']=99
    elif tamper=='revision_bool':ref['current_revision']=True
    else:ref['source_revision']=1
    before=deepcopy(store.get(ann['id']));response=call('ana','PATCH',f"/annotations/{ann['id']}",headers={'If-Match':str(ann['rev'])},json=body)
    assert response.status_code in (409,422),response.text
    assert store.get(ann['id'])==before


def test_restore_consumption_replay_permissions_and_new_confirmation(api):
    store,call,users,_,_=api;original,ann,body,_=deleted_source(api,confirmed=True)
    before=deepcopy(store.get(ann['id']))
    assert call('beto','PATCH',f"/annotations/{ann['id']}",headers={'If-Match':str(ann['rev'])},json=body).status_code==403
    assert store.get(ann['id'])==before
    first=call('ana','PATCH',f"/annotations/{ann['id']}",headers={'If-Match':str(ann['rev'])},json=body)
    assert first.status_code==200,first.text;saved=first.json();event=saved['data']['events'][0]
    assert event['legacy_original']==original['data']['events'][0]['legacy_original'] and not event['provenance']['confirmed']
    assert event['type_known'] is False and event['is_disfluent'] is None and event['provenance']['actor_id']==users['ana']
    assert saved['data']['lineage_restore']==[]
    retry=call('ana','PATCH',f"/annotations/{ann['id']}",headers={'If-Match':str(ann['rev'])},json=body)
    assert retry.status_code==200 and retry.json()==saved
    replay=call('ana','PATCH',f"/annotations/{ann['id']}",headers={'If-Match':str(saved['rev'])},json={**body,'client_op':'replay-consumed-handle'})
    assert replay.status_code==409,replay.text
    old=deepcopy(saved['data']);old['events'][0].update(type_known=True,is_disfluent=True)
    old['actions']=original['data']['actions'];old['client_op']='replay-old-confirmation'
    assert call('ana','PATCH',f"/annotations/{ann['id']}",headers={'If-Match':str(saved['rev'])},json=old).status_code==409
    state=js_actions("load('e:'+state.events[0].event_id);values['#f-known'].checked=true;values['#f-context'].value='true';saveDraft();",saved['data'])
    fresh=call('ana','PATCH',f"/annotations/{ann['id']}",headers={'If-Match':str(saved['rev'])},json=js_save_payload(state,client_op='new-explicit-human-review'))
    assert fresh.status_code==200,fresh.text
    assert fresh.json()['data']['events'][0]['type_known'] is True and fresh.json()['data']['events'][0]['is_disfluent'] is True


def test_restore_handle_reissued_after_successive_autosaves_and_repeated_undo(api):
    store,call,_,_,_=api;original,ann,oldbody,changed=deleted_source(api)
    for i in range(2):
        empty=deepcopy(ann['data']);empty['client_op']=f'empty-autosave-{i}'
        response=call('ana','PATCH',f"/annotations/{ann['id']}",headers={'If-Match':str(ann['rev'])},json=empty)
        assert response.status_code==200,response.text;ann=response.json()
    assert oldbody['events'][0]['source_lineage']['current_revision']!=ann['rev']
    assert call('ana','PATCH',f"/annotations/{ann['id']}",headers={'If-Match':str(ann['rev'])},json={**oldbody,'client_op':'old-handle'}).status_code==409
    undone=actual_history_action(changed['state'],'undo',undo_stack=changed['undo'],server_saved=ann['data'])
    assert undone['state']['events']==original['data']['events']
    restored=call('ana','PATCH',f"/annotations/{ann['id']}",headers={'If-Match':str(ann['rev'])},json=js_save_payload(undone['state'],client_op='fresh-undo'))
    assert restored.status_code==200,restored.text;ann=restored.json()
    for i in range(2):
        changed=actual_history_action(ann['data'],'delete')
        result=call('ana','PATCH',f"/annotations/{ann['id']}",headers={'If-Match':str(ann['rev'])},json=js_save_payload(changed['state'],client_op=f'delete-{i}'))
        assert result.status_code==200,result.text;deleted=result.json()
        undone=actual_history_action(changed['state'],'undo',undo_stack=changed['undo'],server_saved=deleted['data'])
        result=call('ana','PATCH',f"/annotations/{ann['id']}",headers={'If-Match':str(deleted['rev'])},json=js_save_payload(undone['state'],client_op=f'undo-{i}'))
        assert result.status_code==200,result.text;ann=result.json()
        assert not ann['data']['events'][0]['provenance']['confirmed']


def test_restore_post_write_failure_rolls_back_consumption(api,monkeypatch):
    store,call,_,_,_=api;_,ann,body,_=deleted_source(api);before=deepcopy(store.get(ann['id']));update=Store.update
    def fail_after_write(self,id,*args,**kwargs):
        result=update(self,id,*args,**kwargs)
        if id==ann['id']:raise RuntimeError('Synthetic failure after annotation write')
        return result
    with monkeypatch.context() as patcher:
        patcher.setattr(Store,'update',fail_after_write)
        with pytest.raises(RuntimeError,match='Synthetic failure'):
            call('ana','PATCH',f"/annotations/{ann['id']}",headers={'If-Match':str(ann['rev'])},json=body)
    assert store.get(ann['id'])==before
    response=call('ana','PATCH',f"/annotations/{ann['id']}",headers={'If-Match':str(ann['rev'])},json=body)
    assert response.status_code==200,response.text


def test_confirmed_undo_snapshots_never_resurrect_knowledge_on_repeated_autosaves(api):
    store,call,_,_,_=api;original,deleted,_,changed=deleted_source(api,confirmed=True)
    undone=actual_history_action(changed['state'],'undo',undo_stack=changed['undo'],server_saved=deleted['data'])
    state=undone['state'];assert state['events']==original['data']['events'] and state['events'][0]['type_known'] is True
    current=deleted
    for i in range(4):
        payload=js_save_payload(state,client_op=f'pending-restore-autosave-{i}')
        response=call('ana','PATCH',f"/annotations/{current['id']}",headers={'If-Match':str(current['rev'])},json=payload)
        assert response.status_code==200,response.text;current=response.json()
        event=current['data']['events'][0]
        assert not event['type_known'] and event['is_disfluent'] is None and not event['provenance']['confirmed']
        assert event['provenance']['action']=='restored_pending_human_review'
        state=js_actions('refreshHumanLineages(state, SAVED);state.actions=state.actions.map(a=>Object.fromEntries(Object.entries(a).reverse()));'.replace('SAVED',json.dumps(current['data'])),state)
        assert state['events'][0]['type_known'] is True,'reference refresh must not rewrite the Undo judgment snapshot'


def test_repeated_merge_undo_after_successive_autosaves(api):
    store,call,_,_,_=api;_,ann=legacy_annotation(api)
    data=deepcopy(ann['data']);second=deepcopy(data['events'][0]);second.update(event_id='second-original',start_ms=400,end_ms=600);data['events'].append(second)
    result=call('ana','PATCH',f"/annotations/{ann['id']}",headers={'If-Match':str(ann['rev'])},json={**data,'client_op':'seed-merge-history'})
    assert result.status_code==200,result.text;ann=upgrade(api,result.json())
    for i in range(2):
        original=deepcopy(ann['data']);changed=actual_history_action(original,'merge')
        result=call('ana','PATCH',f"/annotations/{ann['id']}",headers={'If-Match':str(ann['rev'])},json=js_save_payload(changed['state'],client_op=f'merge-history-{i}'))
        assert result.status_code==200,result.text;merged=result.json()
        result=call('ana','PATCH',f"/annotations/{ann['id']}",headers={'If-Match':str(merged['rev'])},json=js_save_payload(merged['data'],client_op=f'merged-autosave-{i}'))
        assert result.status_code==200,result.text;merged=result.json()
        undone=actual_history_action(changed['state'],'undo',undo_stack=changed['undo'],server_saved=merged['data'])
        assert undone['state']['events']==original['events']
        result=call('ana','PATCH',f"/annotations/{ann['id']}",headers={'If-Match':str(merged['rev'])},json=js_save_payload(undone['state'],client_op=f'undo-merge-{i}'))
        assert result.status_code==200,result.text;ann=result.json()
        assert len(ann['data']['events'])==2 and all(not e['provenance']['confirmed'] for e in ann['data']['events'])


@pytest.mark.parametrize('bad_id',['prior-human-confirmation',True,{'opaque':1}])
def test_restored_confirmation_requires_a_new_string_identifier(api,bad_id):
    store,call,_,_,_=api;original,deleted,body,_=deleted_source(api,confirmed=True)
    response=call('ana','PATCH',f"/annotations/{deleted['id']}",headers={'If-Match':str(deleted['rev'])},json=body)
    assert response.status_code==200,response.text;saved=response.json()
    payload=deepcopy(saved['data']);event=payload['events'][0];event.update(type_known=True,is_disfluent=False)
    action=confirmation(event);action['confirmation_id']=bad_id;payload['actions'].append(action);payload['client_op']='changed-content-same-confirmation-id'
    before=deepcopy(store.get(saved['id']))
    result=call('ana','PATCH',f"/annotations/{saved['id']}",headers={'If-Match':str(saved['rev'])},json=payload)
    assert result.status_code==409,result.text
    assert store.get(saved['id'])==before


def test_restored_confirmation_id_is_new_across_immutable_annotation_history(api):
    store,call,_,_,_=api;original,deleted,body,_=deleted_source(api,confirmed=True)
    result=call('ana','PATCH',f"/annotations/{deleted['id']}",headers={'If-Match':str(deleted['rev'])},json=body)
    assert result.status_code==200,result.text;saved=result.json()
    shorter=deepcopy(saved['data']);shorter.update(actions=[],client_op='older-undo-action-history')
    result=call('ana','PATCH',f"/annotations/{saved['id']}",headers={'If-Match':str(saved['rev'])},json=shorter)
    assert result.status_code==200,result.text;saved=result.json()
    data=deepcopy(saved['data']);event=data['events'][0];event.update(type_known=True,is_disfluent=False)
    action=confirmation(event);action['confirmation_id']='prior-human-confirmation';data['actions']=[action];data['client_op']='replay-archived-confirmation-id'
    before=deepcopy(store.get(saved['id']))
    result=call('ana','PATCH',f"/annotations/{saved['id']}",headers={'If-Match':str(saved['rev'])},json=data)
    assert result.status_code==409,result.text
    assert store.get(saved['id'])==before
