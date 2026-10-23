"""Independent bounded Undo/cache review; canonical probes11003 only."""
from copy import deepcopy
import json

from tests.test_human_semantics import api, confirmation
from tests.test_human_undo import actual_history_action, deleted_source
from tests.test_human_semantics_review import js_actions, js_save_payload


def save(api, ann, state, op):
    return api[1](
        'ana', 'PATCH', f"/annotations/{ann['id']}",
        headers={'If-Match': str(ann['rev'])},
        json=js_save_payload(state, client_op=op))


def restored_raw_state(api):
    original, deleted, _, changed = deleted_source(api, confirmed=True)
    undone = actual_history_action(changed['state'], 'undo',
        undo_stack=changed['undo'], server_saved=deleted['data'])
    raw = undone['state']
    assert raw['events'] == original['data']['events']
    restored = save(api, deleted, raw, 'independent-cache-restore')
    assert restored.status_code == 200, restored.text
    current = restored.json()
    raw = js_actions('refreshHumanLineages(state, SAVED);'.replace('SAVED', json.dumps(current['data'])), raw)
    assert raw['events'][0]['type_known'] is True
    assert current['data']['events'][0]['provenance']['action'] == 'restored_pending_human_review'
    return original, current, raw


def test_new_confirmation_from_raw_undo_cache_is_savable_and_stays_confirmed(api):
    original, current, state = restored_raw_state(api)
    state = js_actions("load('e:'+state.events[0].event_id);values['#f-known'].checked=true;values['#f-context'].value='false';saveDraft();", state)
    fresh_id = next(a for a in reversed(state['actions']) if a.get('type') == 'confirm_human_event')['confirmation_id']
    assert fresh_id != 'prior-human-confirmation'
    for i in range(3):
        response = save(api, current, state, f'independent-confirmed-cache-{i}')
        assert response.status_code == 200, response.text
        current = response.json()
        event = current['data']['events'][0]
        assert event['legacy_original'] == original['data']['events'][0]['legacy_original']
        assert event['type_known'] is True and event['is_disfluent'] is False
        assert event['provenance']['confirmed'] is True
        assert event['provenance']['actor_id'] == api[2]['ana']
        state = js_actions('refreshHumanLineages(state, SAVED);'.replace('SAVED', json.dumps(current['data'])), state)


def test_split_descendant_cannot_replay_old_restoration_confirmation_identifier(api):
    _, current, state = restored_raw_state(api)
    changed = actual_history_action(state, 'split')
    result = save(api, current, changed['state'], 'independent-pending-split')
    assert result.status_code == 200, result.text
    current = result.json()
    events = current['data']['events']
    assert len(events) == 2 and all(not e['provenance']['confirmed'] for e in events)
    child = next(e for e in events if e['event_id'] == 'independent-child-1')
    assert child['legacy_original'] == events[0]['legacy_original']
    body = deepcopy(current['data'])
    child = next(e for e in body['events'] if e['event_id'] == child['event_id'])
    child.update(type_known=True, is_disfluent=True)
    replay = confirmation(child)
    replay['confirmation_id'] = 'prior-human-confirmation'
    body['actions'].append(replay)
    before = deepcopy(api[0].get(current['id']))
    response = save(api, current, body, 'independent-descendant-replayed-nonce')
    assert response.status_code == 409, response.text
    assert api[0].get(current['id']) == before


import pytest
from core.human_events import HUMAN_ANNOTATION_SCHEMA
from tests.test_human_semantics import task
from tests.test_human_semantics_review import SOURCE


def split_pending(state, event_id, at, generation):
    """Extract the real split/invalidation actions with deterministic clock/ID stubs."""
    invalidation = 'function invalidateHuman(mark) {' + SOURCE.split('function invalidateHuman(mark) {', 1)[1].split("\n  regions.on('region-created'", 1)[0]
    split = "$('#split-ev').onclick = () => {" + SOURCE.split("$('#split-ev').onclick = () => {", 1)[1].split("\n  $('#merge-ev').onclick", 1)[0]
    script = f"values['#split-ev']={{}};const now=()=>{at};counter={generation * 10};" + invalidation + split
    return js_actions(script + ";\nload('e:' + TARGET);values['#split-ev'].onclick();".replace('TARGET', json.dumps(event_id)), state)


def assert_pending(event):
    assert event['provenance']['action'] == 'restored_pending_human_review'
    assert event['provenance']['confirmed'] is False
    assert event['type_known'] is False
    assert event['is_disfluent'] is None and event['contextual_state'] is None


def test_real_repeated_split_inherits_pending_from_authenticated_source(api):
    original, current, state = restored_raw_state(api)
    event_id = current['data']['events'][0]['event_id']
    for generation, at in enumerate((200, 250, 275), 1):
        changed = split_pending(state, event_id, at, generation)
        result = save(api, current, changed, f'pending-generation-{generation}')
        assert result.status_code == 200, result.text
        current = result.json()
        assert len(current['data']['events']) == generation + 1
        for event in current['data']['events']:
            assert_pending(event)
            assert event['legacy_original'] == original['data']['events'][0]['legacy_original']
        event_id = f'ui-{generation * 10 + 1}'
        state = deepcopy(current['data'])
    child = next(event for event in state['events'] if event['event_id'] == event_id)
    child.update(type_known=True, is_disfluent=True)
    action = confirmation(child)
    action['confirmation_id'] = 'prior-human-confirmation'
    state['actions'] = [action]
    before = deepcopy(api[0].get(current['id']))
    replay = save(api, current, state, 'third-generation-replay')
    assert replay.status_code == 409, replay.text
    assert api[0].get(current['id']) == before


@pytest.mark.parametrize('nonce', ['prior-human-confirmation', '', None, True, {'opaque': 1}])
@pytest.mark.parametrize('track', ['events', 'contextual'])
def test_new_id_transfer_cannot_replay_nonce_from_pending_source_history(api, nonce, track):
    _, current, _ = restored_raw_state(api)
    # A new child uses an exact current, authenticated source-v1 ref; it does not
    # inherit client provenance or rely on same-ID lookup in the old document.
    body = deepcopy(current['data'])
    child = deepcopy(body['events'][0])
    child.update(event_id='transferred-child', type_known=True, is_disfluent=False,
                 expression='este', provenance={'confirmed': True, 'action': 'invented'})
    body['events'] = [child] if track == 'events' else []
    body['contextual'] = [child] if track == 'contextual' else []
    action = confirmation(child)
    action['confirmation_id'] = nonce
    body['actions'] = [action]  # prior nonce survives only in immutable history
    before = deepcopy(api[0].get(current['id']))
    result = save(api, current, body, 'pending-transfer-replayed-nonce')
    assert result.status_code == 409, result.text
    assert api[0].get(current['id']) == before


def test_fresh_descendant_confirmation_and_later_real_ui_autosaves_remain_valid(api):
    _, current, state = restored_raw_state(api)
    changed = split_pending(state, state['events'][0]['event_id'], 200, 1)
    result = save(api, current, changed, 'pending-fresh-split')
    assert result.status_code == 200, result.text
    current = result.json()
    child = next(event for event in current['data']['events'] if event['event_id'] == 'ui-11')
    assert_pending(child)
    state = js_actions("load('e:ui-11');values['#f-known'].checked=true;values['#f-context'].value='false';saveDraft();", current['data'])
    for index in range(3):
        response = save(api, current, state, f'fresh-descendant-autosave-{index}')
        assert response.status_code == 200, response.text
        current = response.json()
        child = next(event for event in current['data']['events'] if event['event_id'] == 'ui-11')
        assert child['provenance']['action'] == 'confirm_human_event'
        assert child['provenance']['confirmed'] is True
        assert child['type_known'] is True and child['is_disfluent'] is False
        assert child['provenance']['actor_id'] == api[2]['ana']
        state = js_actions('refreshHumanLineages(state, SAVED);'.replace('SAVED', json.dumps(current['data'])), state)


@pytest.mark.parametrize('attestation', ['copy', 'replayed', 'fresh'])
def test_adjudication_transfer_obeys_pending_source_and_history_nonce(api, attestation):
    store, call, users, _, _ = api
    _, current, _ = restored_raw_state(api)
    assert call('ana', 'POST', f"/review/tasks/{current['parent']}/submit").status_code == 200
    tid, other = task(api, 'beto', mode='blind')
    reviewed = call('beto', 'PATCH', f"/annotations/{other['annotation']['id']}",
                    headers={'If-Match': str(other['annotation']['rev'])},
                    json=dict(semantic_schema=HUMAN_ANNOTATION_SCHEMA, events=[], contextual=[],
                              coverage=[[0, 1000]], actions=[]))
    assert reviewed.status_code == 200, reviewed.text
    assert call('beto', 'POST', f'/review/tasks/{tid}/submit').status_code == 200
    created = call('admin', 'POST', '/adjudication-cases', json=dict(
        task_a=current['parent'], task_b=tid, adjudicator=users['carla']))
    assert created.status_code == 200, created.text
    cid = created.json()['id']
    view = call('carla', 'GET', f'/review/adjudication/{cid}').json()
    # Case source order is canonical task identity order, not request order.
    source = next(track['events'][0] for track in view['tracks'].values() if track['events'])
    event = deepcopy(source)
    event.update(event_id='adjudicated-pending-descendant', type_known=True, is_disfluent=False)
    action = confirmation(event)
    action['confirmation_id'] = 'prior-human-confirmation' if attestation == 'replayed' else 'new-adjudicator-attestation'
    body = dict(semantic_schema=HUMAN_ANNOTATION_SCHEMA, case_id=cid, case_revision=view['case']['rev'],
                events=[event], contextual=[], actions=[] if attestation == 'copy' else [action],
                based_on={side: view['tracks'][side]['rev'] for side in 'AB'},
                reasons={'review': 'Synthetic authenticated pending source transfer'})
    before = deepcopy(store.get(cid))
    response = call('carla', 'POST', '/adjudications', json=body)
    if attestation == 'replayed':
        assert response.status_code == 409, response.text
        assert store.get(cid) == before and store.find('adjudication') == []
        return
    assert response.status_code == 200, response.text
    event = store.get(response.json()['id'])['data']['events'][0]
    if attestation == 'copy':
        assert_pending(event)
    else:
        assert event['type_known'] is True and event['is_disfluent'] is False
        assert event['provenance']['confirmed'] is True and event['provenance']['actor_id'] == users['carla']
    frozen = call('admin', 'POST', '/dataset-snapshots', json=dict(name='pending-descendant-transfer', splits=['dev']))
    assert frozen.status_code == 200, frozen.text
    final = frozen.json()['data']['items'][0]['events'][0]
    assert final == event
