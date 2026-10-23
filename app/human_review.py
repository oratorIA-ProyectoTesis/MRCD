"""Authenticated proposal binding and explicit human confirmation boundaries."""
from copy import deepcopy
import hashlib
import json
import re

from app.store import Conflict, now
from core.cell_decode import result_family, validate_cell_result
from core.human_events import (HUMAN_ANNOTATION_SCHEMA, HUMAN_EVENT_SCHEMA, reviewed_content,
                               annotation_semantics, validate_human_event)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",",":"), allow_nan=False).encode()).hexdigest()


def validate_proposal_ref(ref):
    if (not isinstance(ref, dict) or set(ref) != {"run_id", "run_revision", "event_id", "event_sha256"}
            or any(not isinstance(ref.get(k), str) or not ref[k] for k in ("run_id", "event_id"))
            or type(ref.get("run_revision")) is not int or ref["run_revision"] < 1
            or not isinstance(ref.get("event_sha256"), str) or not re.fullmatch(r"[a-f0-9]{64}", ref["event_sha256"])):
        raise ValueError("proposal_ref requires exact source identity and a positive integer run revision")


LINEAGE_SCHEMA = "mrcd-human-source-v1"
RESTORE_SCHEMA = "mrcd-human-restore-v1"


def source_lineage(annotation_id, revision, event):
    content = {k: v for k, v in event.items() if k != "source_lineage"}
    return dict(schema=LINEAGE_SCHEMA, annotation_id=annotation_id, annotation_revision=revision,
                event_id=event["event_id"], event_sha256=digest(content))


def stamp_lineages(data, annotation_id, revision):
    for kind in ("events", "contextual"):
        for event in data.get(kind, []):
            if event.get("legacy_original") is not None:
                event["source_lineage"] = source_lineage(annotation_id, revision, event)


def annotation_for_review(annotation):
    """Project current reference handles without rewriting stored/frozen history."""
    result = deepcopy(annotation)
    if result["data"].get("semantic_schema") == HUMAN_ANNOTATION_SCHEMA:
        stamp_lineages(result["data"], result["id"], result["rev"])
        for entry in result["data"].get("lineage_restore", []) or []:
            entry["source_lineage"]["current_revision"] = result["rev"]
    return result


def issue_restore_registry(annotation, data, revision):
    """Reissue registered removed origins; clients cannot add registry entries."""
    present = {e["event_id"] for k in ("events", "contextual") for e in data.get(k, [])}
    entries = {e["source_lineage"]["event_id"]: deepcopy(e) for e in annotation["data"].get("lineage_restore", []) or []}
    for kind in ("events", "contextual"):
        for event in annotation["data"].get(kind, []):
            if event.get("legacy_original") is not None and event["event_id"] not in present:
                ref = source_lineage(annotation["id"], annotation["rev"], event)
                entries[event["event_id"]] = dict(legacy_original=deepcopy(event["legacy_original"]), source_lineage=dict(
                    schema=RESTORE_SCHEMA, annotation_id=annotation["id"], current_revision=revision,
                    source_revision=annotation["rev"], event_id=event["event_id"], event_sha256=ref["event_sha256"],
                    original_sha256=digest(event["legacy_original"])))
    result = []
    for event_id, entry in sorted(entries.items()):
        if event_id not in present:
            entry["source_lineage"]["current_revision"] = revision
            result.append(entry)
    return result


def validate_restore_lineage(store, event, documents, ref):
    fields = {"schema", "annotation_id", "current_revision", "source_revision", "event_id", "event_sha256", "original_sha256"}
    if set(ref) != fields or any(type(ref.get(k)) is not int or ref[k] < 1 for k in ("current_revision", "source_revision")):
        raise Conflict("invalid restoration lineage identity/revision")
    document = next((d for d in documents if d["id"] == ref["annotation_id"]), None)
    if not document or ref["current_revision"] != document["rev"] or ref["event_id"] != event["event_id"]:
        raise Conflict("foreign or stale restoration lineage")
    registered = [dict(e["source_lineage"], current_revision=document["rev"]) for e in document["data"].get("lineage_restore", []) or []]
    if ref not in registered:
        raise Conflict("unregistered or consumed restoration lineage")
    historical = next((h for h in store.history(document["id"]) if h["rev"] == ref["source_revision"]), None)
    prior = next((e for k in ("events", "contextual") for e in (historical or {}).get("data", {}).get(k, []) if e.get("event_id") == ref["event_id"]), None)
    if (not prior or prior.get("legacy_original") != event.get("legacy_original")
            or digest(event["legacy_original"]) != ref["original_sha256"]
            or source_lineage(document["id"], ref["source_revision"], prior)["event_sha256"] != ref["event_sha256"]):
        raise Conflict("restoration lineage does not bind exact immutable history/original")
    return deepcopy(ref)


def validate_original_lineage(store, event, documents, *, old=None, allow_restore=False):
    original, ref = event.get("legacy_original"), event.get("source_lineage")
    if original is None:
        if ref is not None:
            raise Conflict("source_lineage requires its immutable original")
        return None
    if ref is None:
        # Compatibility for an existing pre-lineage v1 mark, never a new child.
        if old and old.get("legacy_original") == original and not old.get("source_lineage"):
            return None
        raise Conflict("legacy_original requires authenticated source_lineage")
    if isinstance(ref, dict) and ref.get("schema") == RESTORE_SCHEMA:
        if not allow_restore:
            raise Conflict("restoration lineage is only valid in its own editable annotation")
        return validate_restore_lineage(store, event, documents, ref)
    if (not isinstance(ref, dict) or set(ref) != {"schema", "annotation_id", "annotation_revision", "event_id", "event_sha256"}
            or ref.get("schema") != LINEAGE_SCHEMA or type(ref.get("annotation_revision")) is not int
            or ref["annotation_revision"] < 1):
        raise Conflict("invalid source_lineage identity/revision")
    document = next((d for d in documents if d["id"] == ref["annotation_id"]), None)
    if not document or document["data"].get("semantic_schema") != HUMAN_ANNOTATION_SCHEMA:
        raise Conflict("foreign source_lineage outside this annotation or current A/B")
    current = next((e for k in ("events", "contextual") for e in document["data"].get(k, [])
                    if e.get("event_id") == ref["event_id"]), None)
    if not current or current.get("legacy_original") != original:
        raise Conflict("source_lineage does not bind the immutable original")
    if source_lineage(document["id"], ref["annotation_revision"], current) != ref:
        raise Conflict("stale or forged source_lineage content")
    historical = next((h for h in store.history(document["id"]) if h["rev"] == ref["annotation_revision"]), None)
    prior = next((e for k in ("events", "contextual") for e in (historical or {}).get("data", {}).get(k, [])
                  if e.get("event_id") == ref["event_id"]), None)
    if not prior or source_lineage(document["id"], ref["annotation_revision"], prior) != ref:
        raise Conflict("source_lineage revision is unavailable or does not certify this source")
    return deepcopy(ref)


def require_fresh_restoration_confirmation(store, action, documents, previous):
    """A descendant's new ID cannot reset its authenticated source's nonce history."""
    sources = [previous or {}]
    for document in {d["id"]: d for d in documents if d is not None}.values():
        sources.append(document["data"])
        for revision in store.history(document["id"]):
            if revision["data"].get("semantic_schema") == HUMAN_ANNOTATION_SCHEMA:
                annotation_semantics(revision["data"])
                sources.append(revision["data"])
    prior_ids = [a.get("confirmation_id") for source in sources for a in source.get("actions", [])
                 if a.get("type") == "confirm_human_event"]
    confirmation_id = action.get("confirmation_id")
    if not isinstance(confirmation_id, str) or not confirmation_id.strip() or confirmation_id in prior_ids:
        raise Conflict("restored mark requires a new explicit human confirmation")


def task_run(store, task):
    data = task["data"]
    run = store.get(data.get("run_id") or "", "run")
    if not run or run["parent"] != data["recording_id"] or run["status"] not in ("succeeded", "partial"):
        raise ValueError("task proposal source must be a completed run of the assigned recording")
    result = run["data"].get("result", {})
    family = result_family(result)
    if family == "cell_v2":
        validate_cell_result(result)
        if result["recording_id"] != data["recording_id"]:
            raise ValueError("task and cell result identities disagree")
    return run


def proposals(store, task):
    if task["data"]["mode"] != "assisted" or not task["data"].get("run_id"):
        return []
    run = task_run(store, task)
    d = task["data"]
    result = []
    for event in run["data"].get("result", {}).get("events", []):
        if event["end_ms"] <= d["start_ms"] - d["context_ms"] or event["start_ms"] >= d["end_ms"] + d["context_ms"]:
            continue
        if not event.get("event_id"):
            raise ValueError("task proposal requires stable event identity")
        ref = dict(run_id=run["id"], run_revision=run["rev"], event_id=event["event_id"], event_sha256=digest(event))
        result.append({**{k:event.get(k) for k in ("event_id","start_ms","end_ms","label","text","decision")},
                       "type_known":False,"is_disfluent":None,"contextual_state":None,"proposal_ref":ref,
                       "proposed_contextual_state":event.get("contextual_state"),"source_kind":"model_proposal"})
    if len({p["event_id"] for p in result}) != len(result):
        raise ValueError("ambiguous duplicate task proposal identities")
    return result


def normalize_annotation(store, task, payload, actor_id, *, adjudication_sources=None, previous=None, previous_document=None):
    if payload.get("semantic_schema") != HUMAN_ANNOTATION_SCHEMA:
        raise ValueError("explicit human annotation schema required")
    source_lookup = {p["event_id"]:p for p in proposals(store, task)} if adjudication_sources is None else {}
    lineage_documents = adjudication_sources if adjudication_sources is not None else ([previous_document] if previous_document else [])
    confirmations = [action for action in payload.get("actions", []) if action.get("type") in ("confirm_human_event", "revoke_human_confirmation")]
    data = deepcopy(payload)
    seen = set()
    for kind in ("events","contextual"):
        for event in data.setdefault(kind, []):
            eid = event.get("event_id")
            if not eid or eid in seen:
                raise ValueError("explicit human marks require unique event_id across tracks")
            seen.add(eid)
            if event["end_ms"] > store.get(task["data"]["recording_id"])["data"]["media"]["analysis"]["duration_ms"]:
                raise ValueError("human event outside actual media bounds")
            sid, ref, proposal_bounds = event.get("suggestion_id"), event.get("proposal_ref"), None
            if sid:
                # An old suggestion_id without a reference is diagnosed below;
                # populated identities never use Python's bool/float==int coercion.
                if ref is not None:
                    validate_proposal_ref(ref)
                if adjudication_sources is None:
                    source = source_lookup.get(sid)
                    if not source or ref != source["proposal_ref"]:
                        raise Conflict("stale, spoofed or wrong-run task proposal reference")
                    proposal_bounds = [source["start_ms"],source["end_ms"]]
                else:
                    candidates = [e for ann in adjudication_sources for k in ("events","contextual") for e in ann["data"].get(k,[]) if e.get("suggestion_id")==sid and e.get("proposal_ref")==ref]
                    if not candidates:
                        raise Conflict("adjudication proposal is not bound to current A/B sources")
                    if (not isinstance(ref, dict) or candidates[0].get("human_event_schema") != HUMAN_EVENT_SCHEMA
                            or not isinstance(candidates[0].get("provenance"), dict)
                            or not isinstance(candidates[0]["provenance"].get("proposal_bounds"), list)):
                        raise ValueError("unverified legacy suggestion link: preserve historical sources; explicitly confirm a new human judgment without certifying the old model link")
                    proposal_bounds = candidates[0]["provenance"]["proposal_bounds"]
            elif ref:
                raise ValueError("proposal_ref without suggestion_id")
            matching = [a for a in confirmations if a.get("event_id")==eid]
            content = reviewed_content(event)
            confirmed = bool(matching and matching[-1]["type"] == "confirm_human_event")
            if confirmed and matching[-1].get("reviewed_event") != content:
                raise Conflict("human confirmation is stale for edited bounds/type/context")
            binding = dict(suggestion_id=sid, proposal_ref=ref)
            if confirmed and "reviewed_proposal" in matching[-1] and matching[-1]["reviewed_proposal"] != binding:
                raise Conflict("human confirmation is stale for the original proposal identity")
            old = next((e for k in ("events","contextual") for e in (previous or {}).get(k,[])
                        if e.get("event_id")==eid), None)
            if old and old.get("legacy_original") is not None and event.get("legacy_original") != old["legacy_original"]:
                raise Conflict("existing immutable original cannot be removed or replaced")
            reused = confirmed and matching[-1] in (previous or {}).get("actions", [])
            if old and reused and (dict(suggestion_id=old.get("suggestion_id"), proposal_ref=old.get("proposal_ref")) != binding
                                  or old.get("provenance", {}).get("proposal_bounds") != proposal_bounds):
                raise Conflict("changed proposal identity requires a new explicit human attestation")
            parent = validate_original_lineage(store, event, lineage_documents, old=old, allow_restore=adjudication_sources is None)
            # Resolve only after the existing identity/content/history checks.
            # Client provenance and parent_event_id never establish this state.
            source_document = next((d for d in lineage_documents if parent and d["id"] == parent["annotation_id"]), None)
            source_event = next((e for k in ("events", "contextual") for e in (source_document or {}).get("data", {}).get(k, [])
                                 if parent and parent["schema"] == LINEAGE_SCHEMA and e.get("event_id") == parent["event_id"]), None)
            inherited_pending = any(e and e.get("provenance", {}).get("action") == "restored_pending_human_review"
                                    for e in (old, source_event))
            restoring = bool(parent and parent["schema"] == RESTORE_SCHEMA)
            if restoring:
                confirmed = False
                data["actions"].append(dict(type="revoke_human_confirmation", event_id=eid, reason="undo_restore_pending_review", actor_id=actor_id))
            elif confirmed and inherited_pending:
                require_fresh_restoration_confirmation(store, matching[-1], [previous_document, source_document], previous)
            pending_review = restoring or bool(not confirmed and inherited_pending)
            if confirmed and "reviewed_source_lineage" in matching[-1] and matching[-1]["reviewed_source_lineage"] != event.get("source_lineage"):
                raise Conflict("human confirmation is stale for its source lineage")
            event["type_known"] = event.get("type_known", False) if confirmed else False
            event["is_disfluent"] = event.get("is_disfluent") if confirmed else None
            event["contextual_state"] = None if event["is_disfluent"] is None else "disfluent" if event["is_disfluent"] else "functional"
            event["human_event_schema"] = HUMAN_EVENT_SCHEMA
            event["source_kind"] = ("human_adjudicated" if adjudication_sources is not None else "human_verified_model_suggestion" if sid else "human_assisted_manual" if task["data"]["mode"]=="assisted" else "human_independent") if confirmed else "model_proposal_copy" if sid else "human_unconfirmed"
            event["provenance"] = dict(actor_id=actor_id, task_id=task["id"], confirmed=confirmed,
                                       mode=task["data"]["mode"],
                                       action="restored_pending_human_review" if pending_review else "confirm_human_event" if confirmed else "unconfirmed_copy", recorded_at=now())
            if sid: event["provenance"].update(proposal_ref=ref,proposal_bounds=proposal_bounds)
            if parent:
                event["provenance"]["parent_lineage"] = deepcopy((old or {}).get("provenance", {}).get("parent_lineage") or parent)
            if adjudication_sources is not None:
                event["provenance"]["source_annotations"] = [dict(id=ann["id"],revision=ann["rev"]) for ann in adjudication_sources]
            if old and reused and reviewed_content(old)==content and old.get("provenance",{}).get("confirmed"):
                event["provenance"] = deepcopy(old["provenance"])
            validate_human_event(event)
    return data


def save_annotation(store, ann_id, task, payload, expected_rev, actor_id):
    run_id = task["data"].get("run_id") or ""
    with store.serialized(ann_id, task["id"], run_id) as tx:
        ann, current_task = tx.get(ann_id), tx.get(task["id"])
        if ann["status"] == "submitted":
            raise Conflict("annotation already submitted")
        if current_task["data"] != task["data"] or current_task["owner"] != task["owner"]:
            raise Conflict("task assignment changed")
        family, previous_family = annotation_semantics(payload), annotation_semantics(ann["data"])
        if previous_family == "explicit_v1" and family != previous_family:
            raise Conflict("stored explicit human annotation cannot be downgraded")
        fingerprint = digest(payload)
        if payload.get("client_op") and ann["data"].get("client_op") == payload["client_op"]:
            if ann["data"].get("request_fingerprint") != fingerprint:
                raise Conflict("client_op cannot identify a different human edit")
            return ann
        if family == "explicit_v1":
            if previous_family == "historical" and any(ann["data"].get(k) for k in ("events", "contextual")):
                raise Conflict("historical marks require the explicit upgrade-semantics transition before v1 edits")
            data = normalize_annotation(tx, current_task, payload, actor_id, previous=ann["data"], previous_document=ann)
            # Clients cannot invent/replace server-derived transition provenance.
            data.pop("semantic_transition", None)
            if ann["data"].get("semantic_transition"):
                data["semantic_transition"] = deepcopy(ann["data"]["semantic_transition"])
        else:
            data = deepcopy(payload)
            for kind in ("events", "contextual"):
                for event in data[kind]:
                    event["source_kind"] = "human_legacy_annotation"
                    event["provenance"] = None
        data["request_fingerprint"] = fingerprint
        if family == "explicit_v1":
            data["lineage_restore"] = issue_restore_registry(ann, data, ann["rev"] + 1)
            stamp_lineages(data, ann_id, ann["rev"] + 1)
        saved = tx.update(ann_id,data,expected_rev=expected_rev,author=actor_id)
        if current_task["status"] == "assigned": tx.update(task["id"],status="in_progress",author=actor_id)
        return saved


def upgrade_annotation(store, ann_id, task, payload, actor_id):
    """Explicit, loss-preserving transition; not a human attestation of old marks."""
    with store.serialized(ann_id, task["id"]) as tx:
        ann, current_task = tx.get(ann_id), tx.get(task["id"])
        if ann["rev"] != payload["expected_annotation_revision"]:
            raise Conflict("annotation revision changed; reload before explicit transition")
        if ann["status"] == "submitted" or current_task["data"] != task["data"] or current_task["owner"] != task["owner"]:
            raise Conflict("annotation/task is no longer editable")
        if annotation_semantics(ann["data"]) != "historical":
            raise Conflict("annotation already uses explicit human semantics")
        data = deepcopy(ann["data"])
        data["semantic_schema"] = HUMAN_ANNOTATION_SCHEMA
        transition = dict(schema="mrcd-human-upgrade-v1", source_annotation_id=ann_id, source_revision=ann["rev"],
                          source_sha256=digest(ann["data"]), actor_id=actor_id, reason=payload["reason"], recorded_at=now())
        data["semantic_transition"] = transition
        data["actions"] = [*data.get("actions", []), dict(type="upgrade_human_semantics", **transition)]
        for kind in ("events", "contextual"):
            for index, event in enumerate(data.get(kind, [])):
                original = deepcopy(event)
                event.update(event_id=f"legacy-{kind}-{index}-{digest(original)[:16]}", legacy_original=original,
                             type_known=False, is_disfluent=None, contextual_state=None,
                             human_event_schema=HUMAN_EVENT_SCHEMA, source_kind="human_unconfirmed",
                             suggestion_id=None, proposal_ref=None,
                             provenance=dict(confirmed=False, action="explicit_legacy_transition", actor_id=actor_id,
                                             task_id=task["id"], mode=task["data"]["mode"], recorded_at=now()))
                validate_human_event(event)
        data.pop("client_op", None)
        data.pop("request_fingerprint", None)
        stamp_lineages(data, ann_id, ann["rev"] + 1)
        return tx.update(ann_id, data, expected_rev=ann["rev"], author=actor_id)
