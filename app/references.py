"""Atomic, append-only adjudication and validated frozen reference assembly.

No inference or metric computation occurs here. An explicit legacy admin review
hashes the current local source bytes. Historical frozen
snapshots are never rewritten; a new snapshot must pass the current validation.
"""

from __future__ import annotations

import hashlib
import json
import re

from app import media, media_dir
from app.ops import require_global_adjudicator
from app.store import Conflict, Store, now
from core.constants import TAXONOMY, TAXONOMY_VERSION
from core.human_events import HUMAN_ANNOTATION_SCHEMA, annotation_semantics, validate_human_event

REFERENCE_FIELDS = ("recording_id", "start_ms", "end_ms", "split", "speaker_id", "guideline_version", "mode")


def source_provenance(store: Store, case: dict) -> tuple[dict, dict, dict]:
    """Validate individual tracks separately from the case's aggregate exposure."""
    tasks = case["data"].get("tasks", [])
    if len(tasks) != 2 or len(set(tasks)) != 2:
        raise Conflict("based_on: a case requires two distinct source tracks")
    annotations, task_revisions, modes = {}, {}, {}
    for side, task_id in zip("AB", tasks):
        task, ann = store.get(task_id, "task"), store.get(f"ann_{task_id}", "annotation")
        if not task or not ann or task["status"] != "submitted" or ann["status"] != "submitted":
            raise Conflict("based_on: A/B annotations must still be submitted")
        for field in (*REFERENCE_FIELDS[:-1], "context_ms"):
            if task["data"].get(field) != case["data"].get(field):
                raise Conflict(f"{field}: A/B source no longer matches the case")
        mode = task["data"].get("mode")
        if mode not in ("blind", "assisted"):
            raise Conflict("mode: invalid A/B source exposure")
        annotations[side], task_revisions[side], modes[side] = ann["rev"], task["rev"], mode
    aggregate = "assisted" if "assisted" in modes.values() else "blind"
    if case["data"].get("mode") != aggregate:
        raise Conflict("mode: case exposure must aggregate the A/B source modes")
    if "source_modes" in case["data"] and case["data"]["source_modes"] != modes:
        raise Conflict("source_modes: A/B exposure changed since the case was opened")
    return annotations, task_revisions, modes


def validate_case_identity(case: dict) -> None:
    if not case["data"].get("media_sha256") or not case["data"].get("taxonomy_version"):
        raise ValueError("legacy case provenance is incomplete: an administrator must explicitly review current sources via review-provenance; historical identity is not assumed")


def provenance_review_data(store: Store, case_id: str, review_id: str) -> dict:
    review = store.get(review_id, "provenance_review")
    if not review or review["parent"] != case_id or review["status"] != "reviewed":
        raise Conflict("provenance_review: missing immutable administrator review")
    data = review["data"]
    if data.get("case_id") != case_id or data.get("verification_scope") != "current_source_and_annotations_only" or data.get("historical_identity_verified") is not False:
        raise Conflict("provenance_review: incompatible review scope or historical identity claim")
    return data


def validate_provenance_review(store: Store, case: dict, annotations: dict, tasks: dict) -> None:
    review_id = case["data"].get("provenance_review")
    if not review_id:
        return
    data = provenance_review_data(store, case["id"], review_id)
    if data.get("annotation_revisions") != annotations or data.get("task_revisions") != tasks:
        raise Conflict("provenance_review: A/B changed after current-source review; administrator must review again")
    if any(data.get(k) != case["data"].get(k) for k in ("media_sha256", "taxonomy_version", "guideline_version", "source_modes")):
        raise Conflict("provenance_review: source identity no longer matches the reviewed case")


def review_legacy_provenance(store: Store, case_id: str, payload: dict, actor: dict) -> dict:
    """Attest current sources only; never manufacture pre-upgrade provenance."""
    initial = store.get(case_id, "case")
    if not initial:
        raise ValueError("case_id: case not found")
    ids = initial["data"].get("tasks", [])
    with store.serialized(case_id, *ids, *(f"ann_{t}" for t in ids), initial["data"]["recording_id"], actor["id"]) as tx:
        admin = tx.get(actor["id"], "user")
        if not admin or admin["status"] != "active" or admin["data"].get("role") != "admin" or admin["data"].get("campaign") or tx.get(admin["parent"] or "", "campaign"):
            raise PermissionError("Only an active global administrator may review legacy provenance.")
        case = tx.get(case_id, "case")
        if case["rev"] != payload["expected_case_revision"]:
            raise Conflict("expected_case_revision: case changed; reload before review")
        finals = tx.find("adjudication", parent=case_id, status="final")
        renewing_closed = case["status"] == "closed"
        if renewing_closed:
            if not case["data"].get("provenance_review") or not finals or any(
                d["data"].get("adjudication_schema") != "mrcd-adjudication-v2" for d in finals
            ):
                raise Conflict("legacy final/closed case: current-source review cannot certify unverified historical decisions; preserve originals for explicit curated migration")
            # Validate the saved chain against its active review, not changed A/B.
            # A renewal must not silently alter or recertify the saved decision.
            _effective_finals(tx, finals, check_sources=False)
        elif case["status"] != "open" or finals:
            raise Conflict("legacy final/case state: unfinished work or a previously reviewed v2 final is required")
        if case["data"].get("media_sha256") and case["data"].get("taxonomy_version") and not case["data"].get("provenance_review"):
            raise Conflict("case already has source provenance; do not overwrite it through legacy review")
        previous_review = case["data"].get("pending_provenance_review") or case["data"].get("provenance_review")
        if previous_review:
            provenance_review_data(tx, case_id, previous_review)
        annotations, tasks, modes = source_provenance(tx, case)
        if annotations != payload["expected_annotation_revisions"] or tasks != payload["expected_task_revisions"]:
            raise Conflict("A/B revision changed; reload before current-source review")
        if payload["current_source_reviewed"] is not True or len(payload["reason"].strip()) < 10:
            raise ValueError("current_source_reviewed/reason: explicit current-source review and meaningful reason required")
        if payload["guideline_version"] != case["data"].get("guideline_version"):
            raise Conflict("guideline_version: confirm the manual actually used for A/B; no silent relabeling")
        if payload["taxonomy_version"] != TAXONOMY_VERSION:
            raise ValueError("taxonomy_version: the current taxonomy must be explicitly reviewed")
        data = {**case["data"], "media_sha256": payload["media_sha256"], "taxonomy_version": payload["taxonomy_version"],
                "source_modes": modes, "events": [], "contextual": []}
        rec = tx.get(data["recording_id"], "recording")
        if not rec or not re.fullmatch(r"[a-f0-9]{64}", rec["data"].get("sha256", "")):
            raise ValueError("media_sha256: valid local source identity is required")
        try:
            digest = media.sha256(media_dir(rec) / "analysis.wav")
        except OSError:
            raise ValueError("media_sha256: current local analysis file is unavailable; review cannot be certified") from None
        recorded = rec["data"].get("media", {}).get("analysis", {}).get("sha256")
        if digest != payload["media_sha256"] or digest != recorded:
            raise Conflict("media_sha256: observed bytes, supplied digest, and recording metadata must agree")
        validate_region(tx, data)
        audit = tx.create("provenance_review", {
            "case_id": case_id, "expected_case_revision": case["rev"], "recording_revision": rec["rev"],
            "annotation_revisions": annotations, "task_revisions": tasks, "source_modes": modes,
            "media_sha256": digest, "media_variant": "analysis.wav", "recorded_original_sha256": rec["data"]["sha256"],
            "guideline_version": payload["guideline_version"], "taxonomy_version": TAXONOMY_VERSION,
            "verification_scope": "current_source_and_annotations_only", "historical_identity_verified": False,
            "previous_identity": {k: case["data"].get(k) for k in ("media_sha256", "taxonomy_version", "source_modes")},
            "previous_review": previous_review, "reviewed_at": now(), "reason": payload["reason"].strip(),
            "revises_adjudication": case["data"].get("current_adjudication") if renewing_closed else None,
            "review_kind": "renewal" if previous_review else "initial",
        }, parent=case_id, owner=actor["id"], status="reviewed", author=actor["id"])
        identity = {"media_sha256": digest, "taxonomy_version": TAXONOMY_VERSION, "source_modes": modes, "provenance_review": audit["id"]}
        update = {"pending_provenance_review": audit["id"]} if renewing_closed else identity
        updated = tx.update(case_id, {**case["data"], **update}, expected_rev=case["rev"], author=actor["id"])
        return {"case": updated, "provenance_review": audit}


def correction_source_case(store: Store, case: dict) -> dict:
    """Only a correction may promote an explicitly renewed current-source review."""
    pending_id = case["data"].get("pending_provenance_review")
    if not pending_id:
        return case
    pending = provenance_review_data(store, case["id"], pending_id)
    if case["status"] != "closed" or pending.get("revises_adjudication") != case["data"].get("current_adjudication"):
        raise Conflict("pending_provenance_review: review targets another adjudication; administrator must review again")
    data = {**case["data"], **{k: pending[k] for k in ("media_sha256", "taxonomy_version", "source_modes")},
            "provenance_review": pending_id}
    data.pop("pending_provenance_review", None)
    return {**case, "data": data}


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def union_intervals(spans) -> list[list[int]]:
    """Union only the coverage; overlapping phenomena remain distinct events."""
    merged = []
    for a, b in sorted(map(tuple, spans)):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return merged


def speaker_key(store: Store, rec: dict, speaker: str | None) -> str | None:
    project = store.get(rec["data"].get("project") or "", "project")
    return (project or {"data": {}})["data"].get("speaker_aliases", {}).get(speaker, speaker)


def validate_region(store: Store, item: dict) -> dict:
    annotation_semantics(item)
    rec = store.get(item.get("recording_id", ""), "recording")
    if not rec or rec["status"] != "ready":
        raise ValueError("recording_id: a ready source recording is required")
    media = rec["data"].get("media")
    analysis = media.get("analysis") if isinstance(media, dict) else None
    if not isinstance(analysis, dict):
        raise ValueError("media_sha256/duration_ms: missing source analysis metadata")
    duration, digest = analysis.get("duration_ms"), analysis.get("sha256", "")
    if type(duration) is not int or duration <= 0 or not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
        raise ValueError("media_sha256/duration_ms: incomplete source identity")
    a, b = item.get("start_ms"), item.get("end_ms")
    if type(a) is not int or type(b) is not int or not 0 <= a < b <= duration:
        raise ValueError("start_ms/end_ms: reference region is outside source media bounds")
    if item.get("split") not in ("pilot", "train", "dev", "test"):
        raise ValueError("split: unknown reference partition")
    if not isinstance(item.get("guideline_version"), str) or not item["guideline_version"].strip():
        raise ValueError("guideline_version: required")
    if item.get("mode") not in ("blind", "assisted") or (item["split"] == "test" and item["mode"] != "blind"):
        raise ValueError("mode/split: test references must be blind")
    if not isinstance(item.get("speaker_id"), str) or not item["speaker_id"].strip():
        raise ValueError("speaker_id: a known canonical speaker is required")
    speaker = speaker_key(store, rec, item["speaker_id"])
    if not speaker or speaker != speaker_key(store, rec, rec["data"].get("speaker_id")):
        raise ValueError("speaker_id: reference and source recording must agree")
    if item.get("media_sha256", digest) != digest:
        raise ValueError("media_sha256: source changed since adjudication")
    if item.get("taxonomy_version", TAXONOMY_VERSION) != TAXONOMY_VERSION:
        raise ValueError("taxonomy_version: incompatible reference taxonomy")
    project = store.get(rec["data"].get("project") or "", "project") or {"data": {}}
    assigned = project["data"].get("splits", {}).get(speaker, project["data"].get("splits", {}).get(item["speaker_id"]))
    if assigned and assigned != item["split"]:
        raise ValueError("split: reference conflicts with the frozen speaker partition")
    seen = set()
    for kind in ("events", "contextual"):
        if not isinstance(item.get(kind), list):
            raise ValueError(f"{kind}: expected an explicit event list")
        for event in item[kind]:
            if not isinstance(event, dict):
                raise ValueError(f"{kind}: expected event objects")
            if event.get("human_event_schema"):
                validate_human_event(event)
            x, y = event.get("start_ms"), event.get("end_ms")
            if type(x) is not int or type(y) is not int or not 0 <= x < y <= duration or y <= a or x >= b:
                raise ValueError(f"{kind}.start_ms/end_ms: invalid media bounds or event outside reviewed region")
            if kind == "events":
                decision = event.get("decision", "event")
                label = event.get("label")
                if decision not in ("event", "uncertain", "not_evaluable") or (
                    label is not None and label not in TAXONOMY
                ) or (decision == "event" and label is None):
                    raise ValueError("events.label/decision: incompatible taxonomy or decision")
                key = (x, y, label, decision)
                if key in seen:
                    raise Conflict("ambiguous duplicate event; review it explicitly before freezing")
                seen.add(key)
    return rec


def finalize_adjudication(store: Store, payload: dict, actor: dict) -> dict:
    semantic_family = annotation_semantics(payload)
    initial = store.get(payload["case_id"], "case")
    if not initial:
        raise ValueError("case_id: case not found")
    tasks = initial["data"]["tasks"]
    lock_ids = [*tasks, *(f"ann_{t}" for t in tasks), initial["data"]["recording_id"], actor["id"]]
    fingerprint = hashlib.sha256(canonical(payload).encode()).hexdigest()
    with store.serialized(initial["id"], *lock_ids) as tx:
        case = tx.get(initial["id"], "case")
        require_global_adjudicator(tx, actor["id"])
        current_actor = tx.get(actor["id"], "user")
        if case["owner"] != actor["id"] and current_actor["data"]["role"] != "admin":
            raise PermissionError("This case is assigned to another adjudicator.")
        if case["data"]["tasks"] != tasks:
            raise Conflict("case changed; reload before adjudicating")
        finals = [a for a in tx.find("adjudication", parent=case["id"], status="final")]
        for decision in finals:
            if decision["data"].get("request_fingerprint") == fingerprint:
                return {"id": decision["id"], "revision": decision["data"]["decision_revision"]}
        if payload["case_revision"] != case["rev"]:
            raise Conflict("case_revision changed; reload the case")
        predecessor = payload.get("supersedes")
        if case["status"] == "closed":
            if not predecessor or predecessor != case["data"].get("current_adjudication"):
                raise Conflict("closed case: correction requires explicit supersedes and a revision reason")
            if len((payload.get("revision_reason") or "").strip()) < 10:
                raise ValueError("revision_reason: explain the explicit correction (at least 10 characters)")
            _effective_finals(tx, finals, check_sources=False)
        elif case["status"] != "open" or predecessor or finals:
            raise Conflict("ambiguous case/final state; explicit integrity review is required")
        elif case["data"].get("current_adjudication") or type(case["data"].get("adjudication_revision", 0)) is not int or case["data"].get("adjudication_revision", 0) != 0:
            raise Conflict("adjudication_revision: an unfinished case cannot already have a decision pointer or counter")
        case = correction_source_case(tx, case)
        validate_case_identity(case)
        current, task_revisions, modes = source_provenance(tx, case)
        validate_provenance_review(tx, case, current, task_revisions)
        if payload["based_on"] != current:
            raise Conflict("A/B revisions changed; reload the case")
        annotations = [tx.get(f"ann_{tid}", "annotation") for tid in tasks]
        source_families = [annotation_semantics(ann["data"]) for ann in annotations]
        if "explicit_v1" in source_families and semantic_family != "explicit_v1":
            raise ValueError("Explicit human semantic schema is required to adjudicate versioned source evidence")
        semantic = payload
        if payload.get("semantic_schema") == HUMAN_ANNOTATION_SCHEMA:
            from app.human_review import normalize_annotation

            semantic = normalize_annotation(tx, case, payload, actor["id"], adjudication_sources=annotations)
        data = {k: case["data"][k] for k in REFERENCE_FIELDS}
        data.update(events=semantic["events"], contextual=semantic["contextual"], taxonomy_version=TAXONOMY_VERSION,
                    media_sha256=case["data"].get("media_sha256"), source_modes=modes)
        if payload.get("semantic_schema"):
            data["semantic_schema"] = payload["semantic_schema"]
        if case["data"].get("provenance_review"):
            data["provenance_review"] = case["data"]["provenance_review"]
        validate_region(tx, data)
        revision = case["data"].get("adjudication_revision", 0) + 1
        decision = tx.create("adjudication", {
            **data, "case_id": case["id"], "adjudication_schema": "mrcd-adjudication-v2",
            "decision_revision": revision, "request_fingerprint": fingerprint,
            "based_on": {side: {"task": task_id, "rev": current[side]} for side, task_id in zip("AB", tasks)},
            "reasons": payload["reasons"], "supersedes": predecessor, "revision_reason": payload.get("revision_reason"),
        }, id=f"adj_{fingerprint[:32]}", parent=case["id"], owner=actor["id"], status="final", author=actor["id"])
        tx.update(case["id"], {**case["data"], "source_modes": modes, "current_adjudication": decision["id"], "adjudication_revision": revision},
                  expected_rev=payload["case_revision"], status="closed", author=actor["id"])
        return {"id": decision["id"], "revision": revision}


def _effective_finals(store: Store, sources: list[dict], *, check_sources: bool = True) -> list[dict]:
    groups = {}
    for source in sources:
        if source["data"].get("adjudication_schema") != "mrcd-adjudication-v2":
            raise ValueError(f"legacy final {source['id']}: unverified historical decision requires explicit curated migration; current-source review is for unfinished cases only")
        case_id = source["data"].get("case_id")
        if not case_id:
            raise ValueError(f"case_id: historical adjudication {source['id']} lacks reviewable provenance")
        groups.setdefault(case_id, []).append(source)
    effective = []
    for case_id, decisions in groups.items():
        by_id = {d["id"]: d for d in decisions}
        superseded = {d["data"].get("supersedes") for d in decisions} - {None}
        heads = set(by_id) - superseded
        if len(heads) != 1 or not superseded.issubset(by_id):
            raise Conflict(f"ambiguous historical finals for case {case_id}; no automatic winner")
        head = by_id[next(iter(heads))]
        visited, chain, cursor = set(), [], head
        while cursor:
            if cursor["id"] in visited:
                raise Conflict(f"ambiguous cyclic revisions for case {case_id}")
            visited.add(cursor["id"])
            chain.append(cursor)
            cursor = by_id.get(cursor["data"].get("supersedes"))
        if visited != set(by_id):
            raise Conflict(f"ambiguous disconnected revisions for case {case_id}")
        for expected, decision in enumerate(reversed(chain), 1):
            data = decision["data"]
            if decision["parent"] != case_id or type(data.get("decision_revision")) is not int or data["decision_revision"] != expected:
                raise Conflict(f"decision_revision: non-contiguous revision provenance for case {case_id}")
            reason = data.get("revision_reason")
            if expected > 1 and (not isinstance(reason, str) or len(reason.strip()) < 10):
                raise Conflict(f"revision_reason: correction {decision['id']} requires an explicit meaningful reason")
        case = store.get(case_id, "case")
        if not case or case["status"] != "closed" or case["data"].get("current_adjudication") != head["id"]:
            raise Conflict(f"case_id/current_adjudication: stale or unreviewed case {case_id}")
        if len(case["data"].get("tasks", [])) != 2:
            raise Conflict("based_on: a case must identify two source tracks")
        for decision in chain:
            based = decision["data"].get("based_on")
            if not isinstance(based, dict) or set(based) != {"A", "B"}:
                raise Conflict("based_on: every decision revision must identify both source tracks")
            for side, tid in zip("AB", case["data"]["tasks"]):
                ref = based[side]
                if not isinstance(ref, dict) or set(ref) != {"task", "rev"} or ref["task"] != tid or type(ref["rev"]) is not int or ref["rev"] < 1:
                    raise Conflict("based_on: invalid source provenance in revision history")
                annotation = store.get(f"ann_{tid}", "annotation")
                if not annotation or ref["rev"] > annotation["rev"]:
                    raise Conflict("based_on: revision history references an unavailable annotation revision")
            review_id = decision["data"].get("provenance_review")
            if review_id:
                review = provenance_review_data(store, case_id, review_id)
                if review.get("annotation_revisions") != {side: based[side]["rev"] for side in "AB"} or any(
                    review.get(k) != decision["data"].get(k) for k in ("media_sha256", "taxonomy_version", "guideline_version", "source_modes")
                ):
                    raise Conflict("provenance_review: saved decision does not match its own immutable review")
        data = head["data"]
        if type(case["data"].get("adjudication_revision")) is not int or case["data"]["adjudication_revision"] != data["decision_revision"]:
            raise Conflict(f"adjudication_revision: case and effective decision disagree for {case_id}")
        validate_case_identity(case)
        for field in (*REFERENCE_FIELDS, "taxonomy_version", "media_sha256", "source_modes", "provenance_review"):
            if data.get(field) != case["data"].get(field):
                raise Conflict(f"{field}: adjudication no longer matches its case")
        if check_sources:
            if case["data"].get("pending_provenance_review"):
                raise Conflict("pending_provenance_review: renewed sources require an explicit adjudication correction before certification")
            annotations, tasks, _ = source_provenance(store, case)
            validate_provenance_review(store, case, annotations, tasks)
            if data.get("based_on") != {s: {"task": tid, "rev": annotations[s]} for s, tid in zip("AB", case["data"]["tasks"])}:
                raise Conflict("based_on: stale A/B references; explicit adjudication revision required")
        effective.append(head)
    return effective


def effective_adjudications(store: Store, sources: list[dict]) -> list[dict]:
    """Shared current-reference view; consumers must never pool revision history."""
    heads = _effective_finals(store, sources)
    for head in heads:
        validate_region(store, head["data"])
    return heads


def freeze_snapshot(store: Store, request: dict, actor_id: str) -> dict:
    kind = "adjudication" if request["include"] == "adjudicated" else "task"
    wanted_status = "final" if kind == "adjudication" else "submitted"
    sources = [s for s in store.find(kind, status=wanted_status) if s["data"].get("split") in request["splits"]]
    if not sources:
        raise ValueError("snapshot: no eligible reviewed regions")
    locks = set()
    for source in sources:
        locks.update((source["id"], source["data"].get("recording_id", "")))
        case = store.get(source["data"].get("case_id", ""), "case") if kind == "adjudication" else None
        if case:
            locks.add(case["id"])
            locks.update(case["data"]["tasks"])
            locks.update(f"ann_{t}" for t in case["data"]["tasks"])
        if kind == "task":
            locks.add(f"ann_{source['id']}")
    locks.discard("")
    with store.serialized(*sorted(locks)) as tx:
        sources = [tx.get(s["id"]) for s in sources]
        if any(s["status"] != wanted_status for s in sources):
            raise Conflict("reference state changed; retry snapshot creation")
        # Diagnose leakage even in historical sources before attempting to certify them.
        splits_of = {}
        for source in sources:
            data = source["data"]
            rec = tx.get(data.get("recording_id", ""), "recording")
            if not rec:
                raise ValueError("recording_id: missing source")
            speaker = speaker_key(tx, rec, data.get("speaker_id"))
            digest = rec["data"].get("media", {}).get("analysis", {}).get("sha256")
            for key in (f"speaker:{speaker}", f"media:{digest}"):
                splits_of.setdefault(key, set()).add(data.get("split"))
        leaks = {k: sorted(v) for k, v in splits_of.items() if len(v) > 1 and not k.endswith(":None")}
        if leaks:
            raise Conflict({"leakage": leaks})
        if kind == "adjudication":
            sources = effective_adjudications(tx, sources)
        items, recording_speakers = [], {}
        for source in sources:
            if kind == "adjudication":
                data = dict(source["data"], source=source["id"], source_kind="human_adjudicated")
            else:
                ann = tx.get(f"ann_{source['id']}", "annotation")
                if not ann or ann["status"] != "submitted":
                    raise Conflict("annotation: submitted source no longer available")
                data = {k: source["data"].get(k) for k in REFERENCE_FIELDS}
                data.update(events=ann["data"]["events"], contextual=ann["data"]["contextual"],
                            source=f"{ann['id']}@{ann['rev']}", source_kind="human_single_annotation")
                if ann["data"].get("semantic_schema"):
                    data["semantic_schema"] = ann["data"]["semantic_schema"]
            rec = validate_region(tx, data)
            data["media_sha256"] = rec["data"]["media"]["analysis"]["sha256"]
            recording_speakers[data["recording_id"]] = speaker_key(tx, rec, data["speaker_id"])
            items.append(data)
        if len({i["guideline_version"] for i in items}) != 1:
            raise Conflict("guideline_version: mixed manuals require explicit reconciliation")
        items.sort(key=lambda i: (i["recording_id"], i["start_ms"], i["end_ms"], i["source"]))
        coverage = {}
        previous, event_keys = {}, set()
        for item in items:
            rid = item["recording_id"]
            if rid in previous and item["start_ms"] < previous[rid]:
                message = "independent annotations cannot be pooled as ground truth" if kind == "task" else "ambiguous overlapping reviewed regions require reconciliation"
                raise Conflict(message)
            previous[rid] = item["end_ms"]
            coverage.setdefault(rid, []).append((item["start_ms"], item["end_ms"]))
            for event in item["events"]:
                key = (rid, event["start_ms"], event["end_ms"], event.get("label"), event.get("decision", "event"))
                if key in event_keys:
                    raise Conflict("ambiguous duplicate event across regions; reconcile ownership explicitly")
                event_keys.add(key)
        content = {"snapshot_schema": "mrcd-snapshot-v2", "name": request["name"], "include": request["include"],
                   "taxonomy_version": TAXONOMY_VERSION, "reference_kind": "human_adjudicated" if kind == "adjudication" else "human_single_annotation",
                   "coverage": {rid: union_intervals(spans) for rid, spans in sorted(coverage.items())},
                   "recording_speakers": dict(sorted(recording_speakers.items())), "items": items}
        digest = hashlib.sha256(canonical(content).encode()).hexdigest()
        snapshot_id = f"snp_{digest[:16]}"
        return tx.get(snapshot_id, "snapshot") or tx.create("snapshot", {**content, "sha256": digest},
                    id=snapshot_id, owner=actor_id, status="frozen", author=actor_id)
