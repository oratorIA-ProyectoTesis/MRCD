"""Explicit human semantics; proposed labels and legacy decisions are not truth."""
from copy import deepcopy
from core.constants import TAXONOMY

HUMAN_ANNOTATION_SCHEMA = "mrcd-human-annotation-v1"
HUMAN_EVENT_SCHEMA = "mrcd-human-event-v1"
REVIEW_FIELDS = ("start_ms", "end_ms", "label", "decision", "uncertainty", "type_known", "is_disfluent")


def reviewed_content(event):
    return {key: event.get(key) for key in REVIEW_FIELDS}


def annotation_semantics(data):
    """Fail closed on recognizable new claims, not on genuine historical marks.

    Historical contextual.is_disfluent and unbound suggestion_id predate v1;
    neither is an authenticated v1 attestation on its own.
    """
    for key in ("events", "contextual", "actions"):
        values = data.get(key, [])
        if not isinstance(values, list) or any(not isinstance(value, dict) for value in values):
            raise ValueError(f"{key}: expected a list of objects")
    schema = data.get("semantic_schema")
    if schema is not None:
        if schema != HUMAN_ANNOTATION_SCHEMA:
            raise ValueError("unsupported explicit human annotation schema")
        return "explicit_v1"
    if data.get("semantic_transition") or data.get("lineage_restore") or any(a.get("type") in (
            "confirm_human_event", "revoke_human_confirmation", "upgrade_human_semantics")
            for a in data.get("actions", [])):
        raise ValueError("explicit human schema required for human confirmation claims")
    for kind in ("events", "contextual"):
        for event in data.get(kind, []):
            if (any(event.get(key) is not None for key in (
                    "human_event_schema", "type_known", "contextual_state", "proposal_ref", "legacy_original", "source_lineage"))
                    or event.get("provenance")
                    or event.get("source_kind") in ("human_verified_model_suggestion", "human_assisted_manual",
                                                    "human_unconfirmed", "model_proposal_copy")
                    or kind == "events" and event.get("is_disfluent") is not None):
                raise ValueError("explicit human schema required for proposal/type/provenance claims")
    return "historical"


def validate_human_event(event):
    if event.get("human_event_schema") != HUMAN_EVENT_SCHEMA:
        raise ValueError("unsupported explicit human event schema")
    a, b = event.get("start_ms"), event.get("end_ms")
    if type(a) is not int or type(b) is not int or not 0 <= a < b:
        raise ValueError("human event requires original integer millisecond bounds")
    label, known, context = event.get("label"), event.get("type_known"), event.get("is_disfluent")
    if label is not None and label not in TAXONOMY or type(known) is not bool:
        raise ValueError("human phenotype must have canonical label and explicit known flag")
    decision, uncertainty = event.get("decision", "event"), event.get("uncertainty")
    if decision not in ("event", "uncertain", "not_evaluable") or uncertainty not in (None, "class", "boundaries", "audio"):
        raise ValueError("unknown human decision/uncertainty")
    if known and (label is None or decision == "not_evaluable"):
        raise ValueError("known phenotype contradicts absent label or non-evaluable input")
    if context is not None and type(context) is not bool:
        raise ValueError("human contextual judgment must be boolean or null")
    expected = None if context is None else "disfluent" if context else "functional"
    if event.get("contextual_state") != expected:
        raise ValueError("human context fields disagree")
    provenance = event.get("provenance", {})
    if not provenance.get("confirmed") and (known or context is not None):
        raise ValueError("human conclusions require an explicit confirmation action")
    return deepcopy(event)
