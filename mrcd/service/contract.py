"""MRCD service contract v1: schema (single source of truth), constants and error catalogue.

The schema lives here as a Python dict so it ships with the package (pyproject packages only ``mrcd*``
python modules); ``schemas/mrcd.analysis.v1.schema.json`` is generated from it and a test keeps both
identical.

What the contract deliberately does NOT contain
-----------------------------------------------
* No "uncertain" state: the engine emits one class per window and no abstention, so the contract has none.
* No calibrated probability: every score carries ``calibrated: false`` and the label ``uncalibrated``.
* No psychological inference and no visual evidence: video is declared ``available: false`` with a reason.
* No ``0`` standing for "not measured": a measure is ``{available, value, unavailable_reason}`` and the
  value is ``null`` whenever ``available`` is false.
"""

from __future__ import annotations

ANALYSIS_SCHEMA_VERSION = "mrcd.analysis/v1"
JOB_SCHEMA_VERSION = "mrcd.job/v1"

#: error code -> (HTTP status the dev server uses, retryable)
ERROR_CODES: dict[str, tuple[int, bool]] = {
    "invalid_request": (400, False),
    "payload_too_large": (413, False),
    "invalid_audio_file": (422, False),
    "unsupported_audio_format": (422, False),
    "audio_not_mono": (422, False),
    "unsupported_sample_rate": (422, False),
    "audio_not_finite": (422, False),
    "audio_too_short": (422, False),
    "audio_too_long": (422, False),
    "checkpoint_unavailable": (503, False),
    "asr_unavailable": (503, False),
    "head_unavailable": (503, False),
    "engine_load_failed": (503, False),
    "analysis_failed": (500, True),
    "job_not_found": (404, False),
    "queue_full": (429, True),
}

JOB_STATUSES = ("queued", "running", "completed", "failed")

NOT_CALIBRATED_LABEL = "uncalibrated"
#: Revision of the v1 contract. 1.0 = original; 1.1 = additive: score.kind renamed (see SCORE_KIND_DEPRECATED).
CONTRACT_REVISION = "1.1"
#: Raw, uncalibrated softmax maximum of the window head. NOT a probability.
SCORE_KIND = "max_window_softmax_score"
#: Deprecated alias (revision 1.0 name). Still ACCEPTED by the schema so old documents/consumers validate;
#: new documents never emit it. Remove only in a future major contract (v2).
SCORE_KIND_DEPRECATED = "max_window_class_probability"
INTERVAL_SEMANTICS = "decision_stride_not_human_boundary"

_STR = {"type": "string"}
_NSTR = {"type": ["string", "null"]}
_BOOL = {"type": "boolean"}


def _measure(value_type: str = "number") -> dict:
    """A measurement that may be absent: ``value`` is null exactly when ``available`` is false."""
    return {
        "oneOf": [
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["available", "value", "unavailable_reason"],
                "properties": {
                    "available": {"const": True},
                    "value": {"type": value_type},
                    "unavailable_reason": {"type": "null"},
                },
            },
            {
                "type": "object",
                "additionalProperties": False,
                "required": ["available", "value", "unavailable_reason"],
                "properties": {
                    "available": {"const": False},
                    "value": {"type": "null"},
                    "unavailable_reason": {"type": "string", "minLength": 1},
                },
            },
        ]
    }


_MODALITY = {
    "oneOf": [
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["available", "reason"],
            "properties": {"available": {"const": True}, "reason": {"type": "null"}},
        },
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["available", "reason"],
            "properties": {
                "available": {"const": False},
                "reason": {"type": "string", "minLength": 1},
            },
        },
    ]
}

ANALYSIS_DEF = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema_version",
        "status",
        "generated_at",
        "error",
        "request",
        "events",
        "summary",
        "modalities",
        "provenance",
        "experimental",
        "head_decision",
        "transcript",
        "warnings",
    ],
    "properties": {
        "schema_version": {"const": ANALYSIS_SCHEMA_VERSION},
        "status": {"enum": ["completed", "failed"]},
        "generated_at": {"type": "string", "minLength": 20},
        "error": {"oneOf": [{"type": "null"}, {"$ref": "#/$defs/error"}]},
        "request": {"$ref": "#/$defs/request"},
        "events": {"type": "array", "items": {"$ref": "#/$defs/event"}},
        "summary": {"oneOf": [{"type": "null"}, {"$ref": "#/$defs/summary"}]},
        "modalities": {"$ref": "#/$defs/modalities"},
        "provenance": {"$ref": "#/$defs/provenance"},
        "experimental": {"$ref": "#/$defs/experimental"},
        "head_decision": {"type": ["object", "null"]},
        "transcript": {"type": ["string", "null"]},
        "warnings": {"type": "array", "items": {"$ref": "#/$defs/warning"}},
    },
}

SCHEMA: dict = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "urn:mrcd:service:analysis:v1",
    "title": "MRCD service contract v1 (analysis result and job status)",
    "description": (
        "Events are window-level decisions of an uncalibrated engine trained with weak supervision. "
        "Intervals are the decision stride of the window, not human-annotated boundaries."
    ),
    "oneOf": [{"$ref": "#/$defs/analysis"}, {"$ref": "#/$defs/job"}],
    "$defs": {
        "analysis": ANALYSIS_DEF,
        "job": {
            "type": "object",
            "additionalProperties": False,
            "required": ["schema_version", "job_id", "status", "created_at", "result"],
            "properties": {
                "schema_version": {"const": JOB_SCHEMA_VERSION},
                "job_id": {"type": "string", "pattern": "^[0-9a-f]{32}$"},
                "status": {"enum": list(JOB_STATUSES)},
                "created_at": {"type": "string", "minLength": 20},
                "result": {"oneOf": [{"type": "null"}, {"$ref": "#/$defs/analysis"}]},
            },
        },
        "error": {
            "type": "object",
            "additionalProperties": False,
            "required": ["code", "message", "retryable", "details"],
            "properties": {
                "code": {"enum": sorted(ERROR_CODES)},
                "message": {"type": "string", "minLength": 1},
                "retryable": _BOOL,
                "details": {"type": "object"},
            },
        },
        "request": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "audio_sha256",
                "audio_duration_s",
                "sample_rate_hz",
                "channels",
                "asr_profile",
                "head",
                "include_text",
            ],
            "properties": {
                "audio_sha256": {
                    "type": ["string", "null"],
                    "pattern": "^[0-9a-f]{64}$",
                },
                "audio_duration_s": _measure(),
                "sample_rate_hz": _measure("integer"),
                "channels": _measure("integer"),
                "asr_profile": _STR,
                "head": _NSTR,
                "include_text": _BOOL,
            },
        },
        "event": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "id",
                "class",
                "start_s",
                "end_s",
                "interval_semantics",
                "score",
                "n_windows",
                "modalities_used",
                "text",
                "validation",
                "experimental",
            ],
            "properties": {
                "id": {"type": "string", "pattern": "^ev-[0-9]{4,}$"},
                "class": {"type": "string", "minLength": 1},
                "start_s": {"type": "number", "minimum": 0},
                "end_s": {"type": "number", "minimum": 0},
                "interval_semantics": {"const": INTERVAL_SEMANTICS},
                "score": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["value", "kind", "calibrated", "label"],
                    "properties": {
                        "value": {"type": "number", "minimum": 0, "maximum": 1},
                        "kind": {"enum": [SCORE_KIND, SCORE_KIND_DEPRECATED]},
                        "calibrated": {"const": False},
                        "label": {"const": NOT_CALIBRATED_LABEL},
                    },
                },
                "n_windows": {"type": "integer", "minimum": 1},
                "modalities_used": {"type": "array", "items": _STR},
                "text": {"type": ["string", "null"]},
                "validation": _NSTR,
                "experimental": _BOOL,
            },
        },
        "summary": {
            "type": "object",
            "additionalProperties": False,
            "required": ["n_events", "events_per_minute", "counts_by_class"],
            "properties": {
                "n_events": {"type": "integer", "minimum": 0},
                "events_per_minute": _measure(),
                "counts_by_class": {
                    "type": "object",
                    "additionalProperties": {"type": "integer"},
                },
            },
        },
        "modalities": {
            "type": "object",
            "additionalProperties": False,
            "required": ["acoustic", "linguistic", "kinesic_video"],
            "properties": {
                "acoustic": _MODALITY,
                "linguistic": _MODALITY,
                "kinesic_video": _MODALITY,
            },
        },
        "provenance": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "service",
                "checkpoint",
                "asr",
                "head",
                "config",
                "environment",
            ],
            "properties": {
                "service": {"type": "object"},
                "checkpoint": {"type": ["object", "null"]},
                "asr": {"type": ["object", "null"]},
                "head": {"type": ["object", "null"]},
                "config": {"type": ["object", "null"]},
                "environment": {"type": ["object", "null"]},
            },
        },
        "experimental": {
            "type": "object",
            "additionalProperties": False,
            "required": ["active", "components"],
            "properties": {
                "active": _BOOL,
                "components": {"type": "array", "items": _STR},
            },
        },
        "warning": {
            "type": "object",
            "additionalProperties": False,
            "required": ["code", "message"],
            "properties": {
                "code": {"type": "string", "minLength": 1},
                "message": {"type": "string", "minLength": 1},
            },
        },
    },
}


def validate_document(doc: object) -> list[str]:
    """Validate an analysis result or a job status against the v1 contract. Empty list = valid."""
    from mrcd.service.validate import validate

    return validate(doc, SCHEMA)
