"""Minimal JSON Schema validator (subset) used for the MRCD service contract.

``jsonschema`` is not installed in the project environment, so this module implements only what the
contract uses: ``$ref`` (local ``#/$defs/...``), ``type``, ``enum``, ``const``, ``required``,
``properties``, ``additionalProperties``, ``items``, ``minItems``, ``minimum``, ``maximum``,
``minLength``, ``pattern`` and ``oneOf``. An unknown keyword raises, so the schema can never silently
rely on a rule this validator does not enforce.
"""

from __future__ import annotations

import math
import re
from typing import Any

_KNOWN = {
    "$schema", "$id", "$defs", "$ref", "title", "description", "type", "enum", "const", "required",
    "properties", "additionalProperties", "items", "minItems", "minimum", "maximum", "minLength",
    "pattern", "oneOf", "default",
}


def _is_type(value: Any, name: str) -> bool:
    if name == "null":
        return value is None
    if name == "boolean":
        return isinstance(value, bool)
    if name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if name == "number":
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)  # NaN / Infinity are not JSON numbers
        )
    if name == "string":
        return isinstance(value, str)
    if name == "array":
        return isinstance(value, list)
    if name == "object":
        return isinstance(value, dict)
    raise ValueError(f"unsupported type {name!r} in schema")


def _resolve(ref: str, root: dict) -> dict:
    if not ref.startswith("#/"):
        raise ValueError(f"only local $ref is supported, got {ref!r}")
    node: Any = root
    for part in ref[2:].split("/"):
        node = node[part]
    return node


def _check(value: Any, schema: dict, root: dict, path: str, errors: list[str]) -> None:
    unknown = set(schema) - _KNOWN
    if unknown:
        raise ValueError(f"unsupported schema keyword(s) at {path}: {sorted(unknown)}")
    if "$ref" in schema:
        _check(value, _resolve(schema["$ref"], root), root, path, errors)
        return
    if "oneOf" in schema:
        matches = 0
        for sub in schema["oneOf"]:
            sub_errors: list[str] = []
            _check(value, sub, root, path, sub_errors)
            matches += not sub_errors
        if matches != 1:
            errors.append(f"{path}: must match exactly one alternative of oneOf (matched {matches})")
            return
    if "type" in schema:
        names = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not any(_is_type(value, n) for n in names):
            errors.append(f"{path}: expected type {names}, got {type(value).__name__}")
            return
    if "const" in schema and (value != schema["const"] or type(value) is not type(schema["const"])):
        errors.append(f"{path}: must equal {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: must be one of {schema['enum']}")
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            errors.append(f"{path}: shorter than minLength {schema['minLength']}")
        if "pattern" in schema and not re.search(schema["pattern"], value):
            errors.append(f"{path}: does not match pattern {schema['pattern']!r}")
    if _is_type(value, "number"):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: below minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: above maximum {schema['maximum']}")
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            errors.append(f"{path}: fewer than minItems {schema['minItems']}")
        if "items" in schema:
            for i, item in enumerate(value):
                _check(item, schema["items"], root, f"{path}[{i}]", errors)
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}: missing required property {key!r}")
        props = schema.get("properties", {})
        for key, sub in props.items():
            if key in value:
                _check(value[key], sub, root, f"{path}.{key}", errors)
        extra = schema.get("additionalProperties", True)
        for key in value:
            if key in props:
                continue
            if extra is False:
                errors.append(f"{path}: unexpected property {key!r}")
            elif isinstance(extra, dict):
                _check(value[key], extra, root, f"{path}.{key}", errors)


def validate(instance: Any, schema: dict, *, ref: str | None = None) -> list[str]:
    """Return the list of violations (empty = valid). ``ref`` selects ``#/$defs/<name>`` as entry point."""
    entry = _resolve(ref, schema) if ref else schema
    errors: list[str] = []
    _check(instance, entry, schema, "$", errors)
    return errors
