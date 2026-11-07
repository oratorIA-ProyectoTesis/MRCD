"""The single-file TypeScript client must not drift from the JSON Schema of the contract."""

from __future__ import annotations

import re
from pathlib import Path

from mrcd.service.contract import CONTRACT_REVISION, ERROR_CODES, SCHEMA

TS = (
    Path(__file__).resolve().parents[1] / "clients" / "typescript" / "mrcd.ts"
).read_text(encoding="utf-8")


def _union(name: str) -> set[str]:
    m = re.search(rf"export type {name} =(.*?);", TS, re.S)
    assert m, name
    return set(re.findall(r'"([^"]+)"', m.group(1)))


def test_error_codes_match_the_schema():
    assert (
        _union("ErrorCode")
        == set(SCHEMA["$defs"]["error"]["properties"]["code"]["enum"])
        == set(ERROR_CODES)
    )


def test_decision_profiles_and_states_match_the_schema():
    assert _union("DecisionProfileName") == set(
        SCHEMA["$defs"]["request"]["properties"]["decision_profile"]["enum"]
    )
    states = SCHEMA["$defs"]["decision"]["properties"]["state"]["enum"]
    assert all(f'"{s}"' in TS for s in states)


def test_contract_revision_and_required_fields():
    assert f'CONTRACT_REVISION = "{CONTRACT_REVISION}"' in TS
    for field in SCHEMA["$defs"]["analysis"]["required"]:
        assert re.search(rf"\b{field}\??:", TS), field
    for field in SCHEMA["$defs"]["event"]["required"]:
        assert re.search(rf"\b{field}\??:", TS), field
