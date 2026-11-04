"""Keep the agent-facing docs (.claude/skills) consistent with the service code."""

import re
from pathlib import Path

from mrcd.service import server
from mrcd.service.contract import ERROR_CODES

SKILLS = Path(__file__).resolve().parents[1] / ".claude" / "skills"
TROUBLESHOOT = SKILLS / "mrcd-troubleshoot" / "SKILL.md"
INTEGRATE = SKILLS / "mrcd-integrate" / "SKILL.md"


def _first_column_codes(text: str) -> set[str]:
    codes = set()
    for line in text.splitlines():
        if line.startswith("|"):
            cell = line.split("|")[1]
            codes.update(re.findall(r"`([a-z]+(?:_[a-z]+)+)`", cell))
    return {c for c in codes if not c.endswith("_cuda")}


def test_troubleshoot_error_codes_exist_in_contract():
    codes = _first_column_codes(TROUBLESHOOT.read_text(encoding="utf-8"))
    assert "checkpoint_unavailable" in codes
    assert codes <= set(ERROR_CODES), codes - set(ERROR_CODES)


def test_integrate_error_codes_exist_in_contract():
    text = INTEGRATE.read_text(encoding="utf-8")
    named = set(re.findall(r"`([a-z]+(?:_[a-z]+)+)`", text)) & {
        "invalid_request", "payload_too_large", "queue_full", "job_not_found", "analysis_failed",
    }
    assert named and named <= set(ERROR_CODES)


def test_documented_limits_match_server_constants():
    troubleshoot = TROUBLESHOOT.read_text(encoding="utf-8")
    integrate = INTEGRATE.read_text(encoding="utf-8")
    assert server.MAX_QUEUED_JOBS == 8 and "8 jobs pending" in troubleshoot
    assert server.MAX_RETAINED_JOBS == 32 and "most recent 32" in integrate
    assert server.DEFAULT_MAX_BODY_BYTES == 64 * 1024 * 1024 and "64 MiB" in integrate
