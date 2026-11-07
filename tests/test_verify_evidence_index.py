"""Fast tests for scripts/verify_evidence_index.py (no models, no network)."""

from __future__ import annotations

import hashlib
import importlib.util
import subprocess
import sys

import pytest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "verify_evidence_index.py"

_spec = importlib.util.spec_from_file_location("verify_evidence_index", SCRIPT)
vei = importlib.util.module_from_spec(_spec)
sys.modules["verify_evidence_index"] = vei
_spec.loader.exec_module(vei)

HEADER = "| id | area | claim | path | sha256 | status | review | date | versioned |\n|---|---|---|---|---|---|---|---|---|\n"


def _row(
    rid: str, rel: str, digest: str, status: str = "self", versioned: str = "no"
) -> str:
    return f"| {rid} | OE4 | a claim | `{rel}` | `{digest}` | {status} | - | 2026-11-07 | {versioned} |\n"


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def test_real_index_passes_in_this_checkout():
    if not (REPO / "results" / "dx_20261102" / "REVISION_DX.md").is_file():
        pytest.skip("the reports indexed by EVIDENCIAS.md (and the unsanitized docs they hash) are not distributed in the public repository")
    r = subprocess.run(
        [sys.executable, str(SCRIPT), "-q"], capture_output=True, text=True, cwd=REPO
    )
    assert r.returncode == 0, r.stderr


def test_ok_and_crlf_normalisation(tmp_path):
    (tmp_path / "a.md").write_bytes(b"line1\r\nline2\r\n")
    idx = tmp_path / "EVIDENCIAS.md"
    idx.write_text(
        HEADER + _row("E01", "a.md", _sha(b"line1\nline2\n")), encoding="utf-8"
    )
    assert vei.verify(tmp_path, idx, quiet=True) == 0


def test_hash_mismatch_fails(tmp_path):
    (tmp_path / "a.json").write_text("{}", encoding="utf-8")
    idx = tmp_path / "EVIDENCIAS.md"
    idx.write_text(HEADER + _row("E01", "a.json", "0" * 64), encoding="utf-8")
    assert vei.verify(tmp_path, idx, quiet=True) == 1


def test_missing_versioned_file_fails_but_unversioned_is_skipped(tmp_path):
    idx = tmp_path / "EVIDENCIAS.md"
    idx.write_text(
        HEADER + _row("E01", "gone.md", "1" * 64, versioned="no"), encoding="utf-8"
    )
    assert vei.verify(tmp_path, idx, quiet=True) == 0
    assert vei.verify(tmp_path, idx, strict_all=True, quiet=True) == 1
    idx.write_text(
        HEADER + _row("E01", "gone.md", "1" * 64, versioned="yes"), encoding="utf-8"
    )
    assert vei.verify(tmp_path, idx, quiet=True) == 1


def test_bad_status_duplicate_id_and_empty_index_fail(tmp_path):
    (tmp_path / "a.md").write_text("x", encoding="utf-8")
    h = _sha(b"x")
    idx = tmp_path / "EVIDENCIAS.md"
    idx.write_text(HEADER + _row("E01", "a.md", h, status="maybe"), encoding="utf-8")
    assert vei.verify(tmp_path, idx, quiet=True) == 1
    idx.write_text(
        HEADER + _row("E01", "a.md", h) + _row("E01", "a.md", h), encoding="utf-8"
    )
    assert vei.verify(tmp_path, idx, quiet=True) == 1
    idx.write_text("no rows here\n", encoding="utf-8")
    assert vei.verify(tmp_path, idx, quiet=True) == 1


def test_cli_exit_code_non_zero_on_failure(tmp_path):
    idx = tmp_path / "EVIDENCIAS.md"
    idx.write_text(
        HEADER + _row("E01", "gone.md", "2" * 64, versioned="yes"), encoding="utf-8"
    )
    r = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--root",
            str(tmp_path),
            "--index",
            str(idx),
            "-q",
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 1 and "FAIL" in r.stderr
