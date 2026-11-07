"""Secondary cw2 arm of the confirmatory run: synthetic fixtures only (no corpus, no GPU, no model, no ASR)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "confirmatoria"


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def S():
    return _load("run_secondary_cw2_t", "run_secondary_cw2.py")


@pytest.fixture(scope="module")
def H(S):
    return S.H


def make_primary(H, tmp_path):
    """A complete SYNTHETIC primary output directory (result, completed lock, unit csv) and its cell."""
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    cell = H.synthetic_cell(scratch)
    out = tmp_path / "out"
    out.mkdir()
    res = H.run_all(
        H._StubAnalyzer(),
        H._StubOptions,
        {"nocando": cell},
        "o2026_11",
        out,
        progress=lambda *_: None,
    )
    (out / H.RESULT_NAME).write_text(json.dumps(res, default=float), "utf8")
    (out / H.LOCK_NAME).write_text(
        json.dumps(
            {"status": "completed", "result_sha256": H.sha256_file(out / H.RESULT_NAME)}
        ),
        "utf8",
    )
    return out, cell


def tree_hashes(out: Path, H) -> dict:
    return {
        p.relative_to(out).as_posix(): H.sha256_file(p)
        for p in sorted(out.rglob("*"))
        if p.is_file()
    }


def ok_preflight():
    return None, {"gpu_name": "synthetic"}


def test_primary_harness_is_byte_identical_to_its_frozen_hash(H):
    """The primary path is unchanged: the frozen harness hash of PROTOCOL.sha256.json still matches this tree."""
    if not (H.OUT_DIR / "PROTOCOL.sha256.json").is_file():
        pytest.skip("results/ is not distributed in the public repository")
    fz = json.loads((H.OUT_DIR / "PROTOCOL.sha256.json").read_text("utf8"))
    assert H.sha256_text_file(H.SELF) == fz["harness_sha256"]
    assert H.sha256_text_file(H.PROTOCOL_MD) == fz["protocol_sha256"]


def test_refuses_before_primary_is_done(S, H, tmp_path):
    with pytest.raises(SystemExit):
        S.primary_complete(tmp_path)  # nothing there
    out, _ = make_primary(H, tmp_path)
    assert S.primary_complete(out)["status"] == "completed"
    (out / H.LOCK_NAME).write_text(
        json.dumps({"status": "started"}), "utf8"
    )  # primary crashed / running
    with pytest.raises(SystemExit):
        S.primary_complete(out)
    (out / H.LOCK_NAME).write_text(
        json.dumps({"status": "completed", "result_sha256": "0" * 64}), "utf8"
    )
    with pytest.raises(SystemExit):  # result file differs from the one the lock closed
        S.primary_complete(out)


def test_secondary_lock_is_single_use(S, tmp_path):
    S.acquire_secondary_lock(tmp_path, {"a": 1})
    with pytest.raises(SystemExit):
        S.acquire_secondary_lock(tmp_path, {"a": 1})  # no retries, even after a crash


def test_refuses_on_annex_hash_mismatch(S, H, tmp_path, monkeypatch):
    annex = tmp_path / "ANNEX.md"
    annex.write_text("annex v1\n", "utf8")
    helper = tmp_path / "helper.py"
    helper.write_text("x = 1\n", "utf8")
    fz = tmp_path / "ANNEX.sha256.json"
    monkeypatch.setattr(S, "ANNEX_MD", annex)
    monkeypatch.setattr(S, "ANNEX_FREEZE", fz)
    with pytest.raises(SystemExit):
        S.check_annex()  # not frozen
    monkeypatch.setattr(H, "REPO", tmp_path)
    fz.write_text(
        json.dumps(
            {
                "annex_sha256": H.sha256_text_file(annex),
                "files": {"helper.py": H.sha256_text_file(helper)},
            }
        ),
        "utf8",
    )
    assert S.check_annex()["annex_sha256"] == H.sha256_text_file(annex)
    annex.write_text("annex v2\n", "utf8")
    with pytest.raises(SystemExit):
        S.check_annex()
    annex.write_text("annex v1\n", "utf8")
    helper.write_text("x = 2\n", "utf8")  # a frozen harness/test file changed
    with pytest.raises(SystemExit):
        S.check_annex()


def test_real_annex_is_frozen_and_pins_the_system(S, H):
    """The committed annex, its manifest and the pinned cw2 identity agree."""
    if not S.ANNEX_FREEZE.is_file():
        pytest.skip("annex not yet committed (work unit a); the annex commit adds it and this test then runs")
    fz = S.check_annex()
    text = S.ANNEX_MD.read_text("utf8")
    for pinned in (S.CW2_MODEL_ID, S.CW2_MODEL_REVISION, S.CW2_WEIGHTS_MANIFEST_SHA256):
        assert pinned in text
    assert "scripts/confirmatoria/run_secondary_cw2.py" in fz["files"]
    assert (
        "scripts/confirmatoria/run_confirmatoria.py" not in fz["files"]
    )  # the primary harness is never re-frozen here


@pytest.mark.parametrize("mode", ["preflight", "crash", "recording_error"])
def test_not_available_path(S, H, tmp_path, mode):
    out, cell = make_primary(H, tmp_path)
    S.acquire_secondary_lock(out, {})

    class Boom:
        def analyze_bytes(self, *a, **k):
            raise RuntimeError("CUDA out of memory")

    class Err:
        def analyze_bytes(self, *a, **k):
            return {"error": {"code": "asr_unavailable"}, "events": []}

    pre = (
        (lambda: ("CUDA is not available on this machine", {"gpu_name": None}))
        if mode == "preflight"
        else ok_preflight
    )
    make = {"preflight": H._StubAnalyzer, "crash": Boom, "recording_error": Err}[mode]
    final = S.run_secondary(
        cell,
        out,
        make,
        H._StubOptions,
        {"annex_sha256": "x"},
        preflight=pre,
        progress=lambda *_: None,
    )
    assert final["status"] == "not_available" and final["reason"]
    saved = json.loads((out / S.RESULT_SECONDARY).read_text("utf8"))
    assert saved["status"] == "not_available"
    assert (
        "secondary_tests" not in saved and "classes" not in saved
    )  # never a fabricated 0
    assert (
        json.loads((out / S.LOCK_SECONDARY).read_text("utf8"))["status"] == "completed"
    )  # and no retry possible


def test_holm_family_is_s1_s2_only(S, H, tmp_path):
    out, cell = make_primary(H, tmp_path)
    S.acquire_secondary_lock(out, {})
    final = S.run_secondary(
        cell,
        out,
        H._StubAnalyzer,
        H._StubOptions,
        {},
        preflight=ok_preflight,
        progress=lambda *_: None,
    )
    assert final["status"] == "ok"
    fam = final["holm_family"]
    assert fam["size"] == 2 and fam["members"] == S.FAMILY and fam["alpha"] == 0.05
    assert set(final["secondary_tests"]) == set(S.FAMILY) and len(fam["p_holm"]) == 2
    assert not any(
        k.startswith("H") for k in final["secondary_tests"]
    )  # nothing of the primary family
    for t in final["secondary_tests"].values():
        assert t["sides"] == "two-sided" and t["verdict"] in (
            "supported",
            "not demonstrated",
        )
        assert t["p_holm"] >= t["p_raw"]
    assert "descriptive_cw2_rule_vs_p0_rule" in final["classes"]["repetition"]
    assert (
        "p_two" not in final["classes"]["repetition"]["descriptive_cw2_rule_vs_p0_rule"]
    )  # descriptive, no test
    # Holm over 2: the smaller p is multiplied by 2, the larger by 1 (monotone)
    p = fam["p_raw"]
    assert max(p) <= max(fam["p_holm"]) and min(fam["p_holm"]) == pytest.approx(
        min(1.0, 2 * min(p))
    )


def test_underpowered_class_is_not_evaluable_but_still_counts_in_the_family(S, H, tmp_path):
    out, cell = make_primary(H, tmp_path)
    cell.units = [u for u in cell.units if u["recording"] == "anna_es_1"][:10]
    ids = {u["unit_id"] for u in cell.units}
    rows = {k: v for k, v in S.read_primary_predictions(out).items() if k in ids}
    zeros = {c: np.zeros(len(cell.units), int) for c in ("repetition", "filler_word")}
    res = S.evaluate_secondary(cell, {**zeros, "dropped_events": 0}, rows)
    assert res["holm_family"]["size"] == 2
    assert all(t["verdict"] == "not evaluable" for t in res["secondary_tests"].values())


def test_primary_output_is_byte_identical_with_and_without_secondary(S, H, tmp_path):
    out, cell = make_primary(H, tmp_path)
    before = tree_hashes(out, H)
    S.acquire_secondary_lock(out, {})
    S.run_secondary(
        cell,
        out,
        H._StubAnalyzer,
        H._StubOptions,
        {},
        preflight=ok_preflight,
        progress=lambda *_: None,
    )
    after = tree_hashes(out, H)
    new = set(after) - set(before)
    assert new == {S.RESULT_SECONDARY, S.LOCK_SECONDARY}
    assert {
        k: v for k, v in after.items() if k in before
    } == before  # every primary file untouched, byte for byte
