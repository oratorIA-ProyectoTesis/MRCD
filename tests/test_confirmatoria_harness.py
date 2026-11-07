"""Confirmatory harness: reference/statistics parity with the repo modules, gates, lock and a synthetic dry run."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

HARNESS = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "confirmatoria"
    / "run_confirmatoria.py"
)


@pytest.fixture(scope="module")
def H():
    spec = importlib.util.spec_from_file_location("run_confirmatoria", HARNESS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


TEXTS = [
    "la casa la casa es grande",
    "eh y luego fuimos",
    "no no quiero ir",
    "e indigenas y hijos",
    "bueno hm pues hm eso",
    "el el perro corre ehm no se",
    "uno dos tres uno dos tres",
    "",
]


@pytest.mark.parametrize("text", TEXTS)
def test_reference_parity_with_repo(H, text):
    from mrcd.context_eval.transcript_reference import utterance_reference

    r, mine = utterance_reference(text), H.reference_of(text)
    assert mine["rep_literal"] == r["rep_literal"]
    assert mine["n_fillers"] == r["n_fillers"]


def test_stats_parity_with_repo(H):
    from mrcd.context_eval import metrics as M

    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    p1 = np.where(rng.random(300) < 0.8, y, 1 - y)
    p0 = np.where(rng.random(300) < 0.6, y, 1 - y)
    g = rng.integers(0, 12, 300)
    assert H.boot_ci(H.group_counts(y, p1, g), H.f1_of) == M.boot_f1_ci(y, p1, g)
    mine = H.paired_delta(H.group_counts(y, p1, g), H.group_counts(y, p0, g), H.f1_of)
    ref = M.paired_f1_delta(y, p1, p0, g)
    assert mine["delta"] == pytest.approx(ref["delta"]) and mine["ci"] == ref["ci"]
    assert mine["p_two"] == pytest.approx(ref["p_boot"])
    assert H.holm([0.01, 0.04, 0.03]) == M.holm([0.01, 0.04, 0.03])


def test_holm_family_size_is_fixed_even_with_nan(H):
    out = H.holm([0.01, float("nan"), 0.02, 0.5])
    assert out[0] == pytest.approx(0.04) and np.isnan(out[1])  # m = 4, not 3


LONG = """File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 4
tiers? <exists>
size = 2
item []:
    item [1]:
        class = "IntervalTier"
        name = "transcript"
        xmin = 0
        xmax = 4
        intervals: size = 2
        intervals [1]:
            xmin = 0
            xmax = 1.5
            text = "hola hola"
        intervals [2]:
            xmin = 1.5
            xmax = 4
            text = ""
    item [2]:
        class = "IntervalTier"
        name = "annotation"
        xmin = 0
        xmax = 4
        intervals: size = 1
        intervals [1]:
            xmin = 0
            xmax = 4
            text = "otra"
"""

SHORT = """File type = "ooTextFile"
Object class = "TextGrid"

0
4
<exists>
2
"IntervalTier"
"Tras"
0
4
2
0
1.5
"hola hola"
1.5
4
""
"IntervalTier"
"syntax"
0
4
1
0
4
"otra"
"""


def test_textgrid_first_tier_long_and_short(H):
    want = [(0.0, 1.5, "hola hola"), (1.5, 4.0, "")]
    assert H.parse_textgrid_first_tier(LONG) == want
    assert H.parse_textgrid_first_tier(SHORT) == want


def test_assign_max_overlap_and_drop(H):
    units = [(0, {"start": 0, "end": 2}), (1, {"start": 2, "end": 4})]
    hit, dropped = H.assign_to_units(units, [(1.5, 3.2), (9, 10)])
    assert hit == {1} and dropped == 1  # 0.5 s in unit 0, 1.2 s in unit 1


def test_gate_blocks_underpowered_class(H, tmp_path):
    cell = H.synthetic_cell(tmp_path)
    cell.units = cell.units[:20]  # far below the 30-positive gate
    res = H.score_cell(
        cell,
        H._StubAnalyzer(),
        H._StubOptions,
        ["o2026_11", "v0"],
        progress=lambda *_: None,
    )
    out = H.evaluate_nocando(cell, res, "o2026_11", "v0")
    assert out["classes"]["repetition"]["gate"]["evaluable"] is False
    assert (
        out["classes"]["repetition"]["tests"]["H2_default_vs_rule"]["verdict"]
        == "not evaluable"
    )


def test_dry_run_end_to_end(H, tmp_path):
    scratch = tmp_path / "scratch"
    assert H.main(["dry-run", "--scratch", str(scratch)]) == 0
    res = json.loads((scratch / "DRY_RUN_RESULT.json").read_text("utf8"))
    assert res["DRY_RUN"].startswith("SYNTHETIC")
    nc = res["confirmatory_nocando"]
    assert nc["holm_family"]["size"] == 4
    assert nc["classes"]["repetition"]["gate"]["evaluable"] is True
    assert (scratch / "local" / "nocando_unit_predictions.csv").is_file()
    assert not (
        H.OUT_DIR / H.LOCK_NAME
    ).exists()  # the dry run never touches the official lock


def test_dry_run_refuses_official_output_dir(H):
    with pytest.raises(SystemExit):
        H.main(["dry-run", "--scratch", str(H.OUT_DIR / "x")])


def test_lock_is_single_use(H, tmp_path):
    H.acquire_lock(tmp_path, {"a": 1}, None)
    with pytest.raises(SystemExit):
        H.acquire_lock(tmp_path, {"a": 1}, None)
    annex = tmp_path / "ANNEX.md"
    annex.write_text("crash at recording 7", "utf8")
    H.acquire_lock(
        tmp_path, {"a": 1}, str(annex)
    )  # started + annex: crash recovery allowed
    (tmp_path / H.RESULT_NAME).write_text("{}", "utf8")
    with pytest.raises(SystemExit):
        H.acquire_lock(tmp_path, {"a": 1}, str(annex))  # a result exists: never again


def test_run_refuses_without_freeze(H, tmp_path, monkeypatch):
    monkeypatch.setattr(H, "PROTOCOL_FREEZE", tmp_path / "nope.json")
    with pytest.raises(SystemExit):
        H.check_protocol_hashes()


def test_run_refuses_when_protocol_changed(H, tmp_path, monkeypatch):
    proto = tmp_path / "P.md"
    proto.write_text("v1", "utf8")
    fz = tmp_path / "F.json"
    fz.write_text(
        json.dumps(
            {
                "protocol_sha256": H.sha256_text_file(proto),
                "harness_sha256": H.sha256_text_file(H.SELF),
            }
        ),
        "utf8",
    )
    monkeypatch.setattr(H, "PROTOCOL_MD", proto)
    monkeypatch.setattr(H, "PROTOCOL_FREEZE", fz)
    assert H.check_protocol_hashes()
    proto.write_text("v2", "utf8")
    with pytest.raises(SystemExit):
        H.check_protocol_hashes()


def test_system_tree_changes_with_code(H, tmp_path):
    def g(*a):
        subprocess.run(["git", *a], cwd=tmp_path, check=True, capture_output=True)

    g("init", "-q")
    g("config", "user.email", "t@example.org")
    g("config", "user.name", "t")
    (tmp_path / "mrcd").mkdir()
    (tmp_path / "mrcd" / "a.py").write_text("x = 1\n", "utf8")
    g("add", ".")
    g("commit", "-qm", "init")
    h1 = H.system_tree(tmp_path)["sha256"]
    assert H.system_state_clean(tmp_path)
    (tmp_path / "mrcd" / "a.py").write_text("x = 2\n", "utf8")
    assert not H.system_state_clean(tmp_path)
    assert H.system_tree(tmp_path)["sha256"] != h1


def test_is_ancestor(H, tmp_path):
    def g(*a):
        return subprocess.run(["git", *a], cwd=tmp_path, check=True, capture_output=True, text=True).stdout.strip()

    g("init", "-q")
    g("config", "user.email", "t@example.org")
    g("config", "user.name", "t")
    (tmp_path / "f").write_text("1", "utf8")
    g("add", ".")
    g("commit", "-qm", "one")
    first = g("rev-parse", "HEAD")
    (tmp_path / "f").write_text("2", "utf8")
    g("commit", "-qam", "two")
    assert H.is_ancestor(first, tmp_path)
    g("checkout", "-q", "--orphan", "other")
    g("commit", "-qm", "unrelated", "--allow-empty")
    assert not H.is_ancestor(first, tmp_path)


def test_freeze_and_verify_cycle(H, tmp_path, monkeypatch):
    """freeze-protocol -> freeze-system -> commit -> verify passes; any change to code, protocol or history fails."""
    import argparse

    def g(*a):
        return subprocess.run(["git", *a], cwd=tmp_path, check=True, capture_output=True, text=True).stdout.strip()

    g("init", "-q")
    g("config", "user.email", "t@example.org")
    g("config", "user.name", "t")
    out = tmp_path / "results" / "c"
    out.mkdir(parents=True)
    (tmp_path / "mrcd").mkdir()
    (tmp_path / "mrcd" / "a.py").write_text("x = 1\n", "utf8")
    (tmp_path / "pyproject.toml").write_text("[project]\n", "utf8")
    harness = tmp_path / "harness.py"
    harness.write_text("# stand-in\n", "utf8")
    proto = out / "PROTOCOL_CONFIRMATORIO.md"
    proto.write_text("protocol v1\n", "utf8")
    for name, val in (("REPO", tmp_path), ("OUT_DIR", out), ("PROTOCOL_MD", proto), ("SELF", harness),
                      ("PROTOCOL_FREEZE", out / "PROTOCOL.sha256.json"), ("SYSTEM_FREEZE", out / "SYSTEM_FREEZE.json")):
        monkeypatch.setattr(H, name, val)
    monkeypatch.setattr(H, "data_fingerprint", lambda root: {})
    monkeypatch.setattr(H, "default_profile_name", lambda: "o2026_11")
    args = argparse.Namespace(corpora_root=str(tmp_path), system_commit=None)

    g("add", ".")
    g("commit", "-qm", "base")
    args.system_commit = g("rev-parse", "HEAD")[:12]
    assert H.cmd_freeze_protocol(args) == 0
    g("add", "-f", ".")
    g("commit", "-qm", "protocol freeze")
    args.system_commit = g("rev-parse", "HEAD")[:12]  # the merged system is HEAD when freeze-system runs
    assert H.cmd_freeze_system(args) == 0
    assert json.loads((out / "SYSTEM_FREEZE.json").read_text("utf8"))["system_commit"] == g("rev-parse", "HEAD")
    with pytest.raises(SystemExit):
        H.verify_all(args, check_data=False)  # SYSTEM_FREEZE.json not committed yet
    g("add", "-f", ".")
    g("commit", "-qm", "system freeze")
    assert H.verify_all(args, check_data=False)["system_freeze"]["system_commit"]
    proto.write_text("protocol v2\n", "utf8")
    with pytest.raises(SystemExit):
        H.verify_all(args, check_data=False)
    proto.write_text("protocol v1\n", "utf8")
    (tmp_path / "mrcd" / "a.py").write_text("x = 2\n", "utf8")
    g("commit", "-qam", "tamper")
    with pytest.raises(SystemExit):
        H.verify_all(args, check_data=False)


def test_decode_textgrid_encodings(H):
    s = 'text = "canción"'
    assert H.decode_textgrid(s.encode("utf-16")) == s  # BOM, as the NOCANDO files from berta_es_2 on
    assert H.decode_textgrid(s.encode("utf-8")) == s
    assert H.decode_textgrid(s.encode("latin-1")) == s  # not valid UTF-8: must not be read as UTF-16
