"""ASR-survival benchmark artefact: frame schema, prompts, frozen results and hand-checked survival cases.

Everything here is synthetic or aggregate: no DIHANA text is needed or read.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parents[1] / "benchmarks" / "asr_survival"
sys.path.insert(0, str(BENCH / "scripts"))

import survival_bench as sb  # noqa: E402

from mrcd.asr_recovery import decode  # noqa: E402
from mrcd.constants import WHISPER_INITIAL_PROMPT  # noqa: E402
from mrcd.context_eval.survival import turn_survives  # noqa: E402


# ------------------------------------------------------------------ frame
def test_frame_schema_and_counts():
    with (BENCH / "frame" / "turn_ids.csv").open(encoding="utf8", newline="") as f:
        header = next(csv.reader(f))
    assert tuple(header) == sb.FRAME_COLUMNS
    frame = sb.load_frame()
    assert sb.check_frame(frame) == sb.EXPECTED
    assert len({r["turn_id"] for r in frame}) == len(frame)


def test_frame_holds_identifiers_and_labels_only():
    """No free text: every cell is short and matches a closed pattern."""
    pats = {
        "turn_id": r"[A-Z]\d{3}_[A-Za-z0-9]+_U_\d{3}",
        "speaker_id": r"[A-Z]\d{3}",
        "fold": r"[1-4]",
        "kind": r"pos|neg",
        "sampling_weight": r"\d+(\.\d+)?",
        "rep_literal": r"[01]?",
        "revision": r"[01]",
        "filler_present": r"[01]",
    }
    with (BENCH / "frame" / "turn_ids.csv").open(encoding="utf8", newline="") as f:
        for row in csv.DictReader(f):
            for k, pat in pats.items():
                assert re.fullmatch(pat, row[k]), (k, row[k])


# ------------------------------------------------------------------ prompts
@pytest.mark.parametrize(
    "name,expected",
    [
        ("P0", WHISPER_INITIAL_PROMPT),
        ("P1", decode.P1_PROMPT),
        ("C1", decode.C1_PROMPT),
    ],
)
def test_prompt_files_match_code(name, expected):
    got = (BENCH / "prompts" / f"{name}.txt").read_bytes().decode("utf8")
    assert got == expected
    assert decode.VARIANTS[name]["initial_prompt"] == expected


def test_p_none_has_no_prompt():
    assert decode.VARIANTS["p_none"]["initial_prompt"] is None


# ------------------------------------------------------------------ frozen results
def test_protocol_hash_and_metrics_agree():
    sha = hashlib.sha256((BENCH / "results" / "PROTOCOL.md").read_bytes()).hexdigest()
    rec = json.loads((BENCH / "results" / "PROTOCOL.sha256.json").read_text("utf8"))
    metrics = json.loads((BENCH / "results" / "metrics.json").read_text("utf8"))
    assert sha == rec["sha256"] == metrics["protocol_sha256"]
    assert metrics["frame"]["units"] == sb.EXPECTED["units"]
    assert metrics["frame"]["rep_literal_pos"] == sb.EXPECTED["rep_literal_pos"]


def test_metrics_json_has_no_free_text():
    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                yield from walk(v)
        elif isinstance(o, list):
            for v in o:
                yield from walk(v)
        elif isinstance(o, str):
            yield o

    metrics = json.loads((BENCH / "results" / "metrics.json").read_text("utf8"))
    for s in walk(metrics):
        assert len(s) <= 64 and (" " not in s or len(s.split()) <= 8), s


# ------------------------------------------------------------------ survival, hand-checked
REP = {
    "subtype": "A_literal_repeat",
    "span": "quiero quiero",
    "m": "quiero",
    "c": "quiero",
}
REV = {
    "subtype": "sustitucion",
    "span": "a las ocho digo a las nueve",
    "mt": ["a", "las", "ocho"],
    "ct": ["a", "las", "nueve"],
}


def test_repetition_survives_only_if_adjacent_duplicate_kept():
    assert turn_survives(
        "repetition", [REP], "quiero quiero un billete", {"A_literal_repeat"}
    )
    assert not turn_survives(
        "repetition", [REP], "quiero un billete", {"A_literal_repeat"}
    )


def test_repetition_subtype_filter():
    assert not turn_survives(
        "repetition", [REP], "quiero quiero un billete", {"B_cut_restart"}
    )


def test_revision_needs_both_reparandum_and_correction():
    both = "a las ocho digo a las nueve"
    assert turn_survives("revision", [REV], both, None)
    assert not turn_survives("revision", [REV], "a las nueve", None)


def test_empty_asr_text_never_survives():
    assert not turn_survives("repetition", [REP], "", None)


@pytest.mark.parametrize(
    "tok,nxt,expected",
    [
        ("eh", None, True),
        ("mmm", "bueno", True),
        ("este", None, False),  # lexical, ambiguous: never a filler
        ("e", "indigenas", False),  # conjunction
        ("e", "porque", True),
    ],
)
def test_filler_rule(tok, nxt, expected):
    assert sb.is_nonlexical_filler(tok, nxt) is expected


def test_score_turn_flags():
    ref = {"rep_literal": 1, "revision": 0, "rep_marks": [REP], "rev_marks": []}
    assert sb.score_turn(ref, 1, "eh quiero quiero un billete") == {
        "rep": 1,
        "rev": 0,
        "fill": 1,
    }
    assert sb.score_turn(ref, 1, "quiero un billete") == {"rep": 0, "rev": 0, "fill": 0}
    # a filler in the ASR text does not count when the turn has none in the reference
    assert sb.score_turn(ref, 0, "eh quiero un billete")["fill"] == 0


def test_cw_tags():
    raw = "[UH] quiero [UM] quiero [laughter] un billete"
    assert sb.cw_score_text(raw) == "quiero quiero un billete"
    assert sb.cw_filled_tags(raw) == 2


# ------------------------------------------------------------------ statistics
def test_boot_ratio_point_estimate_and_degenerate_group():
    r = sb.boot_ratio(["a", "a", "b", "c"], [1, 0, 1, 1], [0] * 4, [1] * 4)
    assert r["delta"] == pytest.approx(0.75) and r["n_groups"] == 3
    assert sb.boot_ratio(["a", "a"], [1, 0], [0, 0], [1, 1])["ci"] == [None, None]


def test_bootstrap_is_deterministic():
    args = (["a", "b", "c", "d"] * 5, [1, 0] * 10, [0] * 20, [1] * 20)
    assert sb.boot_ratio(*args) == sb.boot_ratio(*args)


@pytest.mark.parametrize(
    "ci,reading",
    [
        ((0.1, 0.4), "majority erased (CI upper < 0.50)"),
        ((0.4, 0.7), "erasure persists (CI upper < 0.75)"),
        ((0.8, 0.9), "substantially attenuated (CI lower >= 0.75)"),
        ((0.6, 0.9), "inconclusive"),
    ],
)
def test_tier(ci, reading):
    assert sb.tier(ci) == reading


def test_score_system_reads_words_files(tmp_path):
    frame = [
        {
            "turn_id": "T1",
            "speaker": "S1",
            "rep_literal": 1,
            "revision": 0,
            "filler_present": 0,
        },
        {
            "turn_id": "T2",
            "speaker": "S2",
            "rep_literal": 1,
            "revision": 0,
            "filler_present": 0,
        },
    ]
    refs = {
        t: {"rep_literal": 1, "revision": 0, "rep_marks": [REP], "rev_marks": []}
        for t in ("T1", "T2")
    }
    for t, words in (("T1", "quiero quiero hoy"), ("T2", "quiero hoy")):
        (tmp_path / f"{t}.json").write_text(
            json.dumps({"words": [{"text": w} for w in words.split()]}), "utf8"
        )
    flags = sb.score_system(refs, frame, tmp_path)
    assert [flags[t]["rep"] for t in ("T1", "T2")] == [1, 0]
    assert sb.summarize(frame, flags)["survival_rep"]["value"] == pytest.approx(0.5)
    with pytest.raises(FileNotFoundError):
        sb.score_system(refs, frame + [{**frame[0], "turn_id": "T3"}], tmp_path)
