"""ASR-survival benchmark: score ASR transcripts of the 1,025-turn DIHANA frame.

Survival = does the human-annotated disfluency evidence of a turn still show up in the ASR text?
The matching rules are NOT re-implemented here: they live in ``mrcd.context_eval.survival``
(``turn_survives``), the same code used for the paper and for ``results/metrics.json``.

What this module adds on top of it:
* the frame (``frame/turn_ids.csv``) and a check of its labels against the DIHANA XML,
* the filler rule (non-lexical filled pause present in the ASR text),
* the speaker-clustered bootstrap (ratio of sums, seed 20261101, 5000 resamples) and the frozen reading tiers.

No corpus text is read from, or written to, this repository: the DIHANA XML is read from a local copy.

usage:  python survival_bench.py --dihana-root <DIHANA> --asr-dir <dir with turn_id.json files>
            [--kind words|cw] [--out survival.json]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

from mrcd.asr_recovery.common import LITERAL_SUBTYPES
from mrcd.context_eval.reference import _norm_tokens, turn_reference
from mrcd.context_eval.survival import turn_survives
from mrcd.corpora.dihana import import_dihana_turns

HERE = Path(__file__).resolve().parent
FRAME_CSV = HERE.parent / "frame" / "turn_ids.csv"
FRAME_COLUMNS = (
    "turn_id",
    "speaker_id",
    "fold",
    "kind",
    "sampling_weight",
    "rep_literal",
    "revision",
    "filler_present",
)
EXPECTED = {
    "units": 1025,
    "rep_literal_pos": 214,
    "revision_pos": 161,
    "filler_pos": 308,
    "sampled_negatives": 679,
    "speakers": 122,
}
SEED, NBOOT = 20261101, 5000
THRESH_PERSISTS, THRESH_MAJORITY = 0.75, 0.50

# --- non-lexical filled pauses (same rule as the frozen protocol; "este" is NOT a filler) -------------------------
NONLEXICAL_FILLER_RE = re.compile(r"^(e+h*|eh+|ehm+|em+|mm+|hm+|ah+|aa+h*)$")


def is_nonlexical_filler(tok: str, nxt: str | None) -> bool:
    if not NONLEXICAL_FILLER_RE.match(tok):
        return False
    if tok == "e" and nxt is not None and (nxt.startswith("i") or nxt.startswith("hi")):
        return False  # conjunction "e" (e indigenas)
    return True


def n_fillers(text: str) -> int:
    t = _norm_tokens(text)
    return sum(
        is_nonlexical_filler(x, t[i + 1] if i + 1 < len(t) else None)
        for i, x in enumerate(t)
    )


# --- CrisperWhisper verbatim tags (PROTOCOL 4.3); used only if you decode with that model yourself -----------------
TAG_RE = re.compile(r"\[([^\[\]]*)\]")
FILLED_PAUSE_TAGS = frozenset(
    {"uh", "um", "eh", "em", "er", "ah", "ahh", "hm", "hmm", "mm", "mmm"}
)


def cw_score_text(raw: str) -> str:
    """Primary scoring text: every bracketed tag removed."""
    return " ".join(TAG_RE.sub(" ", raw).split())


def cw_filled_tags(raw: str) -> int:
    return sum(
        m.group(1).strip().lower() in FILLED_PAUSE_TAGS for m in TAG_RE.finditer(raw)
    )


# --- frame -----------------------------------------------------------------------------------------------------------
def load_frame(path: Path = FRAME_CSV) -> list[dict]:
    with Path(path).open(encoding="utf8", newline="") as f:
        rd = csv.DictReader(f)
        if tuple(rd.fieldnames or ()) != FRAME_COLUMNS:
            raise ValueError(f"unexpected columns {rd.fieldnames}")
        rows = []
        for r in rd:
            rows.append(
                {
                    "turn_id": r["turn_id"],
                    "speaker": r["speaker_id"],
                    "fold": int(r["fold"]),
                    "kind": r["kind"],
                    "w": float(r["sampling_weight"]),
                    "rep_literal": None
                    if r["rep_literal"] == ""
                    else int(r["rep_literal"]),
                    "revision": int(r["revision"]),
                    "filler_present": int(r["filler_present"]),
                }
            )
    return rows


def check_frame(frame: list[dict]) -> dict:
    got = {
        "units": len(frame),
        "rep_literal_pos": sum(r["rep_literal"] == 1 for r in frame),
        "revision_pos": sum(r["revision"] == 1 for r in frame),
        "filler_pos": sum(r["filler_present"] == 1 for r in frame),
        "sampled_negatives": sum(r["kind"] == "neg" for r in frame),
        "speakers": len({r["speaker"] for r in frame}),
    }
    if got != EXPECTED:
        raise AssertionError(f"frame differs from the frozen one: {got}")
    assert {r["fold"] for r in frame} == {1, 2, 3, 4}
    return got


def build_refs(dihana_root: Path, frame: list[dict]) -> dict[str, dict]:
    """Reference marks of the frame turns, rebuilt from the local DIHANA XML (never stored in this repo)."""
    want = {r["turn_id"]: r for r in frame}
    fold_of = {r["speaker"]: r["fold"] for r in frame}
    refs: dict[str, dict] = {}
    for xml in sorted(Path(dihana_root).glob("data/*/*/*.xml")):
        for ex in import_dihana_turns(xml, dihana_root, fold_of):
            r = want.get(ex.recording_id)
            if r is None:
                continue
            ref = turn_reference(ex.original_text)
            if (ref["rep_literal"], ref["revision"]) != (
                r["rep_literal"],
                r["revision"],
            ):
                raise AssertionError(
                    f"{ex.recording_id}: labels differ from frame/turn_ids.csv"
                )
            if ex.group_id != r["speaker"]:
                raise AssertionError(f"{ex.recording_id}: speaker differs")
            refs[ex.recording_id] = ref
    missing = sorted(set(want) - set(refs))
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} frame turns not found in DIHANA, e.g. {missing[:3]}"
        )
    return refs


# --- scoring ---------------------------------------------------------------------------------------------------------
def asr_text(rec: dict, kind: str) -> tuple[str, int]:
    """(text used for matching, extra filler count from verbatim tags)."""
    if kind == "words":
        return " ".join(w["text"] for w in rec["words"]), 0
    if kind == "cw":
        return cw_score_text(rec["text"]), cw_filled_tags(rec["text"])
    raise ValueError(kind)


def score_turn(ref: dict, filler_present: int, text: str, tag_fill: int = 0) -> dict:
    return {
        "rep": int(
            ref["rep_literal"] == 1
            and turn_survives("repetition", ref["rep_marks"], text, LITERAL_SUBTYPES)
        ),
        "rev": int(
            ref["revision"] == 1
            and turn_survives("revision", ref["rev_marks"], text, None)
        ),
        "fill": int(filler_present == 1 and n_fillers(text) + tag_fill > 0),
    }


def score_system(
    refs: dict[str, dict], frame: list[dict], asr_dir: Path, kind: str = "words"
) -> dict[str, dict]:
    out = {}
    for r in frame:
        p = Path(asr_dir) / f"{r['turn_id']}.json"
        if not p.exists():
            raise FileNotFoundError(f"{p.name}: turn not decoded")
        text, tag_fill = asr_text(json.loads(p.read_text("utf8")), kind)
        out[r["turn_id"]] = score_turn(
            refs[r["turn_id"]], r["filler_present"], text, tag_fill
        )
    return out


# --- statistics (exact copy of the bootstrap used by the frozen analysis, scripts/s2_noloop/analyze.py) ---------------
def boot_ratio(g, n1, n0, d, seed=SEED, nboot=NBOOT):
    """Paired difference of ratio-of-sums (n1 - n0) / d with a group bootstrap; two-sided bootstrap p."""
    g, n1, n0, d = (np.asarray(x) for x in (g, n1, n0, d))
    _, inv = np.unique(g.astype(str), return_inverse=True)
    k = int(inv.max()) + 1
    S1, S0, D = (np.zeros(k) for _ in range(3))
    np.add.at(S1, inv, n1)
    np.add.at(S0, inv, n0)
    np.add.at(D, inv, d)
    delta = float(S1.sum() / D.sum() - S0.sum() / D.sum())
    if k < 2:
        return {"delta": delta, "ci": [None, None], "p": None, "n_groups": k}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, k, (nboot, k))
    den = D[idx].sum(1)
    ok = den > 0
    dd = (S1[idx].sum(1)[ok] - S0[idx].sum(1)[ok]) / den[ok]
    B = len(dd)
    p = min(
        1.0, 2 * min((np.sum(dd <= 0) + 1) / (B + 1), (np.sum(dd >= 0) + 1) / (B + 1))
    )
    return {
        "delta": delta,
        "ci": [float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))],
        "p": float(p),
        "n_groups": k,
    }


def tier(ci) -> str:
    """Frozen reading of a survival interval (PROTOCOL 6.1)."""
    lo, hi = ci
    if hi < THRESH_MAJORITY:
        return "majority erased (CI upper < 0.50)"
    if hi < THRESH_PERSISTS:
        return "erasure persists (CI upper < 0.75)"
    if lo >= THRESH_PERSISTS:
        return "substantially attenuated (CI lower >= 0.75)"
    return "inconclusive"


def summarize(frame: list[dict], flags: dict[str, dict]) -> dict:
    pops = {
        "survival_rep": ("rep", [r for r in frame if r["rep_literal"] == 1]),
        "survival_rev": ("rev", [r for r in frame if r["revision"] == 1]),
        "survival_filler": ("fill", [r for r in frame if r["filler_present"] == 1]),
    }
    out = {}
    for name, (key, rows) in pops.items():
        if (
            not rows
        ):  # absence is not zero: an empty population is reported as unavailable
            out[name] = {
                "k": None,
                "n": 0.0,
                "value": None,
                "ci": [None, None],
                "reading": "unavailable (empty population)",
            }
            continue
        g = [r["speaker"] for r in rows]
        n = np.array([flags[r["turn_id"]][key] for r in rows], float)
        r_ = boot_ratio(g, n, np.zeros(len(n)), np.ones(len(n)))
        out[name] = {
            "k": float(n.sum()),
            "n": float(len(rows)),
            "n_groups": r_["n_groups"],
            "value": r_["delta"],
            "ci": r_["ci"],
            "reading": tier(r_["ci"]),
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--dihana-root",
        required=True,
        type=Path,
        help="extracted DIHANA record (contains data/*/*/*.xml)",
    )
    ap.add_argument(
        "--asr-dir",
        required=True,
        type=Path,
        help="directory with one <turn_id>.json per frame turn",
    )
    ap.add_argument("--kind", default="words", choices=("words", "cw"))
    ap.add_argument("--frame", default=FRAME_CSV, type=Path)
    ap.add_argument("--out", type=Path, help="write the summary JSON here")
    a = ap.parse_args()
    frame = load_frame(a.frame)
    check_frame(frame)
    refs = build_refs(a.dihana_root, frame)
    res = summarize(frame, score_system(refs, frame, a.asr_dir, a.kind))
    txt = json.dumps(res, indent=1)
    print(txt)
    if a.out:
        a.out.write_text(txt, "utf8")


if __name__ == "__main__":
    main()
