"""Turn pool for the ambiguity study: DIHANA folds 1-4 only (fold 5 is never opened)."""

from __future__ import annotations

import csv
import xml.etree.ElementTree as ET
from pathlib import Path

from mrcd.context_ambiguity.occurrences import (
    f_marks_between,
    find_occurrences,
    tokens_of_turn,
)
from mrcd.context_eval.pool import ALLOWED_FOLDS, assert_no_historical


def load_turns(dihana_root: Path, turns_csv: Path) -> list[dict]:
    """One dict per user turn of folds 1-4: ids, speaker, fold, tokens, occurrences."""
    rows = list(csv.DictReader(Path(turns_csv).open(encoding="utf-8-sig")))
    meta = {r["turn_id"]: r for r in rows if int(r["fold"]) in ALLOWED_FOLDS}
    dialogues = sorted({r["dialogue"] for r in meta.values()})
    paths = {p.stem: p for p in Path(dihana_root).glob("data/*/*/*.xml")}
    out: list[dict] = []
    for d in dialogues:
        rt = ET.parse(paths[d]).getroot()
        for tn in rt.iter("turno"):
            if tn.get("hablante") != "usuario":
                continue
            tid = f"{paths[d].stem}_{tn.get('id')}"
            r = meta.get(tid)
            t = tn.find("t")
            if r is None or t is None:
                continue
            toks = tokens_of_turn(t)
            occs = find_occurrences(toks, f_marks_between(t))
            out.append(
                {
                    "turn_id": tid,
                    "speaker": r["speaker"],
                    "dialogue": d,
                    "fold": int(r["fold"]),
                    "tokens": toks,
                    "occs": occs,
                    "n_f": int(r["n_f"]),
                }
            )
    out.sort(key=lambda u: u["turn_id"])
    assert_no_historical(u["fold"] for u in out)
    return out
