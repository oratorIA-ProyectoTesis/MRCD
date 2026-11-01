"""Experimental pool: DIHANA user turns of folds 1-4 (fold 5 is historical and never loaded)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from mrcd.context_eval.reference import turn_reference
from mrcd.corpora.dihana import import_dihana_turns

ALLOWED_FOLDS = (1, 2, 3, 4)
HISTORICAL_FOLD = 5


class HistoricalFoldError(ValueError):
    """An attempt was made to use the historical fold (5) inside the experiment."""


def assert_no_historical(folds) -> None:
    if any(int(f) == HISTORICAL_FOLD for f in folds):
        raise HistoricalFoldError(
            "DIHANA fold 5 is historical: not usable for fitting, selection or evaluation"
        )


def asr_text_from_cache(path: Path) -> str:
    d = json.loads(Path(path).read_text("utf8"))
    return " ".join(w["text"] for w in d["words"])


def build_pool(dihana_root: Path, turns_csv: Path, asr_cache_dir: Path) -> list[dict]:
    """One dict per user turn: ids, speaker, dialogue, fold, text_A (marks stripped), text_B (frozen ASR), reference."""
    rows = list(csv.DictReader(Path(turns_csv).open(encoding="utf-8-sig")))
    meta = {r["turn_id"]: r for r in rows if int(r["fold"]) in ALLOWED_FOLDS}
    dialogues = sorted({r["dialogue"] for r in meta.values()})
    fold_of = {r["speaker"]: int(r["fold"]) for r in meta.values()}
    paths = {p.stem: p for p in Path(dihana_root).glob("data/*/*/*.xml")}
    pool: list[dict] = []
    for d in dialogues:
        for ex in import_dihana_turns(paths[d], dihana_root, fold_of):
            r = meta.get(ex.recording_id)
            if r is None:
                continue
            assert ex.input_view == r["ref_text_marks_stripped"], ex.recording_id
            ref = turn_reference(ex.original_text)
            cache = Path(asr_cache_dir) / f"{ex.recording_id}.json"
            pool.append(
                {
                    "unit_id": ex.recording_id,
                    "speaker": ex.group_id,
                    "dialogue": d,
                    "fold": int(r["fold"]),
                    "text_A": ex.input_view,
                    "text_B": asr_text_from_cache(cache),
                    "ref": ref,
                    "model_features": ex.model_features(),
                }
            )
    pool.sort(key=lambda u: u["unit_id"])
    assert_no_historical(u["fold"] for u in pool)
    return pool
