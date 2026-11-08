"""Plan B data (PROTOCOL_EXPLORATORY_B.md sections 2-3): split, verbatim targets and int16 audio caches.

Writes, under results/plan_b_20261108/cache/data/ (git-ignored, never committed): split.json (turn ids and speakers per
split), targets_<split>.json (turn id -> verbatim target), audio_<split>.npy (int16, concatenated) and
offsets_<split>.json. Prints only counts. Reads DIHANA folds 1-4 only; no ASR output of any kind is read here.

usage: python scripts/verbatim_ft/prepare_data.py     (repo .venv or _venv_ft, PYTHONPATH = repo root)
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import dihana_io as dio  # noqa: E402
import vtext  # noqa: E402

from mrcd.context_eval.reference import _norm_tokens  # noqa: E402
from mrcd.context_eval.transcript_reference import is_nonlexical_filler  # noqa: E402
from mrcd.corpora.dihana import import_dihana_turns  # noqa: E402

OUT = REPO / "results" / "plan_b_20261108" / "cache" / "data"


def main() -> None:
    turns = dio.load_turns()
    pool = dio.load_pool()
    by_spk = collections.defaultdict(list)
    for u in pool:
        by_spk[u["speaker"]].append(u)
    split = vtext.make_split(
        {s: any(u["ref"]["rep_literal"] == 1 for u in us) for s, us in by_spk.items()}
    )
    paths = {p.stem: p for p in dio.DIHANA.glob("data/*/*/*.xml")}
    dialogues = sorted({r["dialogue"] for r in turns.values()})
    xml = {}
    for dg in dialogues:
        for ex in import_dihana_turns(paths[dg], dio.DIHANA):
            if ex.recording_id in turns:
                xml[ex.recording_id] = ex.original_text
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    manifest = {}
    for name, spk in split.items():
        ids = sorted(u["unit_id"] for s in spk for u in by_spk[s])
        dio.assert_folds(turns[i]["fold"] for i in ids)
        targets, chunks, offsets, pos = {}, [], {}, 0
        n_fill = 0
        for i in ids:
            t = vtext.verbatim_target(xml[i])
            toks = _norm_tokens(t)
            no_fill = [
                w
                for k, w in enumerate(toks)
                if not is_nonlexical_filler(
                    w, toks[k + 1] if k + 1 < len(toks) else None
                )
            ]
            # the target minus the reconstructed fillers must be the corpus input view (text A)
            expect = [
                w
                for k, w in enumerate(_norm_tokens(turns[i]["ref_text_marks_stripped"]))
            ]
            if no_fill != [
                w
                for k, w in enumerate(expect)
                if not is_nonlexical_filler(
                    w, expect[k + 1] if k + 1 < len(expect) else None
                )
            ]:
                raise AssertionError(f"target of {i} disagrees with text A")
            n_fill += len(toks) - len(no_fill)
            targets[i] = t
            a = np.round(dio.load_audio(turns[i]) * 32768.0).astype(np.int16)
            offsets[i] = [pos, len(a)]
            pos += len(a)
            chunks.append(a)
        np.save(OUT / f"audio_{name}.npy", np.concatenate(chunks))
        (OUT / f"targets_{name}.json").write_text(
            json.dumps(targets, ensure_ascii=False), "utf8"
        )
        (OUT / f"offsets_{name}.json").write_text(json.dumps(offsets), "utf8")
        manifest[name] = {"speakers": sorted(spk), "turn_ids": ids}
        summary[name] = {
            "speakers": len(spk),
            "turns": len(ids),
            "audio_min": round(pos / dio.SR / 60, 2),
            "filler_tokens_in_targets": n_fill,
            "empty_targets": sum(not v for v in targets.values()),
        }
    (OUT / "split.json").write_text(json.dumps(manifest), "utf8")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1), "utf8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
