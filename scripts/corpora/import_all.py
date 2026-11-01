"""Run every importer over the local corpora and write summaries (python -I -m ... not needed).

Usage: python -I scripts/corpora/import_all.py [--out DIR] [--dihana-root DIR]
Large text outputs (data/) are derived from corpora with unresolved licences: do not commit.
"""

from __future__ import annotations
import os

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from mrcd.corpora.base import LeakError  # noqa: E402
from mrcd.corpora.dihana import import_dihana_turns  # noqa: E402
from mrcd.corpora.dime import (  # noqa: E402
    import_dime_dialog,
    import_utt1,
    import_utt74,
    list_dialog_ids,
)
from mrcd.corpora.eslora import eslora_stats, import_eslora_zip  # noqa: E402
from mrcd.corpora.groups import load_sample_parents  # noqa: E402

MATERIAL = REPO / "material" / "MRCD_Material_4_Corpus_20261030"
XCORR = REPO / "results/ejecucion_material4_20261030/audit/xcorr_samples.csv"
OUT = REPO / "results/ejecucion_material4_20261030/importadores"
DIHANA = Path(
    (os.environ.get("MRCD_CORPORA_ROOT", r"<CORPORA_ROOT>") + r"\01_DIHANA\extraido\dihana")
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--dihana-root", type=Path, default=DIHANA)
    a = ap.parse_args()
    data = a.out / "data"
    data.mkdir(parents=True, exist_ok=True)
    parents = load_sample_parents(XCORR)
    dime = MATERIAL / "DIME"
    summary: dict = {"sample_parents": parents}

    ex74 = import_utt74(dime / "muestras", MATERIAL, parents)
    ex1 = import_utt1(dime / "muestras", MATERIAL, parents)
    (a.out / "utt74_example.json").write_text(
        json.dumps(ex74.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8"
    )
    summary["utt74"] = {
        "group": ex74.group_id,
        "coverage": ex74.coverage,
        "events": ex74.events_as_dataset_dicts(),
        "manifest_human_annotation": ex74.manifest_human_annotation(),
    }
    summary["utt1"] = {"group": ex1.group_id, "coverage": ex1.coverage}

    dialogs = [import_dime_dialog(i, dime, MATERIAL) for i in list_dialog_ids(dime)]
    with (data / "dime_dialogs.jsonl").open("w", encoding="utf-8") as f:
        for e in dialogs:
            f.write(
                json.dumps(e.to_dict(include_annotations=False), ensure_ascii=False)
                + "\n"
            )
    for e in dialogs:
        e.model_features()  # raises LeakError on any leak
    summary["dime_dialogs"] = {
        "n": len(dialogs),
        "all_unknown": all(set(e.coverage.values()) == {"unknown"} for e in dialogs),
        "total_audio_min": round(sum(e.audio_duration_s for e in dialogs) / 60, 3),
        "utterances_parsed": sum(e.extra["utterances_parsed"] for e in dialogs),
        "header_vs_parsed_mismatch": [
            e.recording_id
            for e in dialogs
            if str(e.extra["utterances_parsed"])
            != str(e.extra["header_utterances_declared"])
        ],
    }

    es = import_eslora_zip(MATERIAL / "ESLORA/corpus_publico.zip", MATERIAL)
    for e in es:
        e.model_features()
    stats = eslora_stats(es)
    summary["eslora"] = stats
    with (a.out / "eslora_per_recording.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        w = csv.writer(f)
        w.writerow(
            [
                "recording",
                "turns",
                "words",
                "alargamiento",
                "palabra_cortada",
                "pausa",
                "pausa_larga",
                "silencio",
                "vacilacion",
            ]
        )
        for e in es:
            c = {}
            for x in e.original_annotations:
                c[x.source_label.replace("ó", "o")] = (
                    c.get(x.source_label.replace("ó", "o"), 0) + 1
                )
            w.writerow(
                [e.recording_id, e.extra["n_turns"], e.extra["n_words"]]
                + [
                    c.get(k, 0)
                    for k in (
                        "alargamiento",
                        "palabra_cortada",
                        "pausa",
                        "pausa_larga",
                        "silencio",
                        "vacilacion",
                    )
                ]
            )
    with (data / "eslora_input_views.jsonl").open("w", encoding="utf-8") as f:
        for e in es:
            f.write(
                json.dumps(e.to_dict(include_annotations=False), ensure_ascii=False)
                + "\n"
            )

    if a.dihana_root.is_dir():
        fold_of = {}
        for g in sorted((a.dihana_root / "guides").glob("*.gui")):
            for s in g.read_text().split():
                fold_of[s.strip()] = int(g.stem[-1])
        n = leaks = 0
        for x in sorted((a.dihana_root / "data").glob("*/*/*.xml")):
            for e in import_dihana_turns(x, a.dihana_root, fold_of):
                n += 1
                try:
                    e.model_features()
                except LeakError:
                    leaks += 1
        summary["dihana"] = {"user_turns": n, "leaks": leaks}
    (a.out / "import_summary.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k != "sample_parents"},
            ensure_ascii=False,
        )[:3000]
    )


if __name__ == "__main__":
    main()
