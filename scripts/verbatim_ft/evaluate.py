"""Plan B val metrics (PROTOCOL_EXPLORATORY_B.md section 5). Writes results/plan_b_20261108/RESULTS_B.json (aggregates).

Arms: P0 (frozen cache), P1 (O1-frame cache + decode_val.py P1), and every fine-tuned arm decoded by decode_val.py
(names given on the command line). Same 983 val turns for every arm; speaker-cluster bootstrap.

usage: python scripts/verbatim_ft/evaluate.py FT_smoke [FT_best]     (repo .venv, PYTHONPATH = repo root)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import dihana_io as dio  # noqa: E402
import vft_metrics as vm  # noqa: E402

from mrcd.asr_profiles import detect_asr_anomalies  # noqa: E402
from mrcd.asr_recovery.decode import words_text  # noqa: E402

OUT = REPO / "results" / "plan_b_20261108"
DATA = OUT / "cache" / "data"
ASR = OUT / "cache" / "asr"
NBOOT, SEED = 5000, 20261108
LABEL = "exploratory (DIHANA folds 1-4)"


def load_arm(arm: str, ids: list[str]) -> dict[str, dict]:
    out = {}
    for i in ids:
        if arm == "P0":
            p = dio.P0_CACHE / f"{i}.json"
        elif arm == "P1":
            p = dio.P1_DIR / f"{i}.json"
            p = p if p.exists() else ASR / "P1" / f"{i}.json"
        else:
            p = ASR / arm / f"{i}.json"
        out[i] = json.loads(p.read_text("utf8"))
    return out


def main(ft_arms: list[str]) -> None:
    split = json.loads((DATA / "split.json").read_text("utf8"))
    ids = split["val"]["turn_ids"]
    targets = json.loads((DATA / "targets_val.json").read_text("utf8"))
    pool = {u["unit_id"]: u for u in dio.load_pool()}
    turns = dio.load_turns()
    dio.assert_folds(turns[i]["fold"] for i in ids)
    refs = [vm.UnitRef.from_pool(pool[i], dio.n_nonlex_fillers(turns[i])) for i in ids]
    groups = [r.speaker for r in refs]
    idx = vm.boot_indices(groups, NBOOT, SEED)
    arms = ["P0", "P1", *ft_arms]
    dec = {a: load_arm(a, ids) for a in arms}
    texts = {a: [words_text(dec[a][i]["words"]) for i in ids] for a in arms}
    per = {
        a: vm.unit_scores(refs, texts[a], verbatim=[targets[i] for i in ids])
        for a in arms
    }
    p1_new = [
        i
        for i in ids
        if (ASR / "P1" / f"{i}.json").exists()
        and not (dio.P1_DIR / f"{i}.json").exists()
    ]
    res = {
        "label": LABEL,
        "protocol": "results/plan_b_20261108/PROTOCOL_EXPLORATORY_B.md",
        "n_units": len(ids),
        "n_speakers": len(set(groups)),
        "bootstrap": {"B": NBOOT, "seed": SEED, "cluster": "speaker"},
        "p1_sources": {
            "o1_frame_cache": len(ids) - len(p1_new),
            "decoded_here": len(p1_new),
        },
        "split_summary": json.loads((DATA / "summary.json").read_text("utf8")),
        "arms": {},
        "paired": {},
    }
    for a in arms:
        r = vm.summarise(per[a], idx)
        an = [detect_asr_anomalies(t, "") for t in texts[a]]
        r["loop_turns"] = int(
            sum(any(x["kind"] == "repeated_ngram_loop" for x in z) for z in an)
        )
        r["empty_outputs"] = int(sum(not t.strip() for t in texts[a]))
        if a != "P0":
            sub = ids if a != "P1" else p1_new
            sec = np.array([dec[a][i]["sec"] for i in sub])
            aud = np.array([dio.duration_s(i) for i in sub])
            r["rtf_cpu_int8_4thr"] = {
                "value": float(sec.sum() / aud.sum()),
                "n_units": len(sub),
                "omp": sorted({dec[a][i].get("omp_num_threads") for i in sub} - {None}),
            }
            if a != "P1" and p1_new:
                s2 = np.array([dec[a][i]["sec"] for i in p1_new])
                a2 = np.array([dio.duration_s(i) for i in p1_new])
                r["rtf_cpu_int8_4thr_on_p1_units"] = float(s2.sum() / a2.sum())
        else:
            r["rtf_cpu_int8_4thr"] = {
                "value": None,
                "not_available": "frozen cache, decoded in an earlier campaign",
            }
        res["arms"][a] = r
    for a in ft_arms:
        res["paired"][f"{a}-P0"] = vm.paired(per[a], per["P0"], idx)
        res["paired"][f"{a}-P1"] = vm.paired(per[a], per["P1"], idx)
    res["paired"]["P1-P0"] = vm.paired(per["P1"], per["P0"], idx)
    # training / conversion record of each fine-tuned arm (no text): FT_<run> -> <MODELS>/<run>, <MODELS>/ct2/
    models = Path(os.environ.get("MRCD_FT_MODELS") or Path(__file__).resolve().parents[2].parent / "_models_ft")
    runs = {
        "FT_smoke": ("smoke", "smoke_int8"),
        "FT_best": ("capped", "capped_best_int8"),
    }
    for a in ft_arms:
        if a not in runs:
            continue
        run, ct2 = runs[a]
        st = json.loads((models / run / "state.json").read_text("utf8"))
        conv = json.loads((models / "ct2" / ct2 / "conversion.json").read_text("utf8"))
        res.setdefault("models", {})[a] = {
            "run": run,
            "steps": st["step"],
            "max_steps": st["max_steps"],
            "stop_reason": st.get("stop_reason"),
            "gpu_hours": st["gpu_s"] / 3600,
            "best": st["best"],
            "dev_loss_history": st["history"],
            "config": st["config"],
            "ct2_int8_model_bin_sha256": conv["model_bin_sha256"],
            "conversion_warnings": conv["warnings"],
        }
    probe = OUT / "cache" / "rtf_probe.json"
    if probe.exists():
        res["rtf_probe_quiet_cpu"] = {
            k: {kk: vv for kk, vv in v.items() if kk != "model"}
            for k, v in json.loads(probe.read_text("utf8")).items()
        }
    (OUT / "RESULTS_B.json").write_text(json.dumps(res, indent=1), "utf8")
    keys = (
        "survival_rep",
        "filler_token_recall",
        "wer",
        "wer_verbatim",
        "false_filler_per100",
    )
    for a in arms:
        print(
            a,
            {k: round(res["arms"][a][k]["value"], 3) for k in keys},
            res["arms"][a].get("rtf_cpu_int8_4thr"),
            "loops",
            res["arms"][a]["loop_turns"],
            "empty",
            res["arms"][a]["empty_outputs"],
        )


if __name__ == "__main__":
    main(sys.argv[1:])
