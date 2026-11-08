"""Plan C metrics (PROTOCOL_EXPLORATORY_C.md sections 4-5). Writes results/plan_c_20261108/RESULTS_C.json (aggregates only).

usage: python scripts/plan_c/analyze.py        (repo .venv, cwd = repo root, PYTHONPATH = repo root)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "verbatim_ft"))
import dihana_io as dio  # noqa: E402
import vft_metrics as vm  # noqa: E402

OUT = REPO / "results" / "plan_c_20261108"
CACHE = OUT / "cache"
ARMS = ("HF_full", "HF_greedy", "CT2_fp16", "CT2_int8f16")
CT2 = ("CT2_fp16", "CT2_int8f16")
NBOOT, SEED = 5000, 20261108
LABEL = "exploratory (DIHANA folds 1-4)"


def main() -> None:
    units = json.loads((CACHE / "sample.json").read_text("utf8"))["units"]
    dio.assert_folds(u["fold"] for u in units)
    pool = {u["unit_id"]: u for u in dio.load_pool()}
    turns = dio.load_turns()
    ids = [u["unit_id"] for u in units]
    st = {u["unit_id"]: u["stratum"] for u in units}
    dec = {
        a: {i: json.loads((CACHE / a / f"{i}.json").read_text("utf8")) for i in ids}
        for a in ARMS
    }
    refs = {
        i: vm.UnitRef.from_pool(pool[i], dio.n_nonlex_fillers(turns[i])) for i in ids
    }
    per = {
        a: vm.unit_scores(
            [refs[i] for i in ids],
            [vm.strip_tags_cw(dec[a][i]["text"]) for i in ids],
            raw=[dec[a][i]["text"] for i in ids],
        )
        for a in ARMS
    }
    rng_idx = vm.boot_indices(len(ids), NBOOT, SEED)
    res = {
        "label": LABEL,
        "protocol": "results/plan_c_20261108/PROTOCOL_EXPLORATORY_C.md",
        "n_units": len(ids),
        "n_speakers": len({u["speaker"] for u in units}),
        "strata": {k: sum(v == k for v in st.values()) for k in "RFN"},
        "audio_s": round(sum(dec["HF_full"][i]["audio_s"] for i in ids), 2),
        "bootstrap": {
            "B": NBOOT,
            "seed": SEED,
            "cluster": "speaker (one unit per speaker)",
        },
        "arms": {},
        "paired_vs_HF_full": {},
        "exact_match": {},
    }
    for a in ARMS:
        s = per[a]
        sec = np.array([dec[a][i]["sec"] for i in ids])
        aud = np.array([dec[a][i]["audio_s"] for i in ids])
        res["arms"][a] = {
            **vm.summarise(s, rng_idx),
            "rtf_sum": float(sec.sum() / aud.sum()),
            "rtf_median_unit": float(np.median(sec / aud)),
            "decode_s_total": float(sec.sum()),
        }
    for a in ARMS[1:]:
        res["paired_vs_HF_full"][a] = vm.paired(per[a], per["HF_full"], rng_idx)
    norm = lambda t: " ".join(t.split())  # noqa: E731
    for a in CT2:
        res["exact_match"][a] = {}
        for b in ("HF_full", "HF_greedy"):
            raw = sum(norm(dec[a][i]["text"]) == norm(dec[b][i]["text"]) for i in ids)
            tok = sum(
                vm.tokens(vm.strip_tags_cw(dec[a][i]["text"]))
                == vm.tokens(vm.strip_tags_cw(dec[b][i]["text"]))
                for i in ids
            )
            res["exact_match"][a][b] = {
                "raw_identical": int(raw),
                "token_identical": int(tok),
                "n": len(ids),
                "raw_rate": raw / len(ids),
                "token_rate": tok / len(ids),
            }
    hf = res["arms"]["HF_full"]
    acc = {}
    for a in CT2:
        v = res["arms"][a]
        tol_s = max(0.02, 1 / hf["survival_rep"]["n"])
        tol_f = max(0.02, 1 / hf["filler_token_recall"]["n_ref"])
        crit = {
            "A1_survival": v["survival_rep"]["value"]
            >= hf["survival_rep"]["value"] - tol_s,
            "A2_wer": v["wer"]["value"] <= hf["wer"]["value"] + 0.01,
            "A3_filler_recall": v["filler_token_recall"]["value"]
            >= hf["filler_token_recall"]["value"] - tol_f,
            "A4_rtf": v["rtf_sum"] < hf["rtf_sum"],
        }
        acc[a] = {
            **crit,
            "accepted": all(crit.values()),
            "tol_survival": tol_s,
            "tol_filler": tol_f,
            "speedup_vs_HF_full": hf["rtf_sum"] / v["rtf_sum"],
        }
    res["acceptance"] = acc
    meta = {"hf": json.loads((CACHE / "meta_hf.json").read_text("utf8"))}
    for a in CT2:
        meta[a] = json.loads((CACHE / f"meta_{a}.json").read_text("utf8"))
    res["run_meta"] = meta
    # Added after the protocol (descriptive sanity check, not used for acceptance): cached small-model texts of the same
    # units and identity of HF_full with the 2026-11-04 cw2 cache decoded in another environment.
    extra = {}
    for name, tx in (
        ("P0_cached", [dio.p0_text(i) for i in ids]),
        (
            "P1_cached",
            [
                " ".join(
                    w["text"]
                    for w in json.loads((dio.P1_DIR / f"{i}.json").read_text("utf8"))[
                        "words"
                    ]
                )
                for i in ids
            ],
        ),
    ):
        extra[name] = vm.summarise(vm.unit_scores([refs[i] for i in ids], tx), rng_idx)
    old = dio.DATA_REPO / "results/asr_survival_models_20261104/runs/asr/cw_large"
    same = sum(
        norm(json.loads((old / f"{i}.json").read_text("utf8"))["text"])
        == norm(dec["HF_full"][i]["text"])
        for i in ids
    )
    extra["HF_full_vs_20261104_cw_large_cache"] = {
        "identical": int(same),
        "n": len(ids),
    }
    for f in ("cpu_probe.json", "profile_check.json"):  # post-hoc probes (descriptive, written by other scripts)
        if (CACHE / f).exists():
            extra[f.removesuffix(".json")] = json.loads((CACHE / f).read_text("utf8"))
    res["descriptive_not_in_protocol"] = extra
    (OUT / "RESULTS_C.json").write_text(json.dumps(res, indent=1), "utf8")
    print(
        json.dumps(
            {
                a: {
                    k: res["arms"][a][k]["value"]
                    for k in ("wer", "survival_rep", "filler_token_recall")
                }
                | {"rtf": res["arms"][a]["rtf_sum"]}
                for a in ARMS
            },
            indent=1,
        )
    )
    print(json.dumps(res["exact_match"], indent=1))
    print(json.dumps(acc, indent=1))


if __name__ == "__main__":
    main()
