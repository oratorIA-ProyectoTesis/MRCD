"""CHM150 step 2: frozen metrics (H1 survival, H2 detection, H3 descriptive P1 insertions).

Usage: python -I s2_eval.py   -> OUT/metrics.json, OUT/predictions.csv (ids + labels only, no text), OUT/config.json
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _pin  # noqa: E402,F401  (MUST be first)

import csv  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from freeze import frozen_manifest, verify_frozen  # noqa: E402
from mrcd.asr_recovery.decode import P1_PROMPT, words_text  # noqa: E402
from mrcd.asr_recovery.stats import (  # noqa: E402
    boot_wf1_ci,
    corpus_wer,
    paired_rate_delta,
    paired_wf1_delta,
    prompt_echo,
    rate_ci,
    wconfusion,
)
from mrcd.context_eval.metrics import holm  # noqa: E402
from mrcd.context_eval.reference import _norm_tokens  # noqa: E402
from mrcd.context_eval.survival import turn_survives  # noqa: E402
from mrcd.context_eval.textfeat import rule_rep  # noqa: E402

B_CI, B_TEST = 2000, 5000
ALPHA = 0.05
DIHANA = {
    "survival_P0": 0.491,
    "survival_P1": 0.673,
    "survival_delta": 0.182,
    "survival_delta_ci": [0.132, 0.233],
    "wf1_V0": 0.516,
    "wf1_rule_P1": 0.682,
    "wf1_rep_ctx_p1": 0.699,
    "wf1_rule_P0": 0.577,
}


def loop4(text: str) -> bool:
    """Hallucination-loop marker: the same normalised token 4+ times in a row, or a 1-4-gram repeated 4+ times."""
    t = _norm_tokens(text)
    for n in (1, 2, 3, 4):
        for i in range(len(t) - 4 * n + 1):
            if all(t[i : i + n] == t[i + k * n : i + (k + 1) * n] for k in range(1, 4)):
                return True
    return False


def prf_ci(y, p, g, n_boot=B_CI, seed=_pin.SEED):
    """Speaker-cluster bootstrap 95 % CIs of precision, recall and F1 (unweighted)."""
    y, p = np.asarray(y, int), np.asarray(p, int)
    _, inv = np.unique(np.asarray(g).astype(str), return_inverse=True)
    C = np.zeros((inv.max() + 1, 3))
    np.add.at(C[:, 0], inv, (y == 1) & (p == 1))
    np.add.at(C[:, 1], inv, (y == 0) & (p == 1))
    np.add.at(C[:, 2], inv, (y == 1) & (p == 0))
    rng = np.random.default_rng(seed)
    S = C[rng.integers(0, len(C), (n_boot, len(C)))].sum(1)
    tp, fp, fn = S[:, 0], S[:, 1], S[:, 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        P, R, F = tp / (tp + fp), tp / (tp + fn), 2 * tp / (2 * tp + fp + fn)
    q = lambda a: [float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5))]  # noqa: E731
    return {"precision_ci95": q(P), "recall_ci95": q(R), "f1_ci95": q(F)}


def paired_rate_p(s1, s0, g, n_boot=B_TEST, seed=_pin.SEED):
    """Two-sided speaker-bootstrap p of mean(s1 - s0) (same formula as ``paired_wf1_delta``)."""
    d = np.asarray(s1, float) - np.asarray(s0, float)
    _, inv = np.unique(np.asarray(g).astype(str), return_inverse=True)
    k = inv.max() + 1
    num, den = np.zeros(k), np.zeros(k)
    np.add.at(num, inv, d)
    np.add.at(den, inv, 1.0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, k, (n_boot, k))
    r = num[idx].sum(1) / np.maximum(den[idx].sum(1), 1)
    p = min(
        1.0,
        2
        * min((np.sum(r <= 0) + 1) / (n_boot + 1), (np.sum(r >= 0) + 1) / (n_boot + 1)),
    )
    return {"p_boot": float(p), "n_boot": n_boot, "n_groups": int(k)}


def main():
    verify_frozen()
    t0 = time.time()
    refs = json.loads((_pin.OUT / "reference.json").read_text("utf8"))
    ids = sorted(refs)
    runs = {}
    for u in ids:
        p = _pin.OUT / "runs" / f"{u}.json"
        if not p.exists():
            raise SystemExit(f"missing run for {u}: step 1 incomplete")
        runs[u] = json.loads(p.read_text("utf8"))
    ref_text = {}
    for line in (
        (_pin.CORPUS_ROOT / "data/transcriptions/CHM150.transcription")
        .read_text("utf8")
        .splitlines()
    ):
        if line.strip():
            parts = line.split()
            ref_text[parts[-1]] = " ".join(parts[:-1])
    g = np.array([refs[u]["speaker"] for u in ids])
    y = np.array([refs[u]["rep_literal"] for u in ids])
    tx = {
        v: [words_text(runs[u][f"asr_{v}"]["words"]) for u in ids] for v in ("P0", "P1")
    }
    surv = {
        v: np.array(
            [
                bool(y[j])
                and turn_survives("repetition", refs[u]["rep_marks"], tx[v][j])
                for j, u in enumerate(ids)
            ]
        )
        for v in ("P0", "P1")
    }
    dec = {
        "V0_o2026_11": np.array(
            [runs[u]["v0"]["model_pred"]["repetition"] for u in ids]
        ),
        "rep_ctx_p1": np.array([runs[u]["route_rep_ctx_p1"]["decision"] for u in ids]),
        "rule_P1": np.array([rule_rep(t) for t in tx["P1"]]),
        "rule_P0": np.array([rule_rep(t) for t in tx["P0"]]),
    }
    pos, neg = y == 1, y == 0
    res = {
        "n_utterances": len(ids),
        "n_speakers": int(len(set(g))),
        "n_pos": int(pos.sum()),
        "pos_speakers": int(len(set(g[pos]))),
    }

    # ---------------------------------------------------------------- H1
    h1 = {
        v: rate_ci(surv[v][pos].astype(float), g[pos], B_CI, _pin.SEED)
        for v in ("P0", "P1")
    }
    h1["paired_P1_minus_P0"] = paired_rate_delta(
        surv["P1"][pos], surv["P0"][pos], g[pos], B_CI, _pin.SEED
    )
    h1["paired_P1_minus_P0"].update(
        paired_rate_p(surv["P1"][pos], surv["P0"][pos], g[pos])
    )
    h1["survival_on_P0_destroyed"] = {
        "n": int((pos & ~surv["P0"]).sum()),
        "recovered_by_P1": int((pos & ~surv["P0"] & surv["P1"]).sum()),
    }
    res["H1"] = h1

    # ---------------------------------------------------------------- H2
    h2 = {"systems": {}, "paired": {}}
    for m, d in dec.items():
        met = wconfusion(y, d)
        met.update(prf_ci(y, d, g))
        met["f1_ci95_stats_helper"] = boot_wf1_ci(y, d, g, None, B_CI, _pin.SEED)
        h2["systems"][m] = {k: v for k, v in met.items() if not k.startswith("w_")}
    for name, a, b in (
        ("rep_ctx_p1_minus_V0", "rep_ctx_p1", "V0_o2026_11"),
        ("rule_P1_minus_V0", "rule_P1", "V0_o2026_11"),
        ("rep_ctx_p1_minus_rule_P1", "rep_ctx_p1", "rule_P1"),
    ):
        r = paired_wf1_delta(y, dec[a], dec[b], g, None, B_TEST, _pin.SEED)
        r["changes"] = {
            "new_tp": int(((dec[a] == 1) & pos & (dec[b] == 0)).sum()),
            "lost_tp": int(((dec[a] == 0) & pos & (dec[b] == 1)).sum()),
            "new_fp": int(((dec[a] == 1) & neg & (dec[b] == 0)).sum()),
            "removed_fp": int(((dec[a] == 0) & neg & (dec[b] == 1)).sum()),
        }
        h2["paired"][name] = r
    res["H2"] = h2

    # ---------------------------------------------------------------- Holm family (4 confirmatory tests)
    fam = {
        "H1_survival_P1_minus_P0": (
            h1["paired_P1_minus_P0"]["p_boot"],
            h1["paired_P1_minus_P0"]["delta"],
            h1["paired_P1_minus_P0"]["ci"],
            ">0",
        ),
        "H2a_rep_ctx_p1_minus_V0": (
            h2["paired"]["rep_ctx_p1_minus_V0"]["p_boot"],
            h2["paired"]["rep_ctx_p1_minus_V0"]["delta"],
            h2["paired"]["rep_ctx_p1_minus_V0"]["ci"],
            ">0",
        ),
        "H2b_rule_P1_minus_V0": (
            h2["paired"]["rule_P1_minus_V0"]["p_boot"],
            h2["paired"]["rule_P1_minus_V0"]["delta"],
            h2["paired"]["rule_P1_minus_V0"]["ci"],
            ">0",
        ),
        "H2c_rep_ctx_p1_minus_rule_P1": (
            h2["paired"]["rep_ctx_p1_minus_rule_P1"]["p_boot"],
            h2["paired"]["rep_ctx_p1_minus_rule_P1"]["delta"],
            h2["paired"]["rep_ctx_p1_minus_rule_P1"]["ci"],
            "!=0",
        ),
    }
    adj = holm([v[0] for v in fam.values()])
    res["holm_family"] = {}
    for (k, (p, d, ci, direction)), pa in zip(fam.items(), adj):
        excl = ci[0] > 0 or ci[1] < 0
        ok_dir = d > 0 if direction == ">0" else True
        res["holm_family"][k] = {
            "delta": d,
            "ci95": ci,
            "p_boot": p,
            "p_holm": pa,
            "expected_direction": direction,
            "verdict": "supported"
            if (pa < ALPHA and excl and ok_dir)
            else "not demonstrated",
        }
    res["dihana_reference_values"] = DIHANA
    res["H1"]["dihana_delta_inside_chm150_ci"] = bool(
        h1["paired_P1_minus_P0"]["ci"][0]
        <= DIHANA["survival_delta"]
        <= h1["paired_P1_minus_P0"]["ci"][1]
    )

    # ---------------------------------------------------------------- H3 (descriptive)
    h3 = {"n_negative_utterances": int(neg.sum()), "fp_rate_on_reference_negatives": {}}
    for m, d in dec.items():
        h3["fp_rate_on_reference_negatives"][m] = rate_ci(
            d[neg].astype(float), g[neg], B_CI, _pin.SEED
        )
    intro = (dec["rule_P1"] == 1) & (dec["rule_P0"] == 0)
    remov = (dec["rule_P1"] == 0) & (dec["rule_P0"] == 1)
    h3["P1_introduced_repetition_on_negatives"] = rate_ci(
        intro[neg].astype(float), g[neg], B_CI, _pin.SEED
    )
    h3["P1_removed_repetition_on_negatives"] = rate_ci(
        remov[neg].astype(float), g[neg], B_CI, _pin.SEED
    )
    h3["rule_P1_minus_rule_P0_fp_rate_on_negatives"] = paired_rate_delta(
        dec["rule_P1"][neg], dec["rule_P0"][neg], g[neg], B_CI, _pin.SEED
    )
    for v in ("P0", "P1"):
        h3[f"anomalies_{v}"] = {
            "prompt_echo_4gram": int(sum(prompt_echo(t, P1_PROMPT) for t in tx[v])),
            "loop4": int(sum(loop4(t) for t in tx[v])),
            "loop4_on_negatives": int(
                sum(loop4(tx[v][j]) for j in np.flatnonzero(neg))
            ),
            "empty_output": int(sum(not t.strip() for t in tx[v])),
            "mean_words": float(
                np.mean([len(runs[u][f"asr_{v}"]["words"]) for u in ids])
            ),
            "wer_vs_transcription_all": corpus_wer(
                [(ref_text[u], tx[v][j]) for j, u in enumerate(ids)]
            ),
            "wer_vs_transcription_neg": corpus_wer(
                [(ref_text[ids[j]], tx[v][j]) for j in np.flatnonzero(neg)]
            ),
            "wer_vs_transcription_pos": corpus_wer(
                [(ref_text[ids[j]], tx[v][j]) for j in np.flatnonzero(pos)]
            ),
        }
    res["H3_descriptive"] = h3

    # ---------------------------------------------------------------- run description
    dur = np.array([refs[u]["duration_s"] for u in ids])
    sec = {
        "asr_P0": sum(runs[u]["asr_P0"]["sec"] for u in ids),
        "asr_P1": sum(runs[u]["asr_P1"]["sec"] for u in ids),
        "v0_unit_with_own_asr": sum(runs[u]["v0"]["sec"] for u in ids),
        "route_rep_ctx_p1": sum(runs[u]["route_rep_ctx_p1"]["sec"] for u in ids),
    }
    res["runtime"] = {
        "audio_s": float(dur.sum()),
        "seconds": sec,
        "rtf": {k: float(v / dur.sum()) for k, v in sec.items()},
        "device": "cpu int8 (faster-whisper small), torch cpu, OMP_NUM_THREADS=4 per process",
    }
    from mrcd.service import decision_profile as DP

    res["service_profile_note"] = {
        "o2026_11_pipelined_classes": sorted(DP.O_CLASSES),
        "repetition_in_pipelined_classes": "repetition" in DP.O_CLASSES,
        "utterances_service_would_accept_min_audio_s_13": int((dur >= 13.0).sum()),
        "v0_padded_units": int(sum(runs[u]["v0"]["padded"] for u in ids)),
    }
    same = [
        words_text(runs[u]["v0"]["asr_words"]) == tx["P0"][j]
        for j, u in enumerate(ids)
        if not runs[u]["v0"]["padded"]
    ]
    res["sanity_v0_engine_asr_equals_P0_text_on_unpadded"] = {
        "n": len(same),
        "equal": int(sum(same)),
    }
    res["sensitivity_reference_without_filler_removal_positives"] = int(
        sum(refs[u]["rep_literal_without_filler_removal"] for u in ids)
    )

    rows = []
    for j, u in enumerate(ids):
        r = runs[u]
        rows.append(
            {
                "utt_id": u,
                "speaker": g[j],
                "sex": refs[u]["sex"],
                "duration_s": f"{refs[u]['duration_s']:.3f}",
                "y_rep_literal_ref": int(y[j]),
                "n_ref_marks": refs[u]["n_rep_marks"],
                "n_ref_nonlexical_fillers": refs[u]["n_fillers"],
                "survives_P0": int(surv["P0"][j]),
                "survives_P1": int(surv["P1"][j]),
                "pred_V0_o2026_11": int(dec["V0_o2026_11"][j]),
                "pred_rep_ctx_p1": int(dec["rep_ctx_p1"][j]),
                "pred_rule_P1": int(dec["rule_P1"][j]),
                "pred_rule_P0": int(dec["rule_P0"][j]),
                "v0_padded": int(r["v0"]["padded"]),
                "n_words_P0": len(r["asr_P0"]["words"]),
                "n_words_P1": len(r["asr_P1"]["words"]),
                "loop4_P0": int(loop4(tx["P0"][j])),
                "loop4_P1": int(loop4(tx["P1"][j])),
                "prompt_echo_P1": int(prompt_echo(tx["P1"][j], P1_PROMPT)),
            }
        )
    with (_pin.OUT / "predictions.csv").open("w", newline="", encoding="utf8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    import ctranslate2
    import faster_whisper
    import torch

    res["config"] = {
        "frozen_manifest": frozen_manifest(),
        "seed": _pin.SEED,
        "n_boot_ci": B_CI,
        "n_boot_test": B_TEST,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": {
            "numpy": np.__version__,
            "torch": torch.__version__,
            "faster_whisper": faster_whisper.__version__,
            "ctranslate2": ctranslate2.__version__,
        },
        "seconds": round(time.time() - t0, 1),
    }
    (_pin.OUT / "metrics.json").write_text(
        json.dumps(res, indent=1, default=float), "utf8"
    )
    print(
        json.dumps(
            {k: res[k] for k in ("holm_family", "n_pos", "pos_speakers")},
            indent=1,
            default=float,
        )
    )


if __name__ == "__main__":
    main()
