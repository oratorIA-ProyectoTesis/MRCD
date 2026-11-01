"""Live vs offline parity on the 32 AMERESCO pilot clips (+ one 40 s clip), with cached ASR words.

For every clip the SAME audio and the SAME words go through:
  offline        : whole-file extraction (what DisfluencyEngine does)           -> reference
  old_live       : AudioRingBuffer(13 s) + build_latest_contextual_window every 0.5 s tick (previous live path)
  session_oracle : SessionFeatureTracker with the offline F0 centre injected    (everything but the centre)
  session_causal : SessionFeatureTracker with the causal centre, window scored at the tick it became ready

Reported: max/mean |tensor diff| per key, windows that differ, max |prob diff|, argmax changes, and (pilot
clips) the clip-level 'prolongation anywhere in the contained windows' decision.

usage: python run_live_parity.py [--out DIR] [--limit N]
No labels are used except the gold column of references.csv for the pilot confusion matrix.
"""

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import _pin_omp  # noqa: E402,F401  (OMP_NUM_THREADS=4 before numpy)

REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
import csv  # noqa: E402

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from mrcd.constants import SAMPLE_RATE, WINDOW_S  # noqa: E402
from mrcd.dataset import acoustic_windows, center_f0, linguistic_windows  # noqa: E402
from mrcd.extractors.acoustic import extract_acoustic  # noqa: E402
from mrcd.extractors.linguistic import Word, word_features  # noqa: E402
from mrcd.extractors.prosody import prosody_windows, speaker_f0_center  # noqa: E402
from mrcd.inference import load_ckpt, predict  # noqa: E402
from mrcd.live.mic_stream import AudioRingBuffer  # noqa: E402
from mrcd.live.session_features import SessionFeatureTracker  # noqa: E402
from mrcd.live.window_builder import build_latest_contextual_window  # noqa: E402

AMERESCO = REPO / "results" / "baselines_20261030" / "ameresco"
ASR = REPO / "results" / "mejoras_20261030" / "asr_cache" / "ameresco"
EXTRA = (
    REPO / "results" / "paridad_20261030" / "extra_audio" / "yD9Igp27VNE_30s_70s.wav"
)
EXTRA_WORDS = (
    REPO / "results" / "paridad_20261030" / "asr_run1" / "extra_yD9_30_70_asr.json"
)
KEYS = ("ac", "pros", "ling", "lpos", "lmask")
TICK = 8000  # 0.5 s of audio per mic callback batch


def offline_at(frames, c, words, starts, ling_dim=12):
    ling, lpos, lmask = linguistic_windows(
        [w.to_dict() for w in words],
        word_features(words),
        np.asarray(starts, np.float64),
        context=True,
    )
    return {
        "ac": acoustic_windows(center_f0(frames, c), starts),
        "pros": prosody_windows(frames, starts, c),
        "ling": ling[:, :, :ling_dim],
        "lpos": lpos,
        "lmask": lmask,
    }


def stack(ws):
    return {k: np.concatenate([w[k] for w in ws]) for k in ws[0]}


def diff_stats(live, ref):
    out = {}
    for k in KEYS:
        d = np.abs(live[k].astype(np.float64) - ref[k].astype(np.float64)).reshape(
            len(live[k]), -1
        )
        out[k] = {
            "max": float(d.max()),
            "mean": float(d.mean()),
            "n_differing": int((d.max(1) > 1e-5).sum()),
        }
    return out


def merge_stats(parts):
    """Aggregate per-clip diff_stats: max of max, window-weighted mean, summed counts."""
    agg = {k: {"max": 0.0, "sum": 0.0, "n_differing": 0} for k in KEYS}
    n = 0
    for st, nw in parts:
        n += nw
        for k in KEYS:
            agg[k]["max"] = max(agg[k]["max"], st[k]["max"])
            agg[k]["sum"] += st[k]["mean"] * nw
            agg[k]["n_differing"] += st[k]["n_differing"]
    return {
        k: {
            "max": v["max"],
            "mean": v["sum"] / max(n, 1),
            "n_differing": v["n_differing"],
        }
        for k, v in agg.items()
    }, n


def confusion(y, p):
    tp = sum(a and b for a, b in zip(y, p))
    fp = sum((not a) and b for a, b in zip(y, p))
    fn = sum(a and (not b) for a, b in zip(y, p))
    tn = sum((not a) and (not b) for a, b in zip(y, p))
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "results" / "paridad_live_20261030"))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    net, norm, labels, _ = load_ckpt(REPO / "models" / "ckpt_audio_text.pt", "cpu")
    pi = list(labels).index("prolongation")
    gold = {
        r["unit_id"]: int(r["prolongation_present"])
        for r in csv.DictReader(
            (AMERESCO / "references.csv").open(encoding="utf-8-sig")
        )
    }
    man = [
        m
        for m in json.loads((AMERESCO / "manifest.json").read_text("utf8"))
        if m["in_pilot32"]
    ]
    jobs = [
        {
            "id": m["unit_id"],
            "wav": m["wav"],
            "words": ASR / f"{m['unit_id']}.json",
            "a": m["target_start_s"],
            "b": m["target_end_s"],
            "pilot": True,
        }
        for m in man
    ]
    jobs.append(
        {
            "id": "extra_yD9_30_70",
            "wav": str(EXTRA),
            "words": EXTRA_WORDS,
            "a": None,
            "b": None,
            "pilot": False,
        }
    )
    if a.limit:
        jobs = jobs[: a.limit]
    res = {"per_clip": {}, "paths": {}}
    parts = {p: [] for p in ("old_live", "session_oracle", "session_causal")}
    pr = {
        p: {"argmax_changes": 0, "n": 0, "max_prob_diff": 0.0, "sum_prob_diff": 0.0}
        for p in parts
    }
    dec = {p: [] for p in ("offline", "session_oracle", "session_causal")}
    ys, centre_gap = [], []
    causal_rows = []  # (window start, max |ac diff|, argmax changed, max |prob diff|)
    frames_total = [0, 0]  # committed frames, frames that differ from whole-file extraction
    t_all = time.time()
    for j in jobs:
        t0 = time.time()
        audio, sr = sf.read(j["wav"], dtype="float32")
        assert sr == SAMPLE_RATE and audio.ndim == 1
        raw = json.loads(Path(j["words"]).read_text("utf8"))
        words = [Word(**w) for w in (raw["words"] if isinstance(raw, dict) else raw)]
        dur = len(audio) / sr
        frames = extract_acoustic(audio, use_silero=True).frame_matrix()
        c_off = speaker_f0_center(frames)
        # ---- session tracker replay (one pass; both centre policies evaluated when a window becomes ready)
        tr = SessionFeatureTracker(f0_center="causal")
        seen, got = set(), {"session_oracle": {}, "session_causal": {}}
        for i in range(0, len(audio), TICK):
            tr.append(audio[i : i + TICK])
            s = tr.latest_start()
            if s is None:
                continue
            for st in np.arange(0.0, s + 1e-9, 0.5):
                if round(st / 0.5) in seen:
                    continue
                seen.add(round(st / 0.5))
                got["session_causal"][round(st / 0.5)] = tr.window(
                    float(st), words, ling_dim=12
                )
                tr.f0_center_mode = c_off
                got["session_oracle"][round(st / 0.5)] = tr.window(
                    float(st), words, ling_dim=12
                )
                tr.f0_center_mode = "causal"
        c_last = tr.f0_center_st()
        fr = tr.frame_matrix()
        bad_rows = int((np.abs(fr - frames[: len(fr)]).max(1) > 0).sum())
        frames_total[0] += len(fr)
        frames_total[1] += bad_rows
        centre_gap.append(
            {"clip": j["id"], "c_offline": c_off, "c_causal_final": c_last}
        )
        # ---- old live path: ring buffer + slice extraction at every tick
        ring, old = AudioRingBuffer(), []
        for i in range(0, len(audio), TICK):
            ring.append(audio[i : i + TICK])
            if ring.duration_s >= 13.0:
                buf, b0 = ring.latest(13.0)
                ws = words  # same words as everywhere else (absolute times)
                tens, st, _ = build_latest_contextual_window(buf, b0, ws, ling_dim=12)
                old.append((st, tens))
        row = {
            "dur_s": dur,
            "c_offline": c_off,
            "c_causal_final": c_last,
            "n_session_windows": len(seen), "committed_frames": len(fr), "frame_rows_differing": bad_rows,
            "n_old_windows": len(old),
        }
        paths = {}
        if old:
            starts = [st for st, _ in old]
            paths["old_live"] = (
                stack([t for _, t in old]),
                offline_at(frames, c_off, words, starts),
            )
        for name in ("session_oracle", "session_causal"):
            ks = sorted(got[name])
            starts = np.asarray(ks) * 0.5
            paths[name] = (
                stack([got[name][k] for k in ks]),
                offline_at(frames, c_off, words, starts),
            )
        for name, (live, ref) in paths.items():
            st = diff_stats(live, ref)
            p_l, p_r = (
                predict(net, norm, live, "cpu", bs=1),
                predict(net, norm, ref, "cpu", bs=1),
            )
            n = len(p_l)
            parts[name].append((st, n))
            d = np.abs(p_l - p_r)
            pr[name]["n"] += n
            pr[name]["argmax_changes"] += int((p_l.argmax(1) != p_r.argmax(1)).sum())
            pr[name]["max_prob_diff"] = max(pr[name]["max_prob_diff"], float(d.max()))
            pr[name]["sum_prob_diff"] += float(d.mean()) * n
            if name == "session_causal":
                acd = np.abs(live["ac"].astype(np.float64) - ref["ac"]).reshape(n, -1).max(1)
                flips = p_l.argmax(1) != p_r.argmax(1)
                for st_, d_, f_, pd_ in zip(starts, acd, flips, d.max(1)):
                    causal_rows.append((float(st_), float(d_), bool(f_), float(pd_)))
            row[name] = {
                "n_windows": n,
                "tensors": st,
                "max_prob_diff": float(d.max()),
                "argmax_changes": int((p_l.argmax(1) != p_r.argmax(1)).sum()),
            }
            if j["pilot"] and name != "old_live":
                ks = sorted(got[name])
                # clip decision over the pilot windows contained in [a, b] (grid part; the live path has no terminal window)
                sel = [
                    k
                    for k in ks
                    if j["a"] - 1e-6 <= k * 0.5 and k * 0.5 + WINDOW_S <= j["b"] + 1e-6
                ]
                idx = [ks.index(k) for k in sel]
                pred_live = int(any(p_l[i].argmax() == pi for i in idx))
                pred_ref = int(any(p_r[i].argmax() == pi for i in idx))
                (
                    row[name]["decision_live"],
                    row[name]["decision_offline_same_windows"],
                ) = pred_live, pred_ref
                if name == "session_oracle":
                    dec["offline"].append(pred_ref)
                dec[name].append(pred_live)
        if j["pilot"]:
            ys.append(gold[j["id"]])
        res["per_clip"][j["id"]] = row
        print(
            f"{j['id']}: {time.time() - t0:.1f}s  windows session={len(seen)} old={len(old)}",
            flush=True,
        )
    for name in parts:
        stt, n = merge_stats(parts[name])
        res["paths"][name] = {
            "n_windows": n,
            "tensors": stt,
            "argmax_changes": pr[name]["argmax_changes"],
            "max_prob_diff": pr[name]["max_prob_diff"],
            "mean_prob_diff": pr[name]["sum_prob_diff"] / max(pr[name]["n"], 1),
        }
    res["pilot_grid_confusion"] = {
        k: confusion(ys, v) for k, v in dec.items() if len(v) == len(ys)
    }
    res["pilot_decision_flips_vs_offline"] = {
        k: int(sum(a != b for a, b in zip(dec[k], dec["offline"])))
        for k in ("session_oracle", "session_causal")
    }
    res["centre"] = centre_gap
    bands = {}
    for lo, hi in ((0, 5), (5, 10), (10, 1e9)):
        sel = [r for r in causal_rows if lo <= r[0] < hi]
        bands[f"start_{lo}_{hi if hi < 1e8 else 'end'}"] = {
            "n_windows": len(sel),
            "argmax_changes": sum(r[2] for r in sel),
            "max_ac_diff": max((r[1] for r in sel), default=0.0),
            "median_ac_diff": float(np.median([r[1] for r in sel])) if sel else 0.0,
            "max_prob_diff": max((r[3] for r in sel), default=0.0),
        }
    res["causal_by_window_start_s"] = bands
    res["committed_frames_total"], res["committed_frame_rows_differing"] = frames_total
    res["elapsed_s"] = time.time() - t_all
    (out / "live_parity.json").write_text(json.dumps(res, indent=1), "utf8")
    for name, v in res["paths"].items():
        print(
            name,
            "windows",
            v["n_windows"],
            "argmax_changes",
            v["argmax_changes"],
            "max_prob_diff",
            round(v["max_prob_diff"], 6),
            {
                k: (round(t["max"], 4), t["n_differing"])
                for k, t in v["tensors"].items()
            },
        )
    print(
        "pilot grid confusion",
        res["pilot_grid_confusion"],
        "flips",
        res["pilot_decision_flips_vs_offline"],
    )


if __name__ == "__main__":
    main()
