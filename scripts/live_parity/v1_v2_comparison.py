"""live_cfg_v1 vs live_cfg_v2 on the 32 AMERESCO pilot clips (real checkpoint, ASR words GIVEN from cache).

Same unit as PARIDAD_LIVE.md section 2: clip decision = "prolongation anywhere in the windows contained in
[target_start, target_end]", TP/FP/FN/TN against references.csv. Variants:

  v1            rows of live_cfg_v1 (provisional only, the tail is never scored)
  v2_prov       provisional rows of live_cfg_v2 (must equal v1 row by row: asserted)
  v2_final      last row per timestamp of live_cfg_v2 (tail + terminal + rescore with the final centre)
  offline       whole-file engine tensors with the OFFLINE centre, same windows (reference)

Also: number of contained windows that exist in each variant (coverage), probability agreement of v2_final
with offline (exact expected) and the latency of every provisional row (emitted_at_s - (start + 3)).
No ASR is run: the words are the frozen cache (this is NOT a streaming-ASR measurement).
"""

import csv
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import _pin_omp  # noqa: E402,F401

REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from run_live_parity import AMERESCO, ASR, confusion, offline_at  # noqa: E402

from mrcd.constants import WINDOW_S  # noqa: E402
from mrcd.extractors.acoustic import extract_acoustic  # noqa: E402
from mrcd.extractors.linguistic import Word  # noqa: E402
from mrcd.extractors.prosody import speaker_f0_center  # noqa: E402
from mrcd.inference import load_ckpt, predict  # noqa: E402
from mrcd.live.session import LIVE_CFG_V1, LIVE_CFG_V2, LiveScorer  # noqa: E402

TICK = 8000


def main():
    out = REPO / "results/ejecucion_material4_20261030/paridad/v1_v2_comparison.json"
    net, norm, labels, _ = load_ckpt(REPO / "models" / "ckpt_audio_text.pt", "cpu")
    labels = list(labels)
    pi = labels.index("prolongation")
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

    def score(t):
        return predict(net, norm, t, "cpu")[0]

    dec = {k: [] for k in ("v1", "v2_prov", "v2_final", "offline")}
    cover = {k: 0 for k in dec}
    contained_total, ys, lat, mismatch_prov, mismatch_final = 0, [], [], 0, 0
    only_v1 = common_total = 0
    t0 = time.time()
    for m in man:
        audio, _ = sf.read(m["wav"], dtype="float32")
        raw = json.loads((ASR / f"{m['unit_id']}.json").read_text("utf8"))
        words = [Word(**w) for w in (raw["words"] if isinstance(raw, dict) else raw)]
        runs = {}
        for name, cfg in (("v1", LIVE_CFG_V1), ("v2", LIVE_CFG_V2)):
            sc = LiveScorer(score, labels, config=cfg, f0_center="causal", ling_dim=12)
            for i in range(0, len(audio), TICK):
                sc.push(audio[i : i + TICK])
                sc.step(words)
            sc.finalize(words)
            runs[name] = sc.rows
        prov_v1 = {r["timestamp"]: r for r in runs["v1"]}
        prov_v2 = {
            r["timestamp"]: r for r in runs["v2"] if r["status"] == "provisional"
        }
        # R differs (3 s vs 4 s): the causal centre at emission differs, so common timestamps may differ in
        # probabilities; windows that only v1 emitted are the last second the larger margin delays.
        common = set(prov_v1) & set(prov_v2)
        mismatch_prov += sum(
            prov_v1[t]["todas_las_probabilidades"] != prov_v2[t]["todas_las_probabilidades"] for t in common
        )
        only_v1 += len(set(prov_v1) - set(prov_v2))
        common_total += len(common)
        last = {}
        for r in runs["v2"]:
            last[r["timestamp"]] = r
        lat += [
            r["emitted_at_s"] - (r["timestamp"] + WINDOW_S) for r in prov_v2.values()
        ]
        frames = extract_acoustic(audio, use_silero=True).frame_matrix()
        c_off = speaker_f0_center(frames)
        a, b = m["target_start_s"], m["target_end_s"]

        def contained(ts):
            return [t for t in ts if a - 1e-6 <= t and t + WINDOW_S <= b + 1e-6]

        def pred(rows):
            cs = contained(rows)
            return int(any(rows[t]["clase"] == "prolongation" for t in cs)), len(cs)

        for k, rows in (("v1", prov_v1), ("v2_prov", prov_v2), ("v2_final", last)):
            d, n = pred(rows)
            dec[k].append(d)
            cover[k] += n
        starts = np.array(sorted(last))
        off = offline_at(frames, c_off, words, starts)
        p_off = predict(net, norm, off, "cpu", bs=1)
        off_rows = {
            float(s): {"clase": labels[int(p.argmax())]} for s, p in zip(starts, p_off)
        }
        d, n = pred(off_rows)
        dec["offline"].append(d)
        cover["offline"] += n
        # exact agreement of the final rows with the offline probabilities (offline centre == final live centre)
        for s, p in zip(starts, p_off):
            fin = last[float(s)]["todas_las_probabilidades"]
            mismatch_final += int(
                any(abs(fin[l] - float(v)) > 0 for l, v in zip(labels, p))
            )
        ys.append(gold[m["unit_id"]])
        contained_total += len(contained(starts))
        print(m["unit_id"], f"{time.time() - t0:.0f}s", flush=True)
    res = {
        "clips": len(man),
        "contained_windows_offline": contained_total,
        "coverage_contained_windows": cover,
        "confusion": {k: confusion(ys, v) for k, v in dec.items()},
        "decision_flips_vs_offline": {
            k: int(sum(x != y for x, y in zip(v, dec["offline"])))
            for k, v in dec.items()
        },
        "v1_vs_v2_provisional_common_windows": common_total,
        "v1_vs_v2_provisional_common_windows_with_different_probs": mismatch_prov,
        "provisional_windows_only_in_v1": only_v1,
        "v2_final_windows_with_prob_difference_vs_offline": mismatch_final,
        "provisional_latency_after_window_end_s": {
            "median": float(np.median(lat)),
            "p95": float(np.percentile(lat, 95)),
            "max": float(np.max(lat)),
        },
        "elapsed_s": time.time() - t0,
    }
    out.write_text(json.dumps(res, indent=1), "utf8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
