"""Step 0: RTF on 20 turns (P0 settings) + P0 reproducibility vs the frozen cache; then build the O1 subset (protocol 3.2).

Writes results/asr_recovery_20261031/{timing.json,subset.json,PROTOCOL_ANNEX_20261031_1_SUBSET.md}.
"""

import _pin  # noqa: F401  (MUST be first)
import json
import time

import soundfile as sf

from mrcd.asr_recovery.common import ASR, OUT, WAV, WHISPER, load_pool, sha, sha_text
from mrcd.asr_recovery.decode import transcribe_chunked
from mrcd.asr_recovery.subset import build_subset, choose_ratio


def main():
    pool = load_pool()
    elig = [u["unit_id"] for u in pool if u["ref"]["rep_literal"] == 0 and u["ref"]["revision"] == 0]
    elig.sort(key=lambda x: sha_text(f"13|rtf|{x}"))
    probe = elig[:20]
    from mrcd.extractors.linguistic import VerbatimASR

    asr = VerbatimASR(str(WHISPER), device="cpu", compute_type="int8")
    t_all, a_all, same, rows = 0.0, 0.0, 0, []
    for uid in probe:
        audio, sr = sf.read(WAV / f"{uid}.wav", dtype="float32")
        assert sr == 16000 and audio.ndim == 1
        t0 = time.perf_counter()
        words = transcribe_chunked(asr, audio, "P0")
        dt = time.perf_counter() - t0
        cached = json.loads((ASR / f"{uid}.json").read_text("utf8"))["words"]
        eq = [w.text for w in words] == [w["text"] for w in cached]
        same += int(eq)
        t_all += dt
        a_all += len(audio) / 16000
        rows.append({"unit_id": uid, "audio_s": len(audio) / 16000, "sec": dt, "same_text_as_cache": eq})
    rtf = t_all / a_all
    secs = {}
    for r in (1, 2, 3):
        sub = build_subset(pool, r)
        secs[r] = sum(sf.info(WAV / f"{s['unit_id']}.wav").duration for s in sub)
    ratio = choose_ratio(secs, rtf)
    sub = build_subset(pool, ratio)
    n_pos = sum(s["kind"] == "pos" for s in sub)
    est = 1.5 * rtf * secs[ratio] / 60
    timing = {
        "probe_units": probe,
        "rtf": rtf,
        "audio_seconds_probe": a_all,
        "same_text_as_cache": same,
        "budget_min": 90,
        "margin": 1.5,
        "subset_audio_seconds_by_ratio": secs,
        "chosen_ratio": ratio,
        "estimated_min_per_variant": est,
        "p0_control": "frozen cache" if same >= 19 else "re-decode P0 on the subset (control)",
        "rows": rows,
        "omp": __import__("os").environ["OMP_NUM_THREADS"],
    }
    (OUT / "timing.json").write_text(json.dumps(timing, indent=1), "utf8")
    ids = [s["unit_id"] for s in sub]
    lst = {"ratio": ratio, "n_units": len(sub), "n_pos": n_pos, "n_neg": len(sub) - n_pos, "frame": sub}
    (OUT / "subset.json").write_text(json.dumps(lst, indent=1), "utf8")
    h = sha(OUT / "subset.json")
    annex = f"""# Protocol annex 1 (2026-10-31): O1 subset (algorithmic output of protocol 3.2; not a deviation)

* RTF (P0 settings, 20 probe turns, 1 process OMP=4): {rtf:.3f}; audio {a_all:.1f} s; cache reproduced exactly in {same}/20 turns -> P0 control = {timing['p0_control']}.
* Chosen ratio r = {ratio}; estimated time per variant {est:.1f} min (budget 90).
* Units: {len(sub)} ({n_pos} positives of the union, {len(sub) - n_pos} sampled negatives), speakers {len({s['speaker'] for s in sub})}.
* Full id list with weights: `subset.json`, sha256 `{h}`; id-list sha256 `{sha_text(chr(10).join(sorted(ids)))}`.
"""
    (OUT / "PROTOCOL_ANNEX_20261031_1_SUBSET.md").write_text(annex, "utf8")
    print(annex)


if __name__ == "__main__":
    main()
