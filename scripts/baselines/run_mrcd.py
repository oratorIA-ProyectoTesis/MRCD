"""Run the CURRENT MRCD checkpoint + existing fixed rule detectors over a manifest of audio units.

Manifest rows: {unit_id, wav, target_start_s, target_end_s, group, ...}. Short units (< 3 s of target) are zero-padded at the END
to 3 s (recorded as `padded`), so a single 3 s window exists. Windows: 0.5 s stride + terminal window (pilot recipe).
Outputs (resumable, one JSON per line): <out>/<name>_shard<i>.jsonl with ASR words, per-window argmax labels, probabilities,
and the presence of fixed rule detectors (auto_labeler: fillers / prolongations / repetitions / revisions) inside the target span.
NO labels are read here. No training, no tuning.
Usage: python run_mrcd.py MANIFEST.json OUTDIR NAME SHARD NSHARDS [THREADS]
"""
import json, os, sys, time
from pathlib import Path
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, soundfile as sf
from bl_common import REPO, CKPT, WHISPER, sha256_file
sys.path.insert(0, str(REPO))


def main():
    man_path, outdir, name, shard, nshard = sys.argv[1], Path(sys.argv[2]), sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
    threads = int(sys.argv[6]) if len(sys.argv) > 6 else 4
    import torch; torch.set_num_threads(threads)
    from mrcd.inference import load_ckpt, predict
    from mrcd.extractors.linguistic import VerbatimASR, word_features
    from mrcd.extractors.acoustic import extract_acoustic
    from mrcd.extractors.prosody import speaker_f0_center, prosody_windows
    from mrcd.dataset import acoustic_windows, center_f0, linguistic_windows
    from mrcd.annotation.auto_labeler import detect_prolongations, detect_fillers, detect_repetitions, detect_revisions
    man = json.loads(Path(man_path).read_text("utf8"))
    man = [m for i, m in enumerate(man) if i % nshard == shard]
    outdir.mkdir(parents=True, exist_ok=True)
    f_out = outdir / f"{name}_shard{shard}.jsonl"
    done = {json.loads(l)["unit_id"] for fp in outdir.glob(f"{name}_shard*.jsonl") for l in fp.read_text("utf8").splitlines() if l.strip()}  # resume across re-sharding
    net, norm, labels, prov = load_ckpt(CKPT, "cpu")
    asr = VerbatimASR(str(WHISPER), device="cpu", compute_type="int8")
    tic0 = time.perf_counter()
    for k, m in enumerate(man):
        if m["unit_id"] in done: continue
        tic = time.perf_counter()
        audio, sr = sf.read(m["wav"], dtype="float32"); assert sr == 16000 and audio.ndim == 1
        a, b = float(m["target_start_s"]), float(m["target_end_s"])
        padded = False
        if b - a < 3 - 1e-8:
            need = int(round((a + 3.0) * 16000)) - len(audio)
            if need > 0: audio = np.concatenate([audio, np.zeros(need, np.float32)])
            b = a + 3.0; padded = True
        starts = np.arange(a, b - 3 + 1e-9, 0.5)
        if b - 3 - starts[-1] > 1e-6: starts = np.append(starts, b - 3)
        words = asr.transcribe_chunked(audio, chunk_s=8.0)
        track = extract_acoustic(audio, use_silero=True); frames = track.frame_matrix(); c = speaker_f0_center(frames)
        wd = [w.to_dict() for w in words]
        ling, lpos, lmask = linguistic_windows(wd, word_features(words), starts, context=True)
        t = {"ac": acoustic_windows(center_f0(frames, c), starts), "pros": prosody_windows(frames, starts, c),
             "ling": ling[:, :, :12], "lpos": lpos, "lmask": lmask}
        probs = predict(net, norm, t, "cpu")
        argmax = [labels[int(i)] for i in probs.argmax(1)]
        fil = detect_fillers(track, words, "es")
        cand = {"filler_word": fil, "prolongation": detect_prolongations(track, words, "es", fil),
                "repetition": detect_repetitions(words), "revision": detect_revisions(words)}
        inside = lambda cs: int(any(min(x.end, b) > max(x.start, a) for x in cs))
        rec = {"unit_id": m["unit_id"], "group": m.get("group"), "padded": padded, "target": [a, b], "n_windows": len(starts),
               "window_starts": starts.tolist(), "window_labels": argmax,
               "max_prob": {l: float(probs[:, labels.index(l)].max()) for l in labels},
               "model_pred": {l: int(l in argmax) for l in labels},
               "rule_pred": {l: inside(c_) for l, c_ in cand.items()},
               "rule_events": {l: [{"s": round(float(x.start), 2), "e": round(float(x.end), 2), "txt": getattr(x, "label", "")} for x in c_][:6] for l, c_ in cand.items()},
               "asr_text": " ".join(w.text for w in words), "asr_words": wd, "vad": track.vad_backend, "sec": time.perf_counter() - tic}
        with f_out.open("a", encoding="utf8") as f: f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if k % 25 == 0: print(f"shard {shard} {k+1}/{len(man)} elapsed {time.perf_counter()-tic0:.0f}s", flush=True)
    print("DONE", shard, flush=True)


if __name__ == "__main__":
    main()
