"""Model features per unit from the FROZEN ASR cache (never runs Whisper).

For each unit: window plan (pilot recipe, mrcd.windowing), optional padding mode, one acoustic extraction,
windows via the CURRENT mrcd/ package, the base checkpoint's window probabilities and the pooled vector
feeding its head. In pad_mode 'zero' (V0) it also stores the fixed rule detector (auto_labeler) outputs.
Usage: python run_features.py CORPUS PAD_MODE SHARD NSHARDS [TORCH_THREADS]
Output: results/mejoras_20261030/features/<corpus>_<pad_mode>_shard<k>.pkl  (resumable)
"""

import _pin_env  # noqa: F401  (OMP_NUM_THREADS=4 before numpy)
import os
import sys
import json
import pickle
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import REPO, CACHE, FEAT, CKPT, manifest  # noqa
from mrcd.reproducibility import pin_asr_threads

pin_asr_threads(4)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
import numpy as np, soundfile as sf, torch  # noqa


def main():
    corpus, pad_mode, shard, nshard = (
        sys.argv[1],
        sys.argv[2],
        int(sys.argv[3]),
        int(sys.argv[4]),
    )
    torch.set_num_threads(int(sys.argv[5]) if len(sys.argv) > 5 else 2)
    from mrcd.inference import load_ckpt
    from mrcd.adaptation import predict_pooled
    from mrcd.windowing import plan_unit, pad_audio, internal_pause_stats
    from mrcd.extractors.linguistic import Word, word_features
    from mrcd.extractors.acoustic import extract_acoustic
    from mrcd.extractors.prosody import speaker_f0_center, prosody_windows
    from mrcd.dataset import acoustic_windows, center_f0, linguistic_windows
    from mrcd.annotation.auto_labeler import (
        detect_prolongations,
        detect_fillers,
        detect_repetitions,
        detect_revisions,
    )

    net, norm, labels, _ = load_ckpt(CKPT, "cpu")
    FEAT.mkdir(parents=True, exist_ok=True)
    f_out = FEAT / f"{corpus}_{pad_mode}_shard{shard}.pkl"
    recs = pickle.loads(f_out.read_bytes()) if f_out.exists() else {}
    man = [m for i, m in enumerate(manifest(corpus)) if i % nshard == shard]
    t0 = time.perf_counter()
    n_new = 0
    for k, m in enumerate(man):
        uid = m["unit_id"]
        if uid in recs:
            continue
        cp = CACHE / corpus / f"{uid}.json"
        if not cp.exists():
            continue
        asr = json.loads(cp.read_text("utf8"))
        audio, sr = sf.read(m["wav"], dtype="float32")
        assert sr == 16000 and audio.ndim == 1 and len(audio) == asr["n_samples"]
        a, b = float(m["target_start_s"]), float(m["target_end_s"])
        starts, b_eff, n_extra, padded = plan_unit(len(audio), a, b)
        audio_p = pad_audio(audio, n_extra, pad_mode, seed_key=uid)
        words = [Word(**w) for w in asr["words"]]
        track = extract_acoustic(audio_p, use_silero=True)
        frames = track.frame_matrix()
        c = speaker_f0_center(frames)
        ling, lpos, lmask = linguistic_windows(
            [w.to_dict() for w in words], word_features(words), starts, context=True
        )
        t = {
            "ac": acoustic_windows(center_f0(frames, c), starts),
            "pros": prosody_windows(frames, starts, c),
            "ling": ling[:, :, :12],
            "lpos": lpos,
            "lmask": lmask,
        }
        probs, pooled = predict_pooled(net, norm, t, "cpu")
        rec = {
            "unit_id": uid,
            "group": m.get("group"),
            "padded": padded,
            "target": [a, b_eff],
            "dur_s": float(b - a),
            "starts": starts.astype(np.float32),
            "probs": probs.astype(np.float32),
            "pooled": pooled.astype(np.float32),
            "pause": internal_pause_stats(
                track.vad, n_valid=int(np.ceil(len(audio) / 160))
            ),
            "vad": track.vad_backend,
        }
        if pad_mode == "zero":
            fil = detect_fillers(track, words, "es")
            cand = {
                "filler_word": fil,
                "prolongation": detect_prolongations(track, words, "es", fil),
                "repetition": detect_repetitions(words),
                "revision": detect_revisions(words),
            }
            rec["rule_pred"] = {
                l: int(any(min(x.end, b_eff) > max(x.start, a) for x in cs))
                for l, cs in cand.items()
            }
        recs[uid] = rec
        n_new += 1
        if n_new % 100 == 0:
            f_out.write_bytes(pickle.dumps(recs))
            print(
                f"{corpus} {pad_mode} shard {shard} {len(recs)}/{len(man)} {time.perf_counter() - t0:.0f}s",
                flush=True,
            )
    f_out.write_bytes(pickle.dumps(recs))
    print("DONE", corpus, pad_mode, shard, len(recs), "of", len(man), flush=True)


if __name__ == "__main__":
    main()

