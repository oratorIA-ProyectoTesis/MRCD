"""Controlled experiment (current mrcd/, cached words): why do per-slice features differ from whole-file features?
Only grid starts (t multiple of 0.5 -> slice origin is a multiple of the 10 ms hop; terminal off-grid starts excluded to avoid a sub-hop shift artifact).
Variants for each window (words/ling identical in all):
  W   : whole-file frames, whole-file F0 centre            (= Engine/pilot)
  S   : slice frames [t-5,t+8], slice F0 centre            (= per-chunk path)
  S_wc: slice frames, whole-file F0 centre                 (isolates frames/VAD effect)
  W_sc: whole-file frames, slice F0 centre                 (isolates centring effect)
"""
import json, os, sys
os.environ["OMP_NUM_THREADS"] = "4"
from pathlib import Path
import numpy as np, soundfile as sf, torch
torch.set_num_threads(4); torch.manual_seed(0)
sys.path.insert(0, r"C:\zArnes-tesis\mrcd")
from mrcd.inference import load_ckpt, predict
from mrcd.extractors.linguistic import Word, word_features
from mrcd.extractors.acoustic import extract_acoustic
from mrcd.extractors.prosody import speaker_f0_center, prosody_windows
from mrcd.dataset import acoustic_windows, center_f0, linguistic_windows
OUT = Path(r"C:\zArnes-tesis\mrcd\results\paridad_20261030")
net, norm, labels, _ = load_ckpt(Path(r"C:\zArnes-tesis\mrcd\models\ckpt_audio_text.pt"), "cpu"); pi = labels.index("prolongation")
jobs = json.loads((OUT / "jobs.json").read_text("utf8")); sr = 16000
rows = []; colnames = ["log_rms", "f0_st", "voiced_prob", "vad", "df0"]
for j in jobs:
    audio, _ = sf.read(j["wav"], dtype="float32")
    f = j["pilot_words"] or str(OUT / "asr_run1" / f"{j['id']}_asr.json")
    wd = json.loads(Path(f).read_text("utf8")); words = [Word(**w) for w in wd]
    F = extract_acoustic(audio, use_silero=True).frame_matrix(); cW = speaker_f0_center(F)
    st = np.arange(j["a"], j["b"] - 3 + 1e-9, 0.5)
    ling, lpos, lmask = linguistic_windows(wd, word_features(words), st, context=True)
    for i, t in enumerate(st):
        lo = int(round((t - 5.0) * sr)); assert lo % 160 == 0
        Fs = extract_acoustic(audio[lo:lo + 13 * sr], use_silero=True).frame_matrix(); cS = speaker_f0_center(Fs)
        g0 = int(round(t * 100)); s0 = 500
        d = {n: float(np.abs(Fs[s0:s0 + 300, k] - F[g0:g0 + 300, k]).max()) for k, n in enumerate(colnames)}
        vad_frames = int((Fs[s0:s0 + 300, 3] != F[g0:g0 + 300, 3]).sum())
        v = {"W": (F, cW, cW), "S": (Fs, cS, cS), "S_wc": (Fs, cW, cW), "W_sc": (F, cS, cS)}
        pr = {}
        for name, (fr, c, _) in v.items():
            s_ = np.array([t if fr is F else 5.0])
            T = {"ac": acoustic_windows(center_f0(fr, c), s_), "pros": prosody_windows(fr, s_, c),
                 "ling": ling[i:i + 1, :, :12], "lpos": lpos[i:i + 1], "lmask": lmask[i:i + 1]}
            pr[name] = predict(net, norm, T, "cpu")[0]
        base = pr["W"]
        rows.append({"id": j["id"], "kind": j["kind"], "t": float(t), "in_target": bool(t >= j["a"] - 1e-6 and t + 3 <= j["b"] + 1e-6),
                     "center_whole": cW, "center_slice": cS, "frame_col_maxdiff": d, "vad_frames_differing_of_300": vad_frames,
                     **{f"{n}_maxprobdiff": float(np.abs(p - base).max()) for n, p in pr.items() if n != "W"},
                     **{f"{n}_argmax_flip": bool(p.argmax() != base.argmax()) for n, p in pr.items() if n != "W"},
                     "W_prolong_argmax": bool(base.argmax() == pi), **{f"{n}_prolong_argmax": bool(p.argmax() == pi) for n, p in pr.items() if n != "W"}})
    print(j["id"], flush=True)
(OUT / "attribution_slice.json").write_text(json.dumps(rows, indent=1), "utf8")
n = len(rows); g = lambda k: sum(r[k] for r in rows)
summ = {"n_windows": n, "argmax_flips": {k: g(f"{k}_argmax_flip") for k in ("S", "S_wc", "W_sc")},
        "max_prob_diff": {k: max(r[f"{k}_maxprobdiff"] for r in rows) for k in ("S", "S_wc", "W_sc")},
        "center_abs_diff_semitones": {"mean": float(np.mean([abs(r["center_whole"] - r["center_slice"]) for r in rows])), "max": float(max(abs(r["center_whole"] - r["center_slice"]) for r in rows))},
        "frames_max_col_diff_overall": {c: max(r["frame_col_maxdiff"][c] for r in rows) for c in colnames},
        "windows_with_vad_frame_diff": sum(r["vad_frames_differing_of_300"] > 0 for r in rows),
        "max_vad_frames_differing": max(r["vad_frames_differing_of_300"] for r in rows)}
(OUT / "attribution_slice_summary.json").write_text(json.dumps(summ, indent=1), "utf8"); print(json.dumps(summ))
