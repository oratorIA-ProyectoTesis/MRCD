"""Compute tensors + probabilities with ONE code root (snapshot or current), cached ASR words.
Run once per root, in its own process (both roots are named `mrcd`).
usage: run_paths.py --root <dir containing mrcd/> --ckpt <pt> --tag <name> --words pilot|<dir> [--modes batch,bw,engine,slice,batch_bs1]
Modes (starts = pilot grid: a, a+0.5, ..., plus terminal b-3):
  batch      whole-clip extraction once, windows jointly (what run_audio_pilot.py does), predict bs=128
  batch_bs1  same tensors, predict bs=1
  bw         build_window() per start on the whole clip (original per-window code path), predict bs=128 per window
  engine     DisfluencyEngine.analyze(clip) with cached-word stub ASR; grid starts 0,0.5,...; tensors captured from predict()
  slice      build_window() on a [t-5,t+8] slice of the clip (F0 centre / VAD / frames computed on the slice only)
"""
import argparse, hashlib, json, os, sys, time
os.environ["OMP_NUM_THREADS"] = "4"; os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
from pathlib import Path
ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True); ap.add_argument("--ckpt", required=True)
ap.add_argument("--tag", required=True); ap.add_argument("--words", required=True)
ap.add_argument("--modes", default="batch,batch_bs1,bw,engine,slice")
ap.add_argument("--only", default="")
a = ap.parse_args()
sys.path.insert(0, a.root)
import numpy as np, soundfile as sf, torch
torch.set_num_threads(4); torch.manual_seed(0); np.random.seed(0)
import mrcd
assert Path(mrcd.__file__).resolve().is_relative_to(Path(a.root).resolve()), mrcd.__file__
from mrcd.inference import load_ckpt, predict
import mrcd.inference as inf_mod
from mrcd.extractors.linguistic import Word, word_features
from mrcd.extractors.acoustic import extract_acoustic
from mrcd.extractors.prosody import speaker_f0_center, prosody_windows
from mrcd.dataset import acoustic_windows, center_f0, linguistic_windows
from mrcd.live.window_builder import build_window
from mrcd.engine import DisfluencyEngine

OUT = Path(r"C:\zArnes-tesis\mrcd\results\paridad_20261030")
jobs = json.loads((OUT / "jobs.json").read_text("utf8"))
if a.only: jobs = [j for j in jobs if j["id"] in a.only.split(",")]
modes = a.modes.split(",")
net, norm, labels, prov = load_ckpt(Path(a.ckpt), "cpu")
dims = {"ling": net.ling.proj[0].in_features}
LD = dims["ling"]

class CachedASR:
    def __init__(s, words): s.words = words
    def transcribe_chunked(s, audio, chunk_s=8.0): return s.words
    def transcribe(s, audio): return s.words

def load_words(j):
    if a.words == "pilot":
        f = j["pilot_words"] or str(OUT / "asr_run1" / f"{j['id']}_asr.json")
    else:
        f = str(Path(a.words) / f"{j['id']}_asr.json")
    return [Word(**w) for w in json.loads(Path(f).read_text("utf8"))], f

def starts_for(j):
    s = np.arange(j["a"], j["b"] - 3 + 1e-9, 0.5)
    if j["b"] - 3 - s[-1] > 1e-6: s = np.append(s, j["b"] - 3)
    return s

store, meta = {}, {"jobs": {}}
for j in jobs:
    t0 = time.time()
    audio, sr = sf.read(j["wav"], dtype="float32"); assert sr == 16000 and audio.ndim == 1
    words, wfile = load_words(j); wd = [w.to_dict() for w in words]
    starts = starts_for(j); jid = j["id"]
    meta["jobs"][jid] = {"words_file": wfile, "n_words": len(words), "n_starts": len(starts), "dur_s": len(audio) / sr}
    store[f"{jid}/starts"] = starts
    def put(mode, tens, probs):
        for k, v in tens.items(): store[f"{jid}/{mode}/{k}"] = np.asarray(v)
        store[f"{jid}/{mode}/probs"] = np.asarray(probs)
    if "batch" in modes or "batch_bs1" in modes:
        frames = extract_acoustic(audio, use_silero=True).frame_matrix(); c = speaker_f0_center(frames)
        ling, lpos, lmask = linguistic_windows(wd, word_features(words), starts, context=True)
        T = {"ac": acoustic_windows(center_f0(frames, c), starts), "pros": prosody_windows(frames, starts, c),
             "ling": ling[:, :, :LD], "lpos": lpos, "lmask": lmask}
        if "batch" in modes: put("batch", T, predict(net, norm, T, "cpu"))
        if "batch_bs1" in modes: store[f"{jid}/batch_bs1/probs"] = predict(net, norm, T, "cpu", bs=1)
    if "bw" in modes:
        per = [build_window(audio, words, float(t), context=True, include_pros=True, ling_dim=LD) for t in starts]
        T = {k: np.concatenate([p[k] for p in per]) for k in per[0]}
        probs = np.concatenate([predict(net, norm, p, "cpu") for p in per])  # bs default, one window per call
        put("bw", T, probs)
    if "slice" in modes:
        per = []
        for t in starts:
            lo = int(round((t - 5.0) * sr)); hi = lo + 13 * sr
            seg = audio[lo:hi]
            per.append(build_window(seg, words, 5.0, word_offset_s=lo / sr, context=True, include_pros=True, ling_dim=LD))
        T = {k: np.concatenate([p[k] for p in per]) for k in per[0]}
        put("slice", T, predict(net, norm, T, "cpu"))
    if "engine" in modes:
        cap = []
        orig = inf_mod.predict
        def spy(net_, norm_, d, dev, *args, **kw):
            cap.append({k: np.array(v) for k, v in d.items()}); return orig(net_, norm_, d, dev, *args, **kw)
        inf_mod.predict = spy
        try:
            eng = DisfluencyEngine(net, norm, list(labels), CachedASR(words), prov); eng._device = "cpu"
            res = eng.analyze(audio, keep_windows=True)
        finally: inf_mod.predict = orig
        ws = res.windows; T = {k: np.concatenate([c[k] for c in cap]) for k in cap[0]}
        assert len(T["ac"]) == len(ws), (len(T["ac"]), len(ws))
        probs = np.array([[w.probabilities[l] for l in labels] for w in ws])
        put("engine", T, probs); store[f"{jid}/engine/starts"] = np.array([w.start for w in ws])
    print(jid, round(time.time() - t0, 1), flush=True)
meta.update({"root": a.root, "ckpt": a.ckpt, "labels": list(labels), "modes": modes, "words": a.words,
             "ling_dim": LD, "torch": torch.__version__, "np": np.__version__})
np.savez_compressed(OUT / "paths" / f"{a.tag}.npz", **store)
(OUT / "paths" / f"{a.tag}.json").write_text(json.dumps(meta, indent=1), "utf8")
