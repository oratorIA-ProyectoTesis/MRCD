"""Technical sensitivity of block / rhetorical_pause / neutral_pause emission to tail padding, turn duration and
linguistic context, on DIHANA DEV units (folds 1-3). NOT a validity study and NOT an F1: these classes have no reference
(DIHANA 'p' marks a silent pause without duration or function). Words come from the frozen ASR cache and are never
changed; only the acoustic padding (or the linguistic context flag) changes between conditions.

Units: a seeded sample (seed 20261030) stratified by duration (<1.5, 1.5-3, 3-6, >=6 s), folds 1-3 only (fold 5 is not touched,
fold 4 is excluded as well so that nothing here overlaps the tuning fold used elsewhere).
Conditions (all with the V0 checkpoint, default recipe unchanged):
  zero        V0 behaviour (digital zeros to one 3 s window; long turns unpadded)
  noise       pad with noise at the unit's own noise floor (mrcd.windowing, existing option)
  zero+1.5    1.5 s MORE digital zeros appended (any duration)
  noise+1.5   1.5 s MORE noise-floor padding
  noctx       zero padding, linguistic context OFF (context=False)
"Emitted" = argmax of any window of the unit equals the class. Paired comparisons against `zero` use the same units; Clopper-Pearson
95% intervals. Also: share of emitting windows (in `zero`) whose span reaches into the padded tail.

usage: python pause_sensitivity.py [N_PER_BIN=60] ; writes results/ejecucion_material4_20261030/pausas/pause_sensitivity.json
"""

import _pin_env  # noqa: F401  (OMP_NUM_THREADS=4 before numpy)
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import CACHE, CKPT, REPO, manifest, references  # noqa: E402

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from scipy.stats import beta  # noqa: E402

NOREF = ["block", "rhetorical_pause", "neutral_pause"]
BINS = [(0, 1.5), (1.5, 3), (3, 6), (6, 1e9)]
OUT = REPO / "results" / "ejecucion_material4_20261030" / "pausas"


def cp(k, n, a=0.05):
    if n == 0:
        return [None, None]
    lo = 0.0 if k == 0 else float(beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a / 2, k + 1, n - k))
    return [round(lo, 4), round(hi, 4)]


def main():
    n_per_bin = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    from mrcd.dataset import acoustic_windows, center_f0, linguistic_windows
    from mrcd.extractors.acoustic import extract_acoustic
    from mrcd.extractors.linguistic import Word, word_features
    from mrcd.extractors.prosody import prosody_windows, speaker_f0_center
    from mrcd.inference import load_ckpt, predict
    from mrcd.windowing import pad_audio, plan_unit

    net, norm, labels, _ = load_ckpt(CKPT, "cpu")
    refs = references("dihana")
    man = {m["unit_id"]: m for m in manifest("dihana")}
    cand = [u for u in sorted(man) if refs[u]["fold"] in (1, 2, 3) and (CACHE / "dihana" / f"{u}.json").exists()]
    assert all(refs[u]["fold"] != 5 for u in cand)
    dur = {u: sf.info(man[u]["wav"]).duration for u in cand}
    rng = np.random.default_rng(20261030)
    units = []
    for lo, hi in BINS:
        pool = [u for u in cand if lo <= dur[u] < hi]
        units += sorted(rng.choice(pool, min(n_per_bin, len(pool)), replace=False).tolist())
    torch.set_num_threads(2)

    def run(uid, *, pad_mode="zero", extra_pad_s=0.0, context=True):
        audio, _ = sf.read(man[uid]["wav"], dtype="float32")
        words = [Word(**w) for w in json.loads((CACHE / "dihana" / f"{uid}.json").read_text("utf8"))["words"]]
        a, b = 0.0, len(audio) / 16000
        starts, b_eff, n_extra, padded = plan_unit(len(audio), a, b)
        if extra_pad_s > 0:
            n_extra += int(round(extra_pad_s * 16000))
            b_eff += extra_pad_s
            starts = np.arange(a, b_eff - 3 + 1e-9, 0.5)
            if b_eff - 3 - starts[-1] > 1e-6:
                starts = np.append(starts, b_eff - 3)
        fm = extract_acoustic(pad_audio(audio, n_extra, pad_mode, seed_key=uid), use_silero=True).frame_matrix()
        c = speaker_f0_center(fm)
        ling, lpos, lmask = linguistic_windows([w.to_dict() for w in words], word_features(words), starts, context=context)
        t = {"ac": acoustic_windows(center_f0(fm, c), starts), "pros": prosody_windows(fm, starts, c),
             "ling": ling[:, :, :12], "lpos": lpos, "lmask": lmask}
        return predict(net, norm, t, "cpu"), starts, b

    conds = {"zero": {}, "noise": {"pad_mode": "noise"}, "zero+1.5": {"extra_pad_s": 1.5},
             "noise+1.5": {"pad_mode": "noise", "extra_pad_s": 1.5}, "noctx": {"context": False}}
    P, S, B = {}, {}, {}
    for name, kw in conds.items():
        for i, u in enumerate(units):
            P[name, u], S[name, u], B[name, u] = run(u, **kw)
        print("done", name, flush=True)

    def em(name, u, c):
        return int((P[name, u].argmax(1) == labels.index(c)).any())

    res = {"n_units": len(units), "units_per_bin": {f"{lo}-{hi}s": sum(lo <= dur[u] < hi for u in units) for lo, hi in BINS},
           "folds": [1, 2, 3], "conditions": list(conds), "emission_rate": {}, "paired_vs_zero": {}, "padded_tail_attribution": {}}
    groups = {"all": units, **{f"{lo}-{hi}s": [u for u in units if lo <= dur[u] < hi] for lo, hi in BINS}}
    for g, us in groups.items():
        res["emission_rate"][g] = {
            c: {n: {"rate": round(float(np.mean([em(n, u, c) for u in us])), 4),
                    "k": int(sum(em(n, u, c) for u in us)), "n": len(us),
                    "ci95": cp(int(sum(em(n, u, c) for u in us)), len(us))} for n in conds} for c in NOREF}
        res["paired_vs_zero"][g] = {
            c: {n: {"flips": int(sum(em(n, u, c) != em("zero", u, c) for u in us)),
                    "gained": int(sum(em(n, u, c) > em("zero", u, c) for u in us)),
                    "lost": int(sum(em(n, u, c) < em("zero", u, c) for u in us)),
                    "flip_rate_ci95": cp(int(sum(em(n, u, c) != em("zero", u, c) for u in us)), len(us))}
                for n in conds if n != "zero"} for c in NOREF}
    # windows of the V0 run that emit a no-reference class: do they reach into the padded tail?
    for c in NOREF:
        tot = tail = 0
        for u in units:
            am = P["zero", u].argmax(1) == labels.index(c)
            reach = S["zero", u] + 3.0 > dur[u] + 1e-6
            tot += int(am.sum())
            tail += int((am & reach).sum())
        res["padded_tail_attribution"][c] = {"emitting_windows": tot, "of_which_reaching_past_audio_end": tail}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pause_sensitivity.json").write_text(json.dumps(res, indent=1), "utf8")
    print(json.dumps(res["emission_rate"]["all"], indent=1)[:3000])
    for g in groups:
        for c in NOREF:
            print(g, c, {n: v["flips"] for n, v in res["paired_vs_zero"][g][c].items()}, flush=True)


if __name__ == "__main__":
    main()
