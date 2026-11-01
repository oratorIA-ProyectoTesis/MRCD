"""Extended pYIN margin study: (left x right) margins x stream partitions x all local audio.

Criterion (unchanged from margin_study.py): committed frames (all 5 columns) that differ from the
whole-file extraction. Additionally, after ``finalize()`` EVERY frame must equal the offline frame.

Signals: the 98 AMERESCO clips, the 40 s YouTube clip (results/paridad_20261030/extra_audio), a
4.5 min concatenation of 8 clips and a long real+synthetic session (``synth_session.build_session``).
(There is no other local audio: ``data/`` holds no audio files.)

Partitions of the stream (the order of audio is always the same; what changes is HOW it arrives):
  tick     fixed 0.5 s ticks (the live demo)
  rand     random tick length, log-uniform in [0.02, 3] s, one append + commit per tick
  tiny     tiny random chunks (1..511 samples, not multiples of 160/512) appended with commit=False;
           ``update()`` is called on a random schedule in [0.05, 1.5] s (device callbacks vs. tick)
  oneshot  everything in one append (+ finalize): the "whole-file" arrival

usage: python margin_study_v2.py --tag X --pairs 3:3,4:4 --parts tick,rand,tiny --workers 8 [--long-min 12]
"""

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import _pin_omp  # noqa: E402,F401

REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

SR = 16_000


def load_signal(name):
    from synth_session import build_session

    man = json.loads(
        (REPO / "results/baselines_20261030/ameresco/manifest.json").read_text("utf8")
    )
    wavs = {m["unit_id"]: m["wav"] for m in man}
    if name in wavs:
        return sf.read(wavs[name], dtype="float32")[0]
    if name == "yt40":
        return sf.read(
            str(REPO / "results/paridad_20261030/extra_audio/yD9Igp27VNE_30s_70s.wav"),
            dtype="float32",
        )[0]
    if name == "concat8":
        rng = np.random.default_rng(0)
        pick = rng.choice(len(man), 8, replace=False)
        return np.concatenate(
            [sf.read(man[i]["wav"], dtype="float32")[0] for i in pick]
        )
    if name.startswith("long"):
        minutes, seed = name[4:].split("_")
        return build_session(float(minutes), int(seed))[0]
    raise KeyError(name)


def feed(tr, audio, part, seed):
    rng = np.random.default_rng(seed)
    n, i = len(audio), 0
    if part == "tick":
        while i < n:
            tr.append(audio[i : i + 8000])
            i += 8000
    elif part == "rand":
        while i < n:
            k = max(int(np.exp(rng.uniform(np.log(0.02), np.log(3.0))) * SR), 1)
            tr.append(audio[i : i + k])
            i += k
    elif part == "tiny":
        next_update = 0
        while i < n:
            k = int(rng.integers(1, 512))
            tr.append(audio[i : i + k], commit=False)
            i += k
            if i >= next_update:
                tr.update()
                next_update = i + int(rng.uniform(0.05, 1.5) * SR)
    elif part == "mix":  # the partition of tests/test_live_stream_partitions.py::test_long_session_*
        while i < n:
            k = int(rng.choice([4000, 8000, 8000, 12345, 16000]))
            tr.append(audio[i : i + k])
            i += k
    elif part == "oneshot":
        tr.append(audio)
    else:
        raise KeyError(part)


_OFFLINE = {}


def offline(name, audio):
    from mrcd.extractors.acoustic import extract_acoustic

    if name not in _OFFLINE:
        _OFFLINE.clear()
        _OFFLINE[name] = extract_acoustic(audio, use_silero=True).frame_matrix()
    return _OFFLINE[name]


def job(args):
    name, L, R, part, seed = args
    from mrcd.live.session_features import SessionFeatureTracker

    import warnings

    warnings.simplefilter("ignore")
    audio = load_signal(name)
    off = offline(name, audio)
    tr = SessionFeatureTracker(f0_center=0.0, left_margin_s=L, right_margin_s=R)
    t0 = time.time()
    feed(tr, audio, part, seed)
    F = tr.frame_matrix()
    d = np.abs(F - off[: len(F)]).max(1) > 0 if len(F) else np.zeros(0, bool)
    bad_t = (np.nonzero(d)[0] * 0.01).round(2).tolist()[:6]
    maxf0 = float(np.abs(F[:, 1] - off[: len(F), 1]).max()) if len(F) else 0.0
    committed = len(F)
    tr.finalize()
    G = tr.frame_matrix()
    fin_bad = int((np.abs(G - off[: len(G)]).max(1) > 0).sum()) if len(G) else 0
    return {
        "signal": name,
        "L": L,
        "R": R,
        "part": part,
        "seed": seed,
        "duration_s": len(audio) / SR,
        "committed": committed,
        "bad_frames": int(d.sum()),
        "max_abs_f0_st": maxf0,
        "first_bad_times_s": bad_t,
        "final_frames": len(G),
        "offline_frames": len(off),
        "final_bad_frames": fin_bad,
        "elapsed_s": time.time() - t0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--parts", default="tick,rand,tiny")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--long-min",
        type=float,
        default=0.0,
        help="add a real+synthetic session of this length",
    )
    ap.add_argument(
        "--long-pairs",
        default="",
        help="pairs to run on the long/concat signals (default: --pairs)",
    )
    ap.add_argument("--signals", default="", help="comma list (e.g. long10.5_3); replaces the default signal set")
    ap.add_argument(
        "--clips",
        type=int,
        default=0,
        help="limit the number of AMERESCO clips (0 = all 98)",
    )
    ap.add_argument(
        "--out", default=str(REPO / "results/ejecucion_material4_20261030/paridad")
    )
    a = ap.parse_args()
    pairs = [tuple(float(x) for x in p.split(":")) for p in a.pairs.split(",")]
    lpairs = (
        [tuple(float(x) for x in p.split(":")) for p in a.long_pairs.split(",")]
        if a.long_pairs
        else pairs
    )
    parts = a.parts.split(",")
    man = json.loads(
        (REPO / "results/baselines_20261030/ameresco/manifest.json").read_text("utf8")
    )
    names = [m["unit_id"] for m in man]
    if a.clips:
        names = names[: a.clips]
    jobs = []
    allnames = a.signals.split(",") if a.signals else names + ["yt40", "concat8"]
    for si, name in enumerate(allnames):
        use = pairs if (a.signals or si < len(names)) else lpairs
        for L, R in use:
            for pi, part in enumerate(parts):
                jobs.append((name, L, R, part, 0 if part == 'mix' else 1000 * si + pi))
    if a.long_min:
        for L, R in lpairs:
            for pi, part in enumerate(parts):
                jobs.append((f"long{a.long_min:g}_1", L, R, part, 77 + pi))
    # group by signal so a worker re-uses the offline extraction
    jobs.sort(key=lambda j: j[0])
    t0 = time.time()
    res = []
    with Pool(a.workers) as pool:
        for k, r in enumerate(pool.imap(job, jobs, chunksize=1)):
            res.append(r)
            if r["bad_frames"] or r["final_bad_frames"] or k % 100 == 0:
                print(
                    k,
                    len(jobs),
                    {
                        x: r[x]
                        for x in (
                            "signal",
                            "L",
                            "R",
                            "part",
                            "bad_frames",
                            "final_bad_frames",
                            "max_abs_f0_st",
                        )
                    },
                    flush=True,
                )
    summ = {}
    for r in res:
        key = f"{r['L']:g}:{r['R']:g}|{r['part']}"
        s = summ.setdefault(
            key,
            {
                "jobs": 0,
                "with_bad": 0,
                "bad_frames": 0,
                "committed": 0,
                "max_abs_f0_st": 0.0,
                "final_bad_frames": 0,
                "worst": None,
            },
        )
        s["jobs"] += 1
        s["with_bad"] += r["bad_frames"] > 0
        s["bad_frames"] += r["bad_frames"]
        s["committed"] += r["committed"]
        s["final_bad_frames"] += r["final_bad_frames"]
        if r["max_abs_f0_st"] >= s["max_abs_f0_st"]:
            s["max_abs_f0_st"] = r["max_abs_f0_st"]
        if s["worst"] is None or r["bad_frames"] > s["worst"]["bad_frames"]:
            s["worst"] = r if r["bad_frames"] else s["worst"]
    out = {
        "args": vars(a),
        "elapsed_s": time.time() - t0,
        "summary": summ,
        "results": res,
    }
    Path(a.out, f"margin_study_v2_{a.tag}.json").write_text(
        json.dumps(out, indent=1), "utf8"
    )
    for k, v in sorted(summ.items()):
        print(
            k,
            {
                x: v[x]
                for x in (
                    "jobs",
                    "with_bad",
                    "bad_frames",
                    "committed",
                    "max_abs_f0_st",
                    "final_bad_frames",
                )
            },
        )
    print("elapsed_s", out["elapsed_s"])


if __name__ == "__main__":
    main()
