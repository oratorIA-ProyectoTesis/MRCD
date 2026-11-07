"""External confirmation on CHM150 (run ONLY after the M1 decision is frozen): base vs the adopted profile.

    .venv\\Scripts\\python.exe scripts\\mejora_servicio\\c1_chm150.py run --part 0 --nparts 2
    .venv\\Scripts\\python.exe scripts\\mejora_servicio\\c1_chm150.py eval

Closest path to the service: CHM150 utterances (median 1.85 s, all < 13 s) concatenated per speaker exactly as the
DIHANA sessions of M1 (2.0 s digital silence, <= 120 s, padded to 13 s), each session WAV through
``MrcdAnalyzer.analyze_bytes`` with ``o2026_11`` (base) and the adopted ``o2026_11b`` (its route as frozen in
``decision_profile.OB_REPETITION_ROUTE``), sharing one chunk cache per session. Max-overlap mapping of events to
utterances. Reference: literal repetition derived from the orthographic transcription
(``results/external_chm150_20261102/reference.json``, local only; 76 positives). Unweighted metrics, speaker bootstrap.
Only aggregate numbers are versioned (``c1/metrics_chm150.json``); per-session files stay local (results/ is ignored).
"""

import common as K  # noqa: F401  (pins OMP before numpy)

import argparse
import json
import time

import numpy as np

from mrcd.service import decision_profile as DP
from mrcd.service.analyzer import AnalysisOptions, MrcdAnalyzer
from mrcd.service.contract import validate_document
from mrcd.service.incremental.cache import ChunkCache
from mrcd.service.incremental.session import session_namespace

import freeze
from m1_run import StoringASR

C1 = K.OUT / "c1"
REF = K.REPO / "results" / "external_chm150_20261102" / "reference.json"
ROOT = K.REPO / "data" / "external" / "CHM150" / "CHM150" / "data"


def read_flac(path) -> np.ndarray:
    import soundfile as sf

    x, sr = sf.read(str(path), dtype="int16")
    if sr != K.SR or x.ndim != 1:
        raise ValueError(f"{path.name}: expected 16 kHz mono")
    return x


def plan() -> list[dict]:
    ref = json.loads(REF.read_text("utf8"))
    rows = [{"unit_id": u, "speaker": r["speaker"], "fold": 0} for u, r in ref.items()]
    sess = K.build_sessions(rows, {u: r["duration_s"] for u, r in ref.items()})
    paths = {u: r["audio_relpath"] for u, r in ref.items()}
    for s in sess:
        for t in s["turns"]:
            t["relpath"] = paths[t["unit_id"]]
    return sess


def session_pcm(sess) -> np.ndarray:
    n = int(round(sess["total_s"] * K.SR))
    pcm = np.zeros(n, np.int16)
    for t in sess["turns"]:
        x = read_flac(ROOT / t["relpath"])
        a = int(round(t["offset_s"] * K.SR))
        pcm[a : a + len(x)] = x[: max(0, n - a)]
    return pcm


def run(args):
    freeze.verify()
    sessions = plan()
    dest = C1 / "runs"
    dest.mkdir(parents=True, exist_ok=True)
    an = MrcdAnalyzer(device="cpu")
    an.preload(AnalysisOptions(decision_profile=DP.PROFILE_OB))
    todo = [s for i, s in enumerate(sessions) if i % args.nparts == args.part]
    for i, s in enumerate(todo):
        f = dest / f"{s['session_id']}.json"
        if f.exists():
            continue
        data = K.wav_bytes(session_pcm(s))
        cache = ChunkCache()
        res = {"session_id": s["session_id"], "runs": {}}
        for name, prof in (("base", DP.PROFILE_O), ("adopted", DP.PROFILE_OB)):
            opts = AnalysisOptions(decision_profile=prof)
            t0 = time.perf_counter()
            doc = an.analyze_bytes(
                data,
                opts,
                asr_wrapper=lambda asr, o=opts: StoringASR(
                    asr, cache, session_namespace(o, asr)
                ),
            )
            c = K.compact_doc(doc)
            c["valid_errors"] = validate_document(doc)
            c["wall_s"] = round(time.perf_counter() - t0, 3)
            res["runs"][name] = c
        f.write_text(json.dumps(res), "utf8")
        print(
            f"[{args.part}] {i + 1}/{len(todo)} {s['session_id']} {s['total_s']:.1f}s",
            flush=True,
        )


def evaluate(_args):
    freeze.verify()
    ref = json.loads(REF.read_text("utf8"))
    sessions = plan()
    pred = {"base": {}, "adopted": {}}
    docs = {"base": [], "adopted": []}
    for s in sessions:
        r = json.loads((C1 / "runs" / f"{s['session_id']}.json").read_text("utf8"))
        for k in pred:
            docs[k].append((r["runs"][k]["status"], len(r["runs"][k]["valid_errors"])))
            for u, cl in K.assign_events(s["turns"], r["runs"][k]["events"]).items():
                pred[k][u] = int("repetition" in cl)
    ids = sorted(ref)
    y = np.array([int(ref[u]["rep_literal"]) for u in ids])
    g = np.array([ref[u]["speaker"] for u in ids])
    w = np.ones(len(ids))
    P = {k: np.array([pred[k][u] for u in ids]) for k in pred}
    d0, dist = K.paired_dist(y, P["adopted"], P["base"], g, w)
    neg = y == 0
    out = {
        "route": DP.OB_REPETITION_ROUTE,
        "n_utterances": len(ids),
        "positives": int(y.sum()),
        "speakers": int(len(set(g.tolist()))),
        "sessions": len(sessions),
        "documents": {
            k: {
                "n": len(v),
                "completed": sum(st == "completed" for st, _ in v),
                "invalid": sum(e > 0 for _, e in v),
            }
            for k, v in docs.items()
        },
        "describe": {k: K.describe(y, P[k], g, w) for k in P},
        "adopted_minus_base_f1": {
            "delta": d0,
            "ci95": K.ci(dist),
            "p_boot": K.p_two_sided(dist),
        },
        "fpr_on_negatives_adopted_minus_base": K.paired_rate_diff(
            P["adopted"][neg], P["base"][neg], g[neg], w[neg]
        ),
    }
    for k in ("fpr_on_negatives_adopted_minus_base",):
        out[k] = {a: b for a, b in out[k].items() if not a.startswith("_")}
    (C1 / "metrics_chm150.json").write_text(
        json.dumps(out, indent=1, default=float), "utf8"
    )
    print(
        json.dumps(
            {k: out[k] for k in ("adopted_minus_base_f1", "documents")},
            indent=1,
            default=float,
        )
    )
    for k in P:
        m = out["describe"][k]
        print(
            k,
            m["tp"],
            m["fp"],
            m["fn"],
            round(m["f1"], 3),
            [round(x, 3) for x in m["f1_ci95"]],
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "eval"])
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--nparts", type=int, default=1)
    a = ap.parse_args()
    (run if a.cmd == "run" else evaluate)(a)


if __name__ == "__main__":
    main()
