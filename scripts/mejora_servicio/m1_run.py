"""M1 measurement: service documents of every session for the current profile and the two candidates.

    .venv\\Scripts\\python.exe scripts\\mejora_servicio\\m1_run.py --part 0 --nparts 2

Per session (in this order, all through ``MrcdAnalyzer.analyze_bytes`` on the session WAV, options = service defaults
except ``decision_profile``):
  base   = o2026_11                                   (current service default)
  cand_a = o2026_11b, repetition route ``rule_p1``
  cand_b = o2026_11b, repetition route ``rep_ctx_p1`` with the route refit WITHOUT the session's outer fold
The three runs share one per-session chunk cache (same keys as the incremental sessions): every chunk is decoded once
per ASR profile, by the first run that needs it, with the unchanged offline decode call. Resumable (one JSON per session).
Only counts/events/warning codes are written (no text).
"""

import common as K  # noqa: F401  (pins OMP before numpy)

import argparse
import json
import time

import numpy as np

from mrcd import fusion_text as F
from mrcd.service import decision_profile as DP
from mrcd.service import s3_features as S3F
from mrcd.service.analyzer import AnalysisOptions, MrcdAnalyzer
from mrcd.service.contract import validate_document
from mrcd.service.incremental.cache import ChunkCache, segment_key
from mrcd.service.incremental.cached_asr import CachedASR, decode_with
from mrcd.service.incremental.session import session_namespace

import freeze

M1 = K.OUT / "m1"


class StoringASR(CachedASR):
    """Answers from the cache; on a miss decodes with the unchanged offline call AND stores the result."""

    def transcribe(self, audio):  # noqa: D102
        start = int(round(getattr(self, "_chunk_offset_s", 0.0) * K.SR))
        key = segment_key(self._namespace, start, audio)
        hit = self._cache.get(key)
        if hit is None:
            self.decode_calls += 1
            words, events = decode_with(
                self._real, np.asarray(audio, dtype=np.float32), start
            )
            self._cache.put(key, words, events)
        else:
            words, events = hit
        if events:
            self.redecode_events = list(getattr(self, "redecode_events", [])) + events
        return words


def run_session(an: MrcdAnalyzer, sess: dict, routes: dict) -> dict:
    data = K.wav_bytes(K.session_pcm(sess))
    cache = ChunkCache()
    out = {
        "session_id": sess["session_id"],
        "fold": sess["fold"],
        "total_s": sess["total_s"],
        "runs": {},
    }
    plans = [
        ("base", AnalysisOptions(decision_profile=DP.PROFILE_O), None),
        ("cand_a", AnalysisOptions(decision_profile=DP.PROFILE_OB), "rule_p1"),
        ("cand_b", AnalysisOptions(decision_profile=DP.PROFILE_OB), "rep_ctx_p1"),
    ]
    orig_route, orig_load = DP.OB_REPETITION_ROUTE, S3F.load_route
    try:
        for name, opts, route in plans:
            DP.OB_REPETITION_ROUTE = route or orig_route
            S3F.load_route = (
                (lambda r=routes[sess["fold"]]: r)
                if route == "rep_ctx_p1"
                else orig_load
            )
            calls = []

            def wrap(asr, opts=opts):
                w = StoringASR(asr, cache, session_namespace(opts, asr))
                calls.append(w)
                return w

            t0 = time.perf_counter()
            doc = an.analyze_bytes(data, opts, asr_wrapper=wrap)
            dt = time.perf_counter() - t0
            c = K.compact_doc(doc)
            c["valid_errors"] = validate_document(doc)
            c["wall_s"] = round(dt, 3)
            c["decodes"] = sum(w.decode_calls for w in calls)
            c["route"] = route
            out["runs"][name] = c
    finally:
        DP.OB_REPETITION_ROUTE, S3F.load_route = orig_route, orig_load
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--nparts", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    freeze.verify()  # refuses to run if the protocol, the code or the inputs changed after the freeze
    sessions = json.loads((M1 / "sessions.json").read_text("utf8"))
    routes = {
        int(p.stem.split("_")[1]): F.TextRoute.load(p)
        for p in sorted((M1 / "routes").glob("fold_*.json"))
    }
    dest = M1 / "runs"
    dest.mkdir(parents=True, exist_ok=True)
    an = MrcdAnalyzer(device="cpu")
    an.preload(
        AnalysisOptions(decision_profile=DP.PROFILE_OB)
    )  # loads the p0 and the p1 engines
    todo = [s for i, s in enumerate(sessions) if i % args.nparts == args.part]
    if args.limit:
        todo = todo[: args.limit]
    for i, s in enumerate(todo):
        f = dest / f"{s['session_id']}.json"
        if f.exists():
            continue
        t0 = time.perf_counter()
        res = run_session(an, s, routes)
        f.write_text(json.dumps(res), "utf8")
        print(
            f"[{args.part}] {i + 1}/{len(todo)} {s['session_id']} {s['total_s']:.1f}s "
            f"{time.perf_counter() - t0:.1f}s "
            + " ".join(
                f"{k}:{v['status']}/{len(v['events'])}" for k, v in res["runs"].items()
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
