"""Real-Whisper equivalence of B incremental vs offline (cases 1-27 of DESIGN_B_INCREMENTAL), full HTTP stack in-process.

    $env:OMP_NUM_THREADS = "4"; $env:HF_HUB_OFFLINE = "1"
    .\\.venv\\Scripts\\python.exe -m scripts.s8_incremental.equivalence_real [--quick] [--out DIR]

Process state is fixed as the service does it: device cpu, preload p0 then p1_disfluent_prompt, warm-up. The offline
reference is always computed in the SAME process. Output: only pass/fail, document digests and counters (no audio, no
text). Document comparison is ``==`` on the sorted-key JSON excluding ``generated_at``.
"""

from __future__ import annotations

import argparse
import os
import hashlib
import json
import random
import sys
import time
import wave
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from starlette.testclient import TestClient  # noqa: E402

from mrcd.extractors.linguistic import VerbatimASR  # noqa: E402
from mrcd.service import AnalysisOptions, MrcdAnalyzer  # noqa: E402
from mrcd.service.incremental import plan  # noqa: E402
from mrcd.service.incremental.session import IncrementalConfig, canonical_wav  # noqa: E402
from mrcd.service.server import create_app, prepare  # noqa: E402
from tests.incremental_rig import Rig, TokenClient, split, strip_volatile  # noqa: E402

SR = 16000
C60 = REPO / "results/cierre_oe1_oe3_20261031/3b/cost/inputs/c60s_RkP96R.wav"
A24 = REPO / "results/baselines_20261030/ameresco/inputs/BAQ_001_03_16_a24.wav"
SYN = REPO / "results/mejora_objetivos_20261101/S7_app/inputs/e2e_synthetic_es.wav"
CALLS: list[int] = []


def pcm_of(path: Path) -> bytes:
    with wave.open(str(path)) as w:
        return w.readframes(w.getnframes())


def cut(pcm: bytes, seconds: float) -> bytes:
    return pcm[: int(round(seconds * SR)) * 2]


def sha(doc: dict) -> str:
    return hashlib.sha256(strip_volatile(doc).encode()).hexdigest()[:16]


class RealRig(Rig):
    def __init__(self, cfg: IncrementalConfig, profiles: list[str]):
        self.analyzer = MrcdAnalyzer(device="cpu")
        self.cfg = cfg
        self.app = create_app(self.analyzer, incremental=cfg, enable_docs=True)
        self.registry = self.app.state.registry
        pre = tuple(AnalysisOptions(asr_profile=p) for p in profiles)
        prepare(self.app, self.analyzer, pre, True)
        self.client = TokenClient(self.app)
        self.client.tokens = {}
        self.raw = TestClient(self.app)

    def wait(self, sid, timeout=3600.0):
        return super().wait(sid, timeout)

    def offline(self, pcm: bytes, **opts):
        # The reference MUST run on the same engine worker thread as the incremental close, exactly as POST /v1/jobs
        # does in the service: the OpenMP/torch thread count is a per-thread setting, so a reference computed on the
        # main thread can differ in provenance.environment.cpu_threads.torch (found by this harness, see README_S8).
        return self.app.state.runner.call(
            lambda: self.analyzer.analyze_bytes(canonical_wav(pcm), AnalysisOptions(**opts))
        )

    def run(self, pcm, batches=None, query="", *, dup=False, order=None):
        CALLS.clear()
        return super().run(pcm, batches, query, dup=dup, order=order)

    @property
    def asr(self):  # noqa: D401 - the base class uses it only to clear a log
        class _L:
            log = CALLS

        return _L


_orig = VerbatimASR.transcribe


def _counting(self, audio):
    CALLS.append(len(audio))
    return _orig(self, audio)


VerbatimASR.transcribe = _counting


COLD_SCRIPT = """
import hashlib, json, sys, threading
from mrcd.service import AnalysisOptions, MrcdAnalyzer
from mrcd.service.incremental.session import canonical_wav
a = MrcdAnalyzer(device="cpu")
for p in ("p0", "p1_disfluent_prompt"):
    a.preload(AnalysisOptions(asr_profile=p))
pcm = sys.stdin.buffer.read()
box = {}
t = threading.Thread(target=lambda: box.update(d=a.analyze_bytes(canonical_wav(pcm), AnalysisOptions(include_text=True))))
t.start(); t.join()
d = box["d"]; d.pop("generated_at", None)
print(hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16])
"""


def cold_process_digest(pcm: bytes) -> str:
    """Digest of the offline document from a fresh process WITHOUT warm-up (preload only), run on a worker thread."""
    import subprocess

    env = {**os.environ, "OMP_NUM_THREADS": "4", "HF_HUB_OFFLINE": "1", "PYTHONPATH": str(REPO)}
    out = subprocess.run(
        [sys.executable, "-c", COLD_SCRIPT], input=pcm, capture_output=True, env=env, cwd=REPO, timeout=900
    )
    return out.stdout.decode().strip().splitlines()[-1] if out.stdout else "error:" + out.stderr.decode()[-200:]


RESULTS: list[dict] = []
OUT_FILE: Path | None = None


def record(case: str, ok: bool, **extra):
    RESULTS.append({"case": case, "pass": bool(ok), **extra})
    print(("PASS" if ok else "FAIL"), case, extra, flush=True)
    if OUT_FILE is not None:
        OUT_FILE.write_text(json.dumps({"results": RESULTS}, indent=2), encoding="utf-8")


def compare(
    rig: RealRig, case: str, pcm: bytes, query: str, opts: dict, batches=None, **drive
):
    t0 = time.perf_counter()
    off = rig.offline(pcm, **opts)
    t_off = time.perf_counter() - t0
    t0 = time.perf_counter()
    st = rig.run(pcm, batches or split(pcm, 2560), query, **drive)
    t_inc = time.perf_counter() - t0
    res = st["result"]
    ok = st["status"] == off["status"] and strip_volatile(res) == strip_volatile(off)

    def n_profile(d):  # events of the o2026_11 pipelines (score kind unit_pipeline_model_score)
        return sum(1 for e in (d or {}).get("events", []) if e["score"]["kind"] == "unit_pipeline_model_score")

    n_off, n_inc = n_profile(off), n_profile(res)
    ok = ok and n_off == n_inc
    if case.startswith("c08_599"):  # the long case must actually exercise the profile
        ok = ok and n_off > 0
    n = len(pcm) // 2
    record(
        case,
        ok,
        seconds=round(n / SR, 3),
        offline_sha=sha(off),
        profile_events_offline=n_off,
        profile_events_incremental=n_inc,
        incremental_sha=sha(res),
        real_decodes=len(CALLS),
        expected_decodes=len(plan.final_plan(n)),
        cache=getattr(rig.registry.get(st["sid"]), "cache_stats", None),
        wall_offline_s=round(t_off, 2),
        wall_incremental_incl_send_s=round(t_inc, 2),
    )
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--quick",
        action="store_true",
        help="skip the 599.9 s case",
    )
    ap.add_argument("--only-long", action="store_true", help="run only the 599.9 s case (p0)")
    ap.add_argument("--tag", default="equivalence_real")
    ap.add_argument(
        "--out", default=str(REPO / "results/mejora_objetivos_20261101/S8_incremental")
    )
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    global OUT_FILE
    OUT_FILE = out / f"{args.tag}.json"
    import faulthandler

    faulthandler.dump_traceback_later(1800, repeat=True)

    c60, a24, syn = pcm_of(C60), pcm_of(A24), pcm_of(SYN)
    cfg = IncrementalConfig(max_batch_bytes=64 << 20, max_sessions=6)
    rig = RealRig(cfg, ["p0", "p1_disfluent_prompt"])
    health = rig.client.get("/v1/health").json()
    print("health", health, flush=True)

    # --- cases 1-8 (+19: p0 and p1)
    cuts = {
        "c01_13.0s": cut(c60, 13.0),
        "c02_13.6s_ameresco": a24,
        "c03_14.0s_6+8": cut(c60, 14.0),
        "c04_30.0s": cut(c60, 30.0),
        "c04_48.0s": cut(c60, 48.0),
        "c05_31.0s": cut(c60, 31.0),
        "c05_32.0s": cut(c60, 32.0),
        "c06_37.0s_last1s": cut(c60, 37.0),
        "c06_36.999s_lastlt1s": cut(c60, 36.999),
        "c07_45.466s_synthetic": syn,
        "c08_60.0s": c60,
    }
    if args.only_long:
        cuts = {}
    for prof in ("p0", "p1_disfluent_prompt"):
        for name, pcm in cuts.items():
            q = f"?asr_profile={prof}&include_text=true&decision=true"
            compare(
                rig,
                f"{name}|{prof}",
                pcm,
                q,
                dict(asr_profile=prof, include_text=True, decision=True),
            )
    if args.only_long or not args.quick:
        long_pcm = (c60 * 10)[: int(599.9 * SR) * 2]
        compare(
            rig,
            "c08_599.9s|p0",
            long_pcm,
            "?include_text=true",
            dict(include_text=True),
            split(long_pcm, 32000),
        )

    if args.only_long:
        return 0 if all(r["pass"] for r in RESULTS) else 1

    # --- case 11 partitions, 12 order, 13 duplicates (45.466 s, p0)
    q, o = "?include_text=true", dict(include_text=True)
    for label, batches in (
        ("160ms", split(syn, 2560)),
        ("2s", split(syn, 32000)),
        ("random", split(syn, rng_seed=20261101)),
        ("one_block", split(syn, len(syn))),
    ):
        compare(rig, f"c11_partition_{label}", syn, q, o, batches)
    b2 = split(syn, 32000)
    order = list(range(len(b2)))
    random.Random(5).shuffle(order)
    compare(rig, "c12_out_of_order", syn, q, o, b2, order=order)
    compare(rig, "c13_duplicates", syn, q, o, b2, dup=True)

    # --- cases 14-16
    sid = rig.open(q)
    assert rig.put(sid, 0, *b2[0]).status_code == 200
    bad = rig.put(sid, 0, b2[0][0], b"\x01\x00" * 16000)
    for i, (off, data) in enumerate(b2[1:], start=1):
        rig.put(sid, i, off, data)
    rig.close(sid, syn, len(b2) - 1)
    st = rig.wait(sid)
    record(
        "c14_conflict",
        bad.status_code == 409
        and bad.json()["error"]["code"] == "sequence_conflict"
        and strip_volatile(st["result"]) == strip_volatile(rig.offline(syn, **o)),
    )
    sid = rig.open(q)
    for i, (off, data) in enumerate(b2):
        if i != 3:
            rig.put(sid, i, off, data)
    gap = rig.close(sid, syn, len(b2) - 1)
    rig.put(sid, 3, *b2[3])
    rig.close(sid, syn, len(b2) - 1)
    st = rig.wait(sid)
    record(
        "c15_gap_then_resend",
        gap.status_code == 409
        and gap.json()["error"]["code"] == "audio_gap"
        and strip_volatile(st["result"]) == strip_volatile(rig.offline(syn, **o)),
    )
    sid = rig.open(q)
    rnd = random.Random(9)
    for i, (off, data) in enumerate(b2):
        for _ in range(rnd.choice([1, 2, 3])):
            rig.put(sid, i, off, data)
    same_pcm = (
        hashlib.sha256(bytes(rig.registry.get(sid).buf)).hexdigest()
        == hashlib.sha256(syn).hexdigest()
    )
    rig.close(sid, syn, len(b2) - 1)
    st = rig.wait(sid)
    record(
        "c16_retries_no_duplicate",
        same_pcm
        and strip_volatile(st["result"]) == strip_volatile(rig.offline(syn, **o)),
    )

    # --- case 17 (restart: a second service has no sessions) and 18 (cancel)
    rig2 = Rig.__new__(Rig)
    rig2.client = TokenClient(create_app(rig.analyzer, incremental=cfg))
    rig2.client.tokens = {}
    r1 = rig2.put(sid, 1, *b2[1])
    record(
        "c17_restart_session_not_found",
        r1.status_code == 404 and r1.json()["error"]["code"] == "session_not_found",
    )
    sid = rig.open()
    rig.put(sid, 0, *b2[0])
    rig.client.delete(f"/v1/sessions/{sid}")
    sess = rig.registry.get(sid) if sid in rig.registry._sessions else None
    gone = (
        rig.put(sid, 1, *b2[1]).status_code == 404
        and rig.close(sid, syn, 1).status_code == 404
    )
    record("c18_cancel", sess is None and gone)

    # --- case 20 noloop: refused for incremental sessions (the sampled rung is not bit-reproducible; S8 review H8)
    for prof in ("p0_noloop", "p1_noloop"):
        rig3 = RealRig(cfg, [prof])
        r = rig3.client.post(f"/v1/sessions?asr_profile={prof}")
        record(
            f"c20_{prof}_refused",
            r.status_code == 422
            and r.json()["error"]["code"] == "profile_not_supported_incremental",
        )

    # --- case 22 interleaving + job, 23 options, 24 errors
    sources = [cut(c60, 30.0), syn, a24]
    allb = [split(p, 32000) for p in sources]
    sids = [rig.open(q) for _ in sources]
    for step in range(max(len(b) for b in allb)):
        for j, b in enumerate(allb):
            if step < len(b):
                rig.put(sids[j], step, *b[step])
        if step == 1:
            rig.client.post(
                "/v1/jobs?include_text=true", content=canonical_wav(sources[0])
            )
    oks = []
    for j, p in enumerate(sources):
        rig.close(sids[j], p, len(allb[j]) - 1)
    for j, p in enumerate(sources):
        oks.append(
            strip_volatile(rig.wait(sids[j])["result"])
            == strip_volatile(rig.offline(p, **o))
        )
    record("c22_interleaved_with_job", all(oks))
    for label, query, opts in (
        (
            "include_out_of_scope",
            "?include_out_of_scope=true",
            dict(output_scope="all"),
        ),
        ("decision", "?decision=true", dict(decision=True)),
        (
            "all",
            "?include_text=true&decision=true&include_out_of_scope=true",
            dict(include_text=True, decision=True, output_scope="all"),
        ),
    ):
        compare(rig, f"c23_{label}", cut(c60, 30.0), query, opts)
    short = cut(c60, 9.0)
    st = rig.run(short, split(short, 32000))
    off = rig.offline(short)
    record(
        "c24_audio_too_short",
        st["status"] == "failed"
        and st["result"]["error"]["code"] == off["error"]["code"] == "audio_too_short",
    )

    # --- cases 21 (warm-up) and 27 (process state)
    cold_sha = cold_process_digest(syn)
    warm_doc = rig.offline(syn, include_text=True)
    record(
        "c21_with_and_without_warmup",
        cold_sha == hashlib.sha256(strip_volatile(warm_doc).encode()).hexdigest()[:16],
        note="cold = fresh process, preload only, no warm-up; warm = preload + warm-up (service)",
    )
    p1_inc = compare(
        rig,
        "c27_p1_incremental_after_p0_offline",
        cuts["c04_30.0s"],
        "?asr_profile=p1_disfluent_prompt",
        dict(asr_profile="p1_disfluent_prompt"),
    )
    p0_inc = compare(
        rig, "c27_p0_incremental_after_p1_offline", cuts["c04_30.0s"], "", {}
    )
    threads = {
        d["provenance"]["environment"]["cpu_threads"]["torch"]
        for d in (
            rig.offline(cuts["c04_30.0s"]),
            rig.offline(cuts["c04_30.0s"], asr_profile="p1_disfluent_prompt"),
        )
    }
    record(
        "c27_single_torch_thread_value",
        len(threads) == 1,
        torch_threads=sorted(threads),
        health_torch_threads=rig.client.get("/v1/health").json()["torch_threads"],
    )

    (out / f"{args.tag}.json").write_text(
        json.dumps({"health": health, "results": RESULTS}, indent=2), encoding="utf-8"
    )
    failed = [r for r in RESULTS if not r["pass"]]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} passed", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
