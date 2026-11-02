"""Benchmark of B original (offline) vs B incremental against a REAL MRCD service process, in series.

    python -m scripts.s8_incremental.benchmark --out DIR [--quick]

The service is spawned here exactly as documented (OMP_NUM_THREADS=4, HF_HUB_OFFLINE=1, --device cpu --preload p0
--warmup --incremental). The client reproduces a session in REAL TIME (batches of 2 s sent at their wall-clock
moment) and, in the "fast" variant, with no pacing. Per run it records: latency after closing the session (offline:
POST /v1/jobs -> completed; incremental: close -> completed), total wall and CPU seconds of the service process,
peak working set of the service process, whether the final document equals the offline reference (everything except
``generated_at``) and the background CPU load observed before the run. No audio content or text is written.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
import wave
from ctypes import wintypes
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from mrcd.service.incremental.session import canonical_wav  # noqa: E402

SR = 16000
PORT = 8791
BASE = f"http://127.0.0.1:{PORT}"
INPUTS = {
    "c60": REPO / "results/cierre_oe1_oe3_20261031/3b/cost/inputs/c60s_RkP96R.wav",
    "a24": REPO / "results/baselines_20261030/ameresco/inputs/BAQ_001_03_16_a24.wav",
    "syn": REPO
    / "results/mejora_objetivos_20261101/S7_app/inputs/e2e_synthetic_es.wav",
}


# ---------------------------------------------------------------- Windows process stats (no psutil)
class _MemCounters(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


_k32 = ctypes.WinDLL("kernel32", use_last_error=True)
_psapi = ctypes.WinDLL("psapi", use_last_error=True)
_k32.OpenProcess.restype = wintypes.HANDLE
_psapi.GetProcessMemoryInfo.argtypes = [
    wintypes.HANDLE,
    ctypes.POINTER(_MemCounters),
    wintypes.DWORD,
]


def proc_stats(pid: int) -> tuple[float, int]:
    """(cpu_seconds user+kernel, working_set_bytes) of a process."""
    handle = _k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return 0.0, 0
    try:
        c, e, k, u = (wintypes.FILETIME() for _ in range(4))
        _k32.GetProcessTimes(
            handle, ctypes.byref(c), ctypes.byref(e), ctypes.byref(k), ctypes.byref(u)
        )
        to_s = lambda f: ((f.dwHighDateTime << 32) | f.dwLowDateTime) / 1e7  # noqa: E731
        mem = _MemCounters()
        mem.cb = ctypes.sizeof(mem)
        _psapi.GetProcessMemoryInfo(handle, ctypes.byref(mem), mem.cb)
        return to_s(k) + to_s(u), int(mem.WorkingSetSize)
    finally:
        _k32.CloseHandle(handle)


def system_cpu_percent(interval: float = 1.0) -> float:
    class FT(ctypes.Structure):
        _fields_ = [("lo", wintypes.DWORD), ("hi", wintypes.DWORD)]

    def times() -> tuple[int, int, int]:
        i, k, u = FT(), FT(), FT()
        _k32.GetSystemTimes(ctypes.byref(i), ctypes.byref(k), ctypes.byref(u))
        v = lambda f: (f.hi << 32) | f.lo  # noqa: E731
        return v(i), v(k), v(u)

    a = times()
    time.sleep(interval)
    b = times()
    idle, kern, user = (b[0] - a[0]), (b[1] - a[1]), (b[2] - a[2])
    total = kern + user
    return round(100.0 * (total - idle) / total, 1) if total else 0.0


class Sampler(threading.Thread):
    def __init__(self, pid: int):
        super().__init__(daemon=True)
        self.pid, self.peak, self._stop_event = pid, 0, threading.Event()

    def run(self) -> None:
        while not self._stop_event.is_set():
            self.peak = max(self.peak, proc_stats(self.pid)[1])
            time.sleep(0.05)

    def stop(self) -> int:
        self._stop_event.set()
        self.join(1)
        return self.peak


# ---------------------------------------------------------------- HTTP helpers
def http(
    method: str, path: str, body: bytes | None = None, headers: dict | None = None
) -> tuple[int, dict]:
    req = urllib.request.Request(
        BASE + path, data=body, method=method, headers=headers or {}
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            raw = r.read()
            return r.status, (json.loads(raw) if raw else {})
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def wait_health(timeout: float = 300.0) -> float:
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        try:
            if http("GET", "/v1/health")[0] == 200:
                return time.perf_counter() - t0
        except OSError:
            pass
        time.sleep(0.2)
    raise TimeoutError("service did not become healthy")


def spawn(flags: list[str], log: Path) -> subprocess.Popen:
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "4",
        "HF_HUB_OFFLINE": "1",
        "PYTHONUNBUFFERED": "1",
    }
    return subprocess.Popen(
        [
            str(REPO / ".venv/Scripts/python.exe"),
            "-m",
            "mrcd.service",
            "serve",
            "--port",
            str(PORT),
            *flags,
        ],
        cwd=REPO,
        env=env,
        stdout=open(log, "ab"),
        stderr=subprocess.STDOUT,
    )


def service_pid(fallback: int) -> int:
    """The venv python.exe is a launcher stub: the real service is the process listening on PORT."""
    out = subprocess.run(["netstat", "-ano"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if f"127.0.0.1:{PORT} " in line and "LISTENING" in line:
            return int(line.split()[-1])
    return fallback


def stop(p: subprocess.Popen) -> None:
    pid = service_pid(p.pid)
    if pid != p.pid:
        subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True)
    p.terminate()
    try:
        p.wait(20)
    except subprocess.TimeoutExpired:
        p.kill()


def strip(doc: dict) -> str:
    d = json.loads(json.dumps(doc))
    d.pop("generated_at", None)
    return json.dumps(d, sort_keys=True)


def read_pcm(path: Path) -> bytes:
    with wave.open(str(path)) as w:
        return w.readframes(w.getnframes())


def recording(seconds: float) -> bytes:
    if seconds <= 60:
        base = read_pcm(INPUTS["c60"])
        if abs(seconds - 13.6) < 0.01:
            return read_pcm(INPUTS["a24"])
        if abs(seconds - 45.466) < 0.01:
            return read_pcm(INPUTS["syn"])
        return base[: int(seconds * SR) * 2]
    return (read_pcm(INPUTS["c60"]) * 3)[: int(seconds * SR) * 2]


# ---------------------------------------------------------------- one run
def run_offline(pid: int, pcm: bytes, pace: float) -> dict:
    cpu0, _ = proc_stats(pid)
    sampler = Sampler(pid)
    sampler.start()
    t_start = time.perf_counter()
    time.sleep(len(pcm) / 2 / SR * pace)  # the session: nothing reaches MRCD
    t_close = time.perf_counter()
    status, job = http(
        "POST",
        "/v1/jobs?include_text=false",
        canonical_wav(pcm),
        {"Content-Type": "audio/wav"},
    )
    while job.get("status") not in ("completed", "failed"):
        time.sleep(0.1)
        status, job = http("GET", f"/v1/jobs/{job['job_id']}")
    t_done = time.perf_counter()
    peak = sampler.stop()
    cpu1, _ = proc_stats(pid)
    return {
        "latency_s": t_done - t_close,
        "total_wall_s": t_done - t_start,
        "cpu_s": cpu1 - cpu0,
        "peak_ws_mb": peak / 2**20,
        "doc": job["result"],
    }


def run_incremental(pid: int, pcm: bytes, pace: float, batch_s: float = 2.0) -> dict:
    cpu0, _ = proc_stats(pid)
    sampler = Sampler(pid)
    sampler.start()
    t_start = time.perf_counter()
    _, sess = http("POST", "/v1/sessions?include_text=false")
    sid = sess["session_id"]
    tok = {"X-Session-Token": sess["subscription_token"]}
    n = len(pcm) // 2
    step = int(batch_s * SR)
    seq = 0
    for off in range(0, n, step):
        end = min(off + step, n)
        due = (
            t_start + (end / SR) * pace
        )  # a batch exists once its last sample has been captured
        time.sleep(max(0.0, due - time.perf_counter()))
        data = pcm[off * 2 : end * 2]
        http(
            "PUT",
            f"/v1/sessions/{sid}/audio/{seq}",
            data,
            {
                "X-Offset-Samples": str(off),
                "X-Content-SHA256": hashlib.sha256(data).hexdigest(),
                "Content-Type": "application/octet-stream",
                **tok,
            },
        )
        seq += 1
    t_close = time.perf_counter()
    http(
        "POST",
        f"/v1/sessions/{sid}/close",
        json.dumps(
            {
                "total_samples": n,
                "wav_sha256": hashlib.sha256(canonical_wav(pcm)).hexdigest(),
                "last_seq": seq - 1,
            }
        ).encode(),
        {"Content-Type": "application/json", **tok},
    )
    while True:
        _, st = http("GET", f"/v1/sessions/{sid}", None, tok)
        if st["status"] in ("completed", "failed"):
            break
        time.sleep(0.1)
    t_done = time.perf_counter()
    peak = sampler.stop()
    cpu1, _ = proc_stats(pid)
    return {
        "latency_s": t_done - t_close,
        "total_wall_s": t_done - t_start,
        "cpu_s": cpu1 - cpu0,
        "peak_ws_mb": peak / 2**20,
        "doc": st["result"],
    }


def med(xs: list[float]) -> str:
    return f"{statistics.median(xs):.1f} ({min(xs):.1f}-{max(xs):.1f})"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = out / "service_benchmark.log"
    runs: list[dict] = []
    startup: dict = {}

    # --- (5) startup and first request: lazy service (B original as documented) vs preload + warm-up
    a24 = recording(13.6)
    for label, flags in (
        ("lazy", []),
        (
            "preload_warmup",
            ["--incremental", "--device", "cpu", "--preload", "p0", "--warmup"],
        ),
    ):
        t0 = time.perf_counter()
        p = spawn(flags, log)
        ready = wait_health()
        spid = service_pid(p.pid)
        first = run_offline(spid, a24, 0)["latency_s"]
        second = run_offline(spid, a24, 0)["latency_s"]
        startup[label] = {
            "time_to_ready_s": round(ready, 2),
            "first_request_latency_s": round(first, 2),
            "second_request_latency_s": round(second, 2),
        }
        print("startup", label, startup[label], flush=True)
        stop(p)
        time.sleep(2)

    # --- main series on one service process (preload + warm-up), serial
    p = spawn(["--incremental", "--device", "cpu", "--preload", "p0", "--warmup"], log)
    wait_health()
    spid = service_pid(p.pid)
    durations = (
        [13.6, 30.0, 45.466] if args.quick else [13.6, 30.0, 45.466, 60.0, 120.0]
    )
    for pace_label, pace in (("real_time", 1.0), ("fast", 0.0)):
        for dur in durations:
            if pace == 1.0 and dur > 60 and args.quick:
                continue
            pcm = recording(dur)
            rounds = 1 + (2 if dur > 60 else args.rounds)
            for rnd in range(rounds):
                ref = None
                for mode in ("offline", "incremental"):
                    load = system_cpu_percent(1.0)
                    res = (run_offline if mode == "offline" else run_incremental)(
                        spid, pcm, pace
                    )
                    if mode == "offline":
                        ref = strip(res["doc"])
                    row = {
                        "pace": pace_label,
                        "seconds": dur,
                        "round": rnd,
                        "cold_discarded": rnd == 0,
                        "mode": mode,
                        "latency_s": round(res["latency_s"], 2),
                        "total_wall_s": round(res["total_wall_s"], 2),
                        "cpu_s": round(res["cpu_s"], 2),
                        "peak_ws_mb": round(res["peak_ws_mb"], 1),
                        "equal_to_offline": strip(res["doc"]) == ref,
                        "background_cpu_pct_before": load,
                        "status": res["doc"]["status"],
                    }
                    runs.append(row)
                    print(row, flush=True)
    stop(p)

    (out / "runs.json").write_text(
        json.dumps({"startup": startup, "runs": runs}, indent=1), encoding="utf-8"
    )
    lines = [
        "# Benchmark B original vs B incremental (generated)",
        "",
        "Median (min-max) of the measured rounds; the first round of each cell is a discarded cold round.",
        "",
    ]
    for pace_label in ("real_time", "fast"):
        lines += [
            f"## Pace: {pace_label}",
            "",
            "| seconds | mode | latency after close (s) | total wall (s) | service CPU (s) | peak working set (MB) | all equal to offline |",
            "|---|---|---|---|---|---|---|",
        ]
        for dur in sorted({r["seconds"] for r in runs if r["pace"] == pace_label}):
            for mode in ("offline", "incremental"):
                rows = [
                    r
                    for r in runs
                    if r["pace"] == pace_label
                    and r["seconds"] == dur
                    and r["mode"] == mode
                    and not r["cold_discarded"]
                ]
                if rows:
                    lines.append(
                        f"| {dur} | {mode} | {med([r['latency_s'] for r in rows])} | {med([r['total_wall_s'] for r in rows])} | {med([r['cpu_s'] for r in rows])} | {med([r['peak_ws_mb'] for r in rows])} | {all(r['equal_to_offline'] for r in rows)} |"
                    )
        lines.append("")
    lines += [
        "## Startup and first request (13.6 s recording, offline job)",
        "",
        "| service | time to ready (s) | first request (s) | second request (s) |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| {k} | {v['time_to_ready_s']} | {v['first_request_latency_s']} | {v['second_request_latency_s']} |"
        for k, v in startup.items()
    ]
    bg = [r["background_cpu_pct_before"] for r in runs]
    lines += [
        "",
        f"Background system CPU before each run (%): median {statistics.median(bg):.1f}, max {max(bg):.1f}.",
    ]
    (out / "BENCHMARK.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
