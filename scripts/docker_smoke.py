#!/usr/bin/env python3
"""Smoke test for a running MRCD service (used by .github/workflows/docker.yml, also handy locally).

    python scripts/docker_smoke.py --url http://127.0.0.1:8765 --expect completed
    python scripts/docker_smoke.py --expect checkpoint_unavailable      # container started without the checkpoint

Steps: wait for GET /v1/health == 200, POST a synthetic 16 kHz mono PCM16 WAV to /v1/jobs, poll the job, validate the
job and its result against mrcd/service/schemas/mrcd.analysis.v1.schema.json (needs `pip install jsonschema`).
The WAV is generated here (harmonic tone + seeded noise, 13 s): no recorded voice is involved. Standard library only,
apart from jsonschema.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import random
import struct
import sys
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path

SCHEMA = (
    Path(__file__).resolve().parents[1]
    / "mrcd"
    / "service"
    / "schemas"
    / "mrcd.analysis.v1.schema.json"
)
RATE = 16000


def synthetic_wav(seconds: float = 13.0) -> bytes:
    rng = random.Random(20261101)
    frames = bytearray()
    for i in range(int(seconds * RATE)):
        t = i / RATE
        x = (
            0.08 * math.sin(2 * math.pi * 140 * t)
            + 0.04 * math.sin(2 * math.pi * 280 * t)
            + 0.01 * rng.gauss(0, 1)
        )
        frames += struct.pack("<h", int(max(-1.0, min(1.0, x)) * 32767))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(bytes(frames))
    return buf.getvalue()


def http(
    method: str, url: str, data: bytes | None = None, timeout: float = 30.0
) -> tuple[int, dict]:
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "audio/wav"} if data else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8765")
    ap.add_argument(
        "--expect", choices=["completed", "checkpoint_unavailable"], default="completed"
    )
    ap.add_argument("--health-timeout-s", type=float, default=600.0)
    ap.add_argument("--job-timeout-s", type=float, default=600.0)
    ap.add_argument("--schema", default=str(SCHEMA))
    ap.add_argument("--write-wav", metavar="PATH", help="only write the synthetic WAV to PATH and exit")
    a = ap.parse_args()
    if a.write_wav:
        Path(a.write_wav).write_bytes(synthetic_wav())
        print("wrote", a.write_wav)
        return 0
    base = a.url.rstrip("/")

    deadline = time.monotonic() + a.health_timeout_s
    while True:
        try:
            status, health = http("GET", f"{base}/v1/health", timeout=5)
            if status == 200:
                break
        except (urllib.error.URLError, OSError, ValueError):
            pass
        if time.monotonic() > deadline:
            print("FAIL: /v1/health never returned 200", file=sys.stderr)
            return 1
        time.sleep(3)
    print("health:", json.dumps(health))

    status, job = http("POST", f"{base}/v1/jobs", synthetic_wav())
    if status != 202:
        print(f"FAIL: POST /v1/jobs returned {status}: {job}", file=sys.stderr)
        return 1
    deadline = time.monotonic() + a.job_timeout_s
    while job["status"] not in ("completed", "failed"):
        if time.monotonic() > deadline:
            print("FAIL: job did not finish in time", file=sys.stderr)
            return 1
        time.sleep(2)
        _, job = http("GET", f"{base}/v1/jobs/{job['job_id']}")
    result = job["result"]
    print("job status:", job["status"], "| error:", (result or {}).get("error"))

    from jsonschema import Draft202012Validator

    schema = json.loads(Path(a.schema).read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    errors = [f"job: {e.message}" for e in validator.iter_errors(job)]
    if result is not None:
        errors += [f"result: {e.message}" for e in validator.iter_errors(result)]
    if errors:
        print("FAIL: schema violations:\n" + "\n".join(errors[:20]), file=sys.stderr)
        return 1

    if a.expect == "completed" and job["status"] != "completed":
        print("FAIL: expected a completed analysis", file=sys.stderr)
        return 1
    if a.expect == "checkpoint_unavailable" and (
        job["status"] != "failed" or result["error"]["code"] != "checkpoint_unavailable"
    ):
        print(
            "FAIL: expected a failed job with error.code == checkpoint_unavailable",
            file=sys.stderr,
        )
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
