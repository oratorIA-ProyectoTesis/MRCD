"""Analyse a WAV against a running MRCD service using only the standard library.

    python examples/python_client.py recording.wav        (MRCD_URL overrides http://127.0.0.1:8765)

The WAV must be mono, 16 kHz, PCM16, at least 13 s long.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

BASE = os.environ.get("MRCD_URL", "http://127.0.0.1:8765").rstrip("/")


def call(method: str, path: str, data: bytes | None = None) -> dict:
    headers = {"Content-Type": "audio/wav"} if data else {}
    req = urllib.request.Request(BASE + path, data=data, method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.loads(response.read())


def analyze(wav: Path, include_text: bool = False) -> dict:
    job = call(
        "POST",
        f"/v1/jobs?include_text={'true' if include_text else 'false'}",
        wav.read_bytes(),
    )
    while job["status"] not in ("completed", "failed"):
        time.sleep(1)
        job = call("GET", f"/v1/jobs/{job['job_id']}")
    return job["result"]


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    print("service:", call("GET", "/v1/health")["status"])
    doc = analyze(Path(sys.argv[1]))
    if doc["status"] != "completed":
        print("failed:", doc["error"]["code"], "-", doc["error"]["message"])
        return 1
    for ev in doc["events"]:
        print(
            f"{ev['start_s']:6.2f}-{ev['end_s']:6.2f} s  {ev['class']:13s} score {ev['score']['value']:.2f} (uncalibrated)"
        )
    epm = doc["summary"]["events_per_minute"]
    print(
        "events per minute:",
        epm["value"]
        if epm["available"]
        else f"not available: {epm['unavailable_reason']}",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
