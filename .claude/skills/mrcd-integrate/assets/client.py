"""Stdlib MRCD client with retry on retryable errors. Usage: python client.py file.wav"""

import json
import os
import sys
import time
import urllib.error
import urllib.request

BASE = os.environ.get("MRCD_URL", "http://127.0.0.1:8765").rstrip("/")


def call(method, path, data=None):
    req = urllib.request.Request(
        BASE + path,
        data=data,
        method=method,
        headers={"Content-Type": "audio/wav"} if data else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read())
    except (
        urllib.error.HTTPError
    ) as e:  # body is an analysis document with status "failed"
        return e.code, json.loads(e.read() or b"{}")


def analyze(wav_bytes, attempts=3):
    for attempt in range(attempts):
        status, job = call("POST", "/v1/jobs?include_text=false", wav_bytes)
        if status == 202:
            break
        err = job.get("error") or {}
        if not err.get("retryable") or attempt == attempts - 1:
            raise RuntimeError(
                f"submit failed: HTTP {status} {err.get('code')}: {err.get('message')}"
            )
        time.sleep(2**attempt)
    while job["status"] not in ("completed", "failed"):
        time.sleep(1)
        _, job = call("GET", f"/v1/jobs/{job['job_id']}")
    result = job["result"]
    if result["status"] == "failed":
        e = result["error"]
        raise RuntimeError(
            f"analysis failed: {e['code']} (retryable={e['retryable']}): {e['message']}"
        )
    return result


if __name__ == "__main__":
    with open(sys.argv[1], "rb") as f:
        doc = analyze(f.read())
    for ev in doc["events"]:
        print(
            ev["class"],
            ev["start_s"],
            ev["end_s"],
            ev["score"]["value"],
            "(uncalibrated)",
        )
    epm = doc["summary"]["events_per_minute"]
    print(
        "events/min:",
        epm["value"]
        if epm["available"]
        else f"not available: {epm['unavailable_reason']}",
    )
