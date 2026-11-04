"""Stdlib MRCD client. Usage: python client.py file.wav

Retries (bounded, exponential backoff) when submit returns a retryable error (queue_full) or when the
finished job failed with error.retryable true (analysis_failed); the whole submit + poll cycle is repeated.
Non-retryable errors raise at once. An unknown or evicted job raises job_not_found.
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request

BASE = os.environ.get("MRCD_URL", "http://127.0.0.1:8765").rstrip("/")


class MrcdError(RuntimeError):
    def __init__(self, code, message, retryable=False):
        super().__init__(f"{code}: {message}")
        self.code, self.retryable = code, retryable


def call(method, path, data=None):
    req = urllib.request.Request(
        BASE + path, data=data, method=method,
        headers={"Content-Type": "audio/wav"} if data else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:  # body is an analysis document with status "failed"
        return e.code, json.loads(e.read() or b"{}")


def _error(status, doc):
    err = doc.get("error") or {}
    return MrcdError(err.get("code", f"http_{status}"), err.get("message", ""), bool(err.get("retryable")))


def _run_once(wav_bytes):
    status, job = call("POST", "/v1/jobs?include_text=false", wav_bytes)
    if status != 202:
        raise _error(status, job)
    while job["status"] not in ("completed", "failed"):
        time.sleep(1)
        status, job = call("GET", f"/v1/jobs/{job['job_id']}")
        if status != 200 or "result" not in job:  # unknown or evicted job: error document, no result
            err = _error(status, job)
            raise MrcdError(err.code if "job_not_found" in err.code else "job_not_found", str(err))
    result = job["result"]
    if result["status"] == "failed":
        raise _error(200, result)
    return result


def analyze(wav_bytes, attempts=3):
    for attempt in range(attempts):
        try:
            return _run_once(wav_bytes)
        except MrcdError as e:
            if not e.retryable or attempt == attempts - 1:
                raise
            time.sleep(2 ** attempt)


if __name__ == "__main__":
    with open(sys.argv[1], "rb") as f:
        doc = analyze(f.read())
    for ev in doc["events"]:
        print(ev["class"], ev["start_s"], ev["end_s"], ev["score"]["value"], "(uncalibrated)")
    epm = doc["summary"]["events_per_minute"]
    print("events/min:", epm["value"] if epm["available"] else f"not available: {epm['unavailable_reason']}")
