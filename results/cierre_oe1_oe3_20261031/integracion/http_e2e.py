"""End-to-end check of the local dev server with the REAL engine: POST audio, poll the job, save the responses.

    OMP_NUM_THREADS=4 .venv/Scripts/python.exe results/cierre_oe1_oe3_20261031/integracion/http_e2e.py <clip.wav>

Starts `python -m mrcd.service serve` on 127.0.0.1:8765 as a subprocess, talks to it with urllib (stdlib),
validates every response against the v1 contract and writes the examples to ../fixtures/http_*.json.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from mrcd.service import validate_document  # noqa: E402

OUT = Path(__file__).resolve().parent / "fixtures"
BASE = "http://127.0.0.1:8765"


def call(method: str, path: str, body: bytes | None = None) -> tuple[int, dict]:
    req = urllib.request.Request(
        BASE + path, data=body, method=method, headers={"Content-Type": "audio/wav"}
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def save(name: str, doc: dict) -> None:
    (OUT / name).write_text(
        json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main(clip: str) -> int:
    server = subprocess.Popen(
        [sys.executable, "-m", "mrcd.service", "serve", "--port", "8765"], cwd=ROOT
    )
    try:
        for _ in range(60):
            try:
                print("health:", call("GET", "/v1/health"))
                break
            except Exception:  # noqa: BLE001 - server still starting
                time.sleep(0.5)
        status, queued = call("POST", "/v1/jobs", Path(clip).read_bytes())
        print("POST /v1/jobs ->", status, queued["status"])
        assert status == 202 and validate_document(queued) == []
        save("http_job_queued.json", queued)
        seen = [queued["status"]]
        while True:
            status, job = call("GET", f"/v1/jobs/{queued['job_id']}")
            assert validate_document(job) == [], validate_document(job)
            if job["status"] != seen[-1]:
                seen.append(job["status"])
            if job["status"] in ("completed", "failed"):
                break
            time.sleep(1)
        print("status sequence:", seen, "events:", len(job["result"]["events"]))
        save("http_job_completed.json", job)
        status, missing = call("GET", "/v1/jobs/" + "0" * 32)
        print("unknown job ->", status, missing["error"]["code"])
        assert status == 404 and validate_document(missing) == []
        save("http_job_not_found.json", missing)
        status, bad = call("POST", "/v1/jobs?asr_profile=p9", b"x")
        print("bad profile ->", status, bad["error"]["code"])
        assert status == 400 and validate_document(bad) == []
        # a body that is not audio is accepted (202) and fails inside the job, with a structured error
        status, queued2 = call("POST", "/v1/jobs", b"not a wav")
        while True:
            _, job2 = call("GET", f"/v1/jobs/{queued2['job_id']}")
            if job2["status"] in ("completed", "failed"):
                break
            time.sleep(0.2)
        print("garbage body -> job", job2["status"], job2["result"]["error"]["code"])
        assert validate_document(job2) == []
        save("http_job_failed_invalid_audio.json", job2)
        return 0
    finally:
        server.terminate()
        server.wait(timeout=10)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
