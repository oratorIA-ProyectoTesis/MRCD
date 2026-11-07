"""HTTP smoke test of a running (or starting) MRCD service; standard library + ``mrcd.service.validate_document``.

    python scripts/dx/smoke_http.py http://127.0.0.1:8791 OUT_DIR [--t0 EPOCH_SECONDS]

Waits for ``/v1/health``, downloads ``/sample_es.wav`` from the service, analyses it through ``POST /v1/jobs``
(``include_text=true``), validates the document against the v1 contract, fetches the page ``/`` and ``/docs``, and
prints wall-clock times measured from ``--t0`` (the moment the service was started) or from the script start.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def get(url: str, timeout: float = 10) -> tuple[int, bytes, dict]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.read(), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read(), dict(e.headers)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("out")
    ap.add_argument("--t0", type=float, default=None)
    ap.add_argument("--timeout-s", type=float, default=900)
    args = ap.parse_args(argv)
    base, out = args.base.rstrip("/"), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = args.t0 or time.time()
    deadline = time.time() + args.timeout_s

    while True:
        try:
            status, body, _ = get(base + "/v1/health", timeout=3)
            if status == 200:
                break
        except OSError:
            pass
        if time.time() > deadline:
            print("FAIL: service never became healthy")
            return 1
        time.sleep(0.5)
    t_health = time.time()
    (out / "health.json").write_bytes(body)
    print("health:", body.decode())

    status, wav, _ = get(base + "/sample_es.wav")
    assert status == 200, status
    (out / "sample_es.wav").write_bytes(wav)
    print("sample sha256:", hashlib.sha256(wav).hexdigest())

    t_post = time.time()
    req = urllib.request.Request(
        base + "/v1/jobs?include_text=true",
        data=wav,
        method="POST",
        headers={"Content-Type": "audio/wav"},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        job = json.loads(r.read())
        print(
            "POST /v1/jobs ->",
            r.status,
            json.dumps({k: job[k] for k in ("schema_version", "status")}),
        )
    while job["status"] in ("queued", "running"):
        time.sleep(0.25)
        job = json.loads(get(base + f"/v1/jobs/{job['job_id']}")[1])
    t_result = time.time()
    doc = job["result"]
    (out / "result.json").write_text(
        json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    from mrcd.service import validate_document

    errors = validate_document(job) + validate_document(doc)
    print("contract:", "valid" if not errors else errors)
    print(
        "status:",
        doc["status"],
        "| events:",
        [(e["class"], e["start_s"], e["end_s"]) for e in doc["events"]],
    )
    print("transcript:", doc.get("transcript"))

    status, page, headers = get(base + "/")
    (out / "page.html").write_bytes(page)
    print(
        "GET / ->",
        status,
        headers.get("content-type"),
        len(page),
        "bytes;",
        "title ok" if b"<title>MRCD" in page else "NO TITLE",
    )
    print("GET /docs ->", get(base + "/docs")[0])
    print(
        f"times: health {t_health - t0:.1f} s after start; first result {t_result - t0:.1f} s after start "
        f"({t_result - t_post:.1f} s for the analysis request)"
    )
    return 0 if not errors and doc["status"] == "completed" and status == 200 else 1


if __name__ == "__main__":
    sys.exit(main())
