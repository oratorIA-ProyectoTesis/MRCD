"""Minimal HTTP client for the MRCD service (standard library only).

    from mrcd.client import MrcdClient

    client = MrcdClient("http://127.0.0.1:8765")
    doc = client.analyze("sample_es.wav", include_text=True)   # blocks until the job ends
    for ev in doc["events"]:
        print(ev["class"], ev["start_s"], ev["end_s"], ev["score"]["value"], "(uncalibrated)")

The service takes a 16 kHz mono 16-bit PCM WAV (``mrcd sample`` writes one). ``analyze`` returns the
``mrcd.analysis/v1`` document whether it completed or failed; check ``doc["status"]`` and ``doc["error"]``.
HTTP errors carry the same structured document in :class:`MrcdError`.

Incremental sessions (service started with ``--incremental``) send raw PCM16 while it is being recorded and get the
result shortly after the close::

    with client.open_session() as session:
        for pcm in chunks:          # bytes, PCM16 little endian, 16 kHz mono
            session.send(pcm)
        doc = session.close()

Standard library only: the file can be copied as is into any project that does not install ``mrcd``
(imported as ``mrcd.client`` it also loads the ``mrcd`` package, which imports numpy).
Command line: ``python -m mrcd.client file.wav [--url URL] [--include-text]``.
"""

from __future__ import annotations

import hashlib
import json
import struct
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_URL = "http://127.0.0.1:8765"
SAMPLE_RATE = 16000
_OPTION_NAMES = (
    "asr_profile",
    "include_text",
    "include_out_of_scope",
    "decision",
    "decision_profile",
    "text_route",
)


class MrcdError(Exception):
    """Non-2xx answer. ``document`` is the service's structured error document when it sent one."""

    def __init__(self, status: int, document: Any, message: str | None = None):
        self.status = status
        self.document = document
        err = (document or {}).get("error") if isinstance(document, dict) else None
        self.code = (err or {}).get("code")
        super().__init__(
            message or f"HTTP {status}: {(err or {}).get('message') or document}"
        )


def pcm16_wav(pcm: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Mono PCM16 LE bytes -> canonical 44-byte-header WAV (the bytes the session close hashes)."""
    n = len(pcm)
    header = (
        b"RIFF"
        + struct.pack("<I", 36 + n)
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
        + b"data"
        + struct.pack("<I", n)
    )
    return header + bytes(pcm)


def _query(options: dict) -> str:
    unknown = set(options) - set(_OPTION_NAMES)
    if unknown:
        raise TypeError(
            f"unknown option(s): {sorted(unknown)}; allowed: {list(_OPTION_NAMES)}"
        )
    q = {}
    for k, v in options.items():
        if v is None:
            continue
        q[k] = ("true" if v else "false") if isinstance(v, bool) else str(v)
    return urllib.parse.urlencode(q)


class MrcdClient:
    def __init__(self, base_url: str = DEFAULT_URL, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # ------------------------------------------------------------------ transport
    def _request(
        self,
        method: str,
        path: str,
        body: bytes | None = None,
        headers: dict | None = None,
        accept_status: tuple[int, ...] = (200, 201, 202, 204),
    ) -> tuple[int, Any]:
        req = urllib.request.Request(
            self.base_url + path, data=body, method=method, headers=headers or {}
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                status, raw = resp.status, resp.read()
        except urllib.error.HTTPError as exc:
            status, raw = exc.code, exc.read()
        doc: Any = None
        if raw:
            try:
                doc = json.loads(raw.decode("utf-8"))
            except ValueError:
                doc = raw.decode("utf-8", "replace")
        if status not in accept_status:
            raise MrcdError(status, doc)
        return status, doc

    # ------------------------------------------------------------------ simple endpoints
    def health(self) -> dict:
        return self._request("GET", "/v1/health")[1]

    def schema(self) -> dict:
        return self._request("GET", "/v1/schema")[1]

    def submit(self, wav: bytes | str | Path, **options: Any) -> dict:
        """POST /v1/jobs. ``wav`` is the WAV bytes or a path. Returns the job (``mrcd.job/v1``)."""
        data = Path(wav).read_bytes() if isinstance(wav, (str, Path)) else bytes(wav)
        q = _query(options)
        return self._request(
            "POST",
            "/v1/jobs" + (f"?{q}" if q else ""),
            data,
            {"Content-Type": "audio/wav"},
        )[1]

    def job(self, job_id: str) -> dict:
        return self._request("GET", f"/v1/jobs/{urllib.parse.quote(job_id)}")[1]

    def analyze(
        self,
        wav: bytes | str | Path,
        *,
        poll_s: float = 0.5,
        timeout_s: float = 900.0,
        **options: Any,
    ) -> dict:
        """Submit and wait. Returns the ``mrcd.analysis/v1`` document (``status`` completed or failed)."""
        job = self.submit(wav, **options)
        deadline = time.monotonic() + timeout_s
        while job["status"] in ("queued", "running"):
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"job {job['job_id']} did not finish in {timeout_s} s"
                )
            time.sleep(poll_s)
            job = self.job(job["job_id"])
        return job["result"]

    def open_session(self, **options: Any) -> "IncrementalSession":
        q = _query(options)
        body = self._request("POST", "/v1/sessions" + (f"?{q}" if q else ""))[1]
        return IncrementalSession(self, body)


class IncrementalSession:
    """One ``/v1/sessions`` session: ``send`` PCM16 batches in order, then ``close`` to get the result."""

    def __init__(self, client: MrcdClient, opened: dict):
        self.client = client
        self.id = opened["session_id"]
        self._token = opened["subscription_token"]
        self.plan = opened.get("plan", {})
        self.limits = opened.get("limits", {})
        self._seq = 0
        self._offset = 0
        self._hash = hashlib.sha256()
        self._pcm = bytearray()
        self.result: dict | None = None

    def _headers(self, extra: dict | None = None) -> dict:
        return {"X-Session-Token": self._token, **(extra or {})}

    def send(self, pcm: bytes) -> dict:
        """PUT one batch of PCM16 LE mono 16 kHz audio. Returns the progress reply."""
        if not pcm or len(pcm) % 2:
            raise ValueError(
                "a batch must be a non-empty whole number of PCM16 samples"
            )
        max_batch = int(self.limits.get("max_batch_bytes") or len(pcm))
        reply: dict = {}
        for i in range(0, len(pcm), max_batch):
            part = bytes(pcm[i : i + max_batch])
            reply = self.client._request(
                "PUT",
                f"/v1/sessions/{self.id}/audio/{self._seq}",
                part,
                self._headers(
                    {
                        "Content-Type": "application/octet-stream",
                        "X-Offset-Samples": str(self._offset),
                        "X-Content-SHA256": hashlib.sha256(part).hexdigest(),
                    }
                ),
            )[1]
            self._seq += 1
            self._offset += len(part) // 2
            self._pcm += part
        return reply

    def state(self) -> dict:
        return self.client._request(
            "GET", f"/v1/sessions/{self.id}", headers=self._headers()
        )[1]

    def close(self, *, poll_s: float = 0.25, timeout_s: float = 900.0) -> dict:
        """Close and wait for the full-audio analysis. Returns the ``mrcd.analysis/v1`` document."""
        payload = {
            "total_samples": self._offset,
            "wav_sha256": hashlib.sha256(pcm16_wav(bytes(self._pcm))).hexdigest(),
            "last_seq": self._seq - 1,
        }
        self.client._request(
            "POST",
            f"/v1/sessions/{self.id}/close",
            json.dumps(payload).encode("utf-8"),
            self._headers({"Content-Type": "application/json"}),
        )
        deadline = time.monotonic() + timeout_s
        while True:
            state = self.state()
            if state.get("status") in ("completed", "failed"):
                self.result = state.get("result")
                self._pcm = bytearray()
                return self.result
            if time.monotonic() > deadline:
                raise TimeoutError(f"session {self.id} did not finish in {timeout_s} s")
            time.sleep(poll_s)

    def delete(self) -> None:
        self.client._request(
            "DELETE", f"/v1/sessions/{self.id}", headers=self._headers()
        )

    def __enter__(self) -> "IncrementalSession":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if (
            exc_type is not None
        ):  # abandon the session on error; a finished session is left to expire
            try:
                self.delete()
            except Exception:  # noqa: BLE001
                pass


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    p = argparse.ArgumentParser(
        prog="python -m mrcd.client",
        description="Analyse a WAV with a running MRCD service.",
    )
    p.add_argument("wav")
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--include-text", action="store_true")
    p.add_argument("--decision", action="store_true")
    p.add_argument("--out", help="write the JSON (UTF-8) here instead of stdout")
    args = p.parse_args(argv)
    try:
        doc = MrcdClient(args.url).analyze(
            args.wav, include_text=args.include_text, decision=args.decision
        )
    except MrcdError as exc:
        print(json.dumps(exc.document, indent=2), file=sys.stderr)
        return 1
    if args.out:
        Path(args.out).write_text(
            json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    else:  # ASCII escapes: safe on any console encoding
        print(json.dumps(doc, indent=2))
    return 0 if doc.get("status") == "completed" else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
