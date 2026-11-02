"""Test rig for the B incremental variant: a content-sensitive fake ASR/engine behind the real service, and helpers
that drive a whole session through the HTTP surface and compare it with the offline path.

The fake ASR derives every word from the exact bytes of each second of the chunk it decodes, so a wrong chunk, a wrong
offset or a wrong merge changes the document (it cannot pass by accident). The real-Whisper equivalence is checked by
``scripts/s8_incremental/equivalence_real.py`` and the ``MRCD_SLOW`` tests.
"""

from __future__ import annotations

import hashlib
import json
import random
import time

import numpy as np
from starlette.testclient import TestClient

from mrcd.constants import SAMPLE_RATE
from mrcd.engine import AnalysisResult, Event
from mrcd.extractors.linguistic import VerbatimASR, Word
from mrcd.service import AnalysisOptions, MrcdAnalyzer
from mrcd.service.incremental.session import IncrementalConfig, canonical_wav
from mrcd.service.server import create_app


class HashASR(VerbatimASR):
    """Words derived from the exact audio of each second. ``log`` records every real decode (shared on copies)."""

    def __init__(self, with_redecode: bool = False):
        self.model_size, self.device, self.compute_type, self.language = (
            "fake",
            "cpu",
            "int8",
            "es",
        )
        self.log: list[tuple[int, str]] = []
        self.with_redecode = with_redecode
        self.redecode_events: list[dict] = []
        self._chunk_offset_s = 0.0
        self.delay_s = 0.0

    def model_info(self) -> dict:
        return {"model_size": "fake", "model_bin_sha256": None, "compute_type": "int8", "device": "cpu"}

    def transcribe(self, audio: np.ndarray) -> list[Word]:
        audio = np.asarray(audio, dtype=np.float32)
        self.log.append((len(audio), hashlib.sha256(audio.tobytes()).hexdigest()[:12]))
        if self.delay_s:
            time.sleep(self.delay_s)
        out = []
        for i in range(len(audio) // SAMPLE_RATE):
            h = hashlib.sha256(
                audio[i * SAMPLE_RATE : (i + 1) * SAMPLE_RATE].tobytes()
            ).digest()
            text = "w" + h.hex()[:6]
            start = i + (h[9] % 40) / 100
            out.append(Word(text, text, start, start + 0.3, 0.5 + h[8] / 512, ""))
        if self.with_redecode and out and out[0].text[1] in "01234567":
            self.redecode_events.append(
                {
                    "attempt": 1,
                    "chunk_offset_s": round(float(self._chunk_offset_s), 3),
                    "words_before": len(out),
                }
            )
        return out


class FakeEngine:
    mode = "audio_text"
    labels = ["fluent", "filler_word"]
    warning = "weak supervision"
    checkpoint_name = "fake.pt"
    checkpoint_sha256 = "d" * 64

    def __init__(self, asr: HashASR):
        self._asr = asr
        self.words_log: list[list[Word]] = []
        self.events_log: list[list[dict]] = []

    def analyze(self, audio, **kw):
        if hasattr(self._asr, "redecode_events"):
            self._asr.redecode_events = []
        words = self._asr.transcribe_chunked(audio, chunk_s=8.0)
        self.words_log.append(words)
        self.events_log.append(list(getattr(self._asr, "redecode_events", [])))
        events = [
            Event(
                "filler_word",
                w.start,
                w.end + 0.2,
                w.prob,
                1,
                modalities=("acoustic", "linguistic"),
            )
            for w in words
            if int(w.text[1:3], 16) % 5 == 0
        ]
        return AnalysisResult(
            events=events,
            transcript=" ".join(w.text for w in words),
            duration=len(audio) / SAMPLE_RATE,
            mode="audio_text",
            warning=self.warning,
            metadata={
                "asr_profile": {
                    "name": "p0",
                    "experimental": False,
                    "model": {},
                    "anomalies": [],
                },
                "window_plan": {},
                "environment": {},
            },
        )


def make_pcm(seconds: float, seed: int = 7) -> bytes:
    rng = np.random.default_rng(seed)
    n = int(round(seconds * SAMPLE_RATE))
    return (rng.integers(-3000, 3000, n)).astype("<i2").tobytes()


def strip_volatile(doc: dict) -> str:
    """Closed list of non-deterministic fields (DESIGN_B_INCREMENTAL section 3): only ``generated_at``."""
    d = json.loads(json.dumps(doc))
    d.pop("generated_at", None)
    return json.dumps(d, sort_keys=True)


class TokenClient(TestClient):
    """TestClient that presents the session token it saw at open time (a test can pass its own ``X-Session-Token``,
    even a wrong one, or use ``Rig.raw`` for requests without any token)."""

    tokens: dict[str, str]

    def request(self, method, url, **kw):  # noqa: D102
        parts = str(url).split("?")[0].split("/")
        if "sessions" in parts:
            i = parts.index("sessions")
            sid = parts[i + 1] if len(parts) > i + 1 else None
            tok = getattr(self, "tokens", {}).get(sid)
            if tok:
                headers = dict(kw.get("headers") or {})
                headers.setdefault("X-Session-Token", tok)
                kw["headers"] = headers
        return super().request(method, url, **kw)


class Rig:
    def __init__(
        self,
        cfg: IncrementalConfig | None = None,
        profiles=("p0",),
        with_redecode=False,
        incremental=True,
    ):
        self.asr = HashASR(with_redecode)
        self.engine = FakeEngine(self.asr)
        self.analyzer = MrcdAnalyzer("u.pt", engine_factory=lambda o: self.engine)
        self.profiles = profiles
        for p in profiles:
            self.analyzer.preload(AnalysisOptions(asr_profile=p))
        self.cfg = cfg or IncrementalConfig()
        self.app = create_app(
            self.analyzer,
            incremental=self.cfg if incremental else None,
            enable_docs=True,
        )
        self.client = TokenClient(self.app)
        self.client.tokens = {}
        self.raw = TestClient(self.app)  # never presents a token
        self.registry = self.app.state.registry

    # ------------------------------------------------------------------ offline reference
    def offline(self, pcm: bytes, **opts) -> dict:
        self.asr.log.clear()
        return self.analyzer.analyze_bytes(canonical_wav(pcm), AnalysisOptions(**opts))

    # ------------------------------------------------------------------ incremental
    def open(self, query: str = "") -> str:
        r = self.client.post(f"/v1/sessions{query}")
        assert r.status_code == 201, r.text
        sid = r.json()["session_id"]
        self.client.tokens[sid] = r.json()["subscription_token"]
        return sid

    def put(
        self, sid: str, seq: int, offset: int, data: bytes, sha: str | None = "auto"
    ):
        headers = {"X-Offset-Samples": str(offset)}
        if sha == "auto":
            sha = hashlib.sha256(data).hexdigest()
        if sha:
            headers["X-Content-SHA256"] = sha
        return self.client.put(
            f"/v1/sessions/{sid}/audio/{seq}", content=data, headers=headers
        )

    def close(self, sid: str, pcm: bytes, last_seq: int):
        wav = canonical_wav(pcm)
        return self.client.post(
            f"/v1/sessions/{sid}/close",
            json={
                "total_samples": len(pcm) // 2,
                "wav_sha256": hashlib.sha256(wav).hexdigest(),
                "last_seq": last_seq,
            },
        )

    def wait(self, sid: str, timeout: float = 30.0) -> dict:
        end = time.time() + timeout
        while time.time() < end:
            st = self.client.get(f"/v1/sessions/{sid}").json()
            if st["status"] in ("completed", "failed", "expired", "cancelled"):
                return st
            time.sleep(0.01)
        raise AssertionError("session did not finish")

    def run(
        self,
        pcm: bytes,
        batches=None,
        query: str = "",
        *,
        dup: bool = False,
        order=None,
    ) -> dict:
        """Whole session: open, send (optionally permuted / duplicated), close, wait. Returns the final state."""
        batches = batches or split(pcm, 2560)
        sid = self.open(query)
        self.asr.log.clear()
        seq_order = order if order is not None else list(range(len(batches)))
        for i in seq_order:
            off, data = batches[i]
            r = self.put(sid, i, off, data)
            assert r.status_code == 200, r.text
            if dup:
                r2 = self.put(sid, i, off, data)
                assert r2.status_code == 200 and r2.json()["duplicate"] is True
        r = self.close(sid, pcm, len(batches) - 1)
        assert r.status_code in (200, 202), r.text
        if dup:
            r = self.close(sid, pcm, len(batches) - 1)
            assert r.status_code in (200, 202), r.text
        st = self.wait(sid)
        st["sid"] = sid
        return st


def split(pcm: bytes, size_samples: int | None = None, *, rng_seed: int | None = None):
    """``[(offset_samples, bytes)]`` in fixed-size or seeded random-size (1..40000 samples) batches."""
    out, pos, n = [], 0, len(pcm) // 2
    rnd = random.Random(rng_seed)
    while pos < n:
        step = rnd.randint(1, 40000) if rng_seed is not None else size_samples
        step = min(step, n - pos)
        out.append((pos, pcm[pos * 2 : (pos + step) * 2]))
        pos += step
    return out
