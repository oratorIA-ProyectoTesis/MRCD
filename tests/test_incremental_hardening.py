"""B incremental hardening (review S8): subscription token (H1), exact zeroing (H2), global byte budget (H3),
cancel while closing (H4), scheduling bounds (H5), terminal-event privacy, cache key with the profile (M3) and the loop
re-decode logic (H8, fake ASR; the real-Whisper counterpart is ``test_incremental_real_whisper.py``)."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from mrcd.service import AnalysisOptions
from mrcd.service.incremental import session as session_mod
from mrcd.service.incremental.cache import ChunkCache, segment_key
from mrcd.service.incremental.cached_asr import CachedASR
from mrcd.service.incremental.session import (
    IncrementalConfig,
    canonical_wav,
    session_namespace,
)
from mrcd.service.server import JobRunner
from tests.incremental_rig import HashASR, Rig, make_pcm, split

pytestmark = pytest.mark.timeout(120)


# ------------------------------------------------------------------ H1 subscription token
def test_every_session_endpoint_requires_the_subscription_token():
    rig = Rig()
    pcm = make_pcm(20.0)
    sid = rig.open()
    assert len(rig.client.tokens[sid]) >= 40
    off, data = split(pcm, 32000)[0]
    hdr = {"X-Offset-Samples": str(off)}
    calls = [
        ("put", f"/v1/sessions/{sid}/audio/0", {"content": data, "headers": hdr}),
        (
            "post",
            f"/v1/sessions/{sid}/close",
            {"json": {"total_samples": 1, "wav_sha256": "a", "last_seq": 0}},
        ),
        ("get", f"/v1/sessions/{sid}", {}),
        ("get", f"/v1/sessions/{sid}/events", {}),
        ("delete", f"/v1/sessions/{sid}", {}),
    ]
    for wrong in (None, "x" * 43, rig.client.tokens[sid][:-1]):
        for verb, path, kw in calls:
            kw = dict(kw)
            if wrong:
                kw["headers"] = {**kw.get("headers", {}), "X-Session-Token": wrong}
            r = getattr(rig.raw, verb)(path, **kw)
            if verb == "delete":  # idempotent 204 that does nothing (no existence oracle)
                assert r.status_code == 204, (path, wrong)
                continue
            assert r.status_code == 404, (verb, path, wrong, r.text)
            assert r.json()["error"]["code"] == "session_not_found"
    # nothing was changed or discarded by the refused calls
    st = rig.client.get(f"/v1/sessions/{sid}").json()
    assert st["status"] == "open" and st["received_samples"] == 0
    # the correct token also works as a bearer
    ok = rig.raw.get(
        f"/v1/sessions/{sid}",
        headers={"Authorization": f"Bearer {rig.client.tokens[sid]}"},
    )
    assert ok.status_code == 200


def test_a_token_is_bound_to_its_own_session():
    rig = Rig()
    a, b = rig.open(), rig.open()
    r = rig.raw.get(
        f"/v1/sessions/{a}", headers={"X-Session-Token": rig.client.tokens[b]}
    )
    assert r.status_code == 404
    assert rig.client.tokens[a] != rig.client.tokens[b]


def test_delete_gives_the_same_204_for_unknown_id_and_wrong_token_and_acts_only_with_the_token():
    rig = Rig()
    sid = rig.open()
    unknown = rig.raw.delete("/v1/sessions/" + "0" * 32)
    wrong = rig.raw.delete(f"/v1/sessions/{sid}")
    assert (unknown.status_code, unknown.content) == (wrong.status_code, wrong.content) == (204, b"")
    assert rig.registry.get(sid).status == "open"  # the tokenless call did nothing
    assert rig.client.delete(f"/v1/sessions/{sid}").status_code == 204
    assert rig.client.get(f"/v1/sessions/{sid}").status_code == 404
    assert rig.client.delete(f"/v1/sessions/{sid}").status_code == 204


def test_a_refused_close_leaves_no_byte_reservation():
    """R2 (revision 2): a wrong wav_sha256 must not leak the WAV reservation (448000 -> 896044 before the fix)."""
    pcm = make_pcm(14.0)  # 448000 bytes
    rig = Rig(IncrementalConfig(max_total_audio_bytes=len(pcm) * 2 + 44))
    sid = rig.open()
    batches = split(pcm, 32000)
    for i, (o, d) in enumerate(batches):
        assert rig.put(sid, i, o, d).status_code == 200
    before = rig.registry.bytes_in_use
    assert before == len(pcm)
    for _ in range(3):
        r = rig.client.post(
            f"/v1/sessions/{sid}/close",
            json={"total_samples": len(pcm) // 2, "wav_sha256": "0" * 64, "last_seq": len(batches) - 1},
        )
        assert r.status_code == 409 and r.json()["error"]["code"] == "close_mismatch"
        assert rig.registry.bytes_in_use == before
    assert rig.close(sid, pcm, len(batches) - 1).status_code == 202  # the correct close still fits
    assert rig.wait(sid)["status"] in ("completed", "failed")



def test_the_failed_terminal_event_is_counts_only_and_the_error_document_comes_from_get():
    rig = Rig()
    pcm = make_pcm(5.0)  # too short -> failed
    st = rig.run(pcm)
    assert (
        st["status"] == "failed" and st["result"]["error"]["code"] == "audio_too_short"
    )
    sess = rig.registry.get(st["sid"])
    last = sess.events[-1]
    assert last["type"] == "failed" and "result" not in last["data"]
    assert set(last["data"]) <= {
        "status",
        "received_s",
        "chunks_decoded",
        "chunks_complete",
        "session_id",
        "result_available",
    }
    assert "error" not in str(last["data"])


# ------------------------------------------------------------------ H2 exact zeroing of the controlled copies
def test_controlled_transient_copies_are_overwritten_with_zeros(monkeypatch):
    rig = Rig()
    pcm = make_pcm(30.0)
    seen = {"wav": [], "seg": [], "body": []}

    real_analyze = rig.analyzer.analyze_bytes

    def spy_analyze(data, *a, **k):
        seen["wav"].append(data)  # the session's own canonical-WAV buffer (a bytearray)
        assert any(data)
        return real_analyze(data, *a, **k)

    monkeypatch.setattr(rig.analyzer, "analyze_bytes", spy_analyze)
    real_decode = session_mod.decode_with

    def spy_decode(real, seg, start):
        seen["seg"].append(seg)
        assert any(seg)
        return real_decode(real, seg, start)

    monkeypatch.setattr(session_mod, "decode_with", spy_decode)
    real_put = session_mod.IncrementalSession.put

    def spy_put(self, seq, offset, data, sha):
        seen["body"].append(data)
        assert any(data)
        return real_put(self, seq, offset, data, sha)

    monkeypatch.setattr(session_mod.IncrementalSession, "put", spy_put)
    st = rig.run(pcm)
    assert st["status"] == "completed"
    assert seen["body"] and all(
        isinstance(b, bytearray) and not any(b) for b in seen["body"]
    )
    assert seen["seg"] and all(not np.any(s) for s in seen["seg"])
    assert seen["wav"] and all(
        isinstance(w, bytearray) and not any(w) for w in seen["wav"]
    )
    assert rig.registry.bytes_in_use == 0


# ------------------------------------------------------------------ H3 global byte budget
def test_global_audio_budget_refuses_growth_and_is_released_on_cancel():
    cap = 3 * 32000  # 3 batches of 1 s
    rig = Rig(IncrementalConfig(max_total_audio_bytes=cap))
    pcm = make_pcm(10.0)
    batches = split(pcm, 16000)
    a, b = rig.open(), rig.open()
    assert rig.put(a, 0, *batches[0]).status_code == 200  # 32000 B
    assert rig.put(b, 0, *batches[0]).status_code == 200  # 64000 B
    assert rig.put(a, 1, *batches[1]).status_code == 200  # 96000 B == cap
    r = rig.put(b, 1, *batches[1])
    assert r.status_code == 429 and r.json()["error"]["code"] == "session_limit"
    assert rig.registry.bytes_in_use == cap
    assert rig.client.delete(f"/v1/sessions/{a}").status_code == 204
    assert rig.registry.bytes_in_use == 32000
    assert rig.put(b, 1, *batches[1]).status_code == 200  # the budget came back
    # refused batch left no trace: the same seq is accepted later with the same bytes
    assert rig.client.delete(f"/v1/sessions/{b}").status_code == 204
    assert rig.registry.bytes_in_use == 0


def test_budget_includes_the_canonical_wav_at_close_and_is_released_when_done():
    pcm = make_pcm(20.0)
    cap = len(pcm) + (len(pcm) + 44) - 1  # one byte short of buffer + wav
    rig = Rig(IncrementalConfig(max_total_audio_bytes=cap))
    sid = rig.open()
    for i, (o, d) in enumerate(split(pcm, 32000)):
        assert rig.put(sid, i, o, d).status_code == 200
    r = rig.close(sid, pcm, len(split(pcm, 32000)) - 1)
    assert r.status_code == 429 and r.json()["error"]["code"] == "session_limit"
    assert rig.registry.get(sid).status == "open"  # a refused close changes nothing
    rig2 = Rig(IncrementalConfig(max_total_audio_bytes=len(pcm) * 2 + 44))
    st = rig2.run(pcm)
    assert st["status"] == "completed" and rig2.registry.bytes_in_use == 0


# ------------------------------------------------------------------ H4 cancel while closing
def test_cancel_during_closing_frees_the_worker_at_the_next_chunk_boundary():
    rig = Rig()
    rig.asr.delay_s = 0.4
    pcm = make_pcm(70.0)  # ~11 chunks
    batches = split(pcm, 160000)
    sid = rig.open()
    # no early decoding: park the worker so every chunk is a miss at close time
    gate = threading.Event()
    rig.registry.submit_task(0, lambda: gate.wait(30))
    for i, (o, d) in enumerate(batches):
        assert rig.put(sid, i, o, d).status_code == 200
    assert rig.close(sid, pcm, len(batches) - 1).status_code == 202
    gate.set()
    deadline = time.time() + 20
    while (
        not rig.asr.log and time.time() < deadline
    ):  # finalize has started its first real decode
        time.sleep(0.01)
    n_cancel = len(rig.asr.log)
    assert rig.client.delete(f"/v1/sessions/{sid}").status_code == 204
    freed = threading.Event()
    t0 = time.time()
    rig.registry.submit_task(0, freed.set)
    assert freed.wait(5), "the worker is still busy with a cancelled analysis"
    # at most the decode in flight plus one more probe boundary, never the remaining ~10 chunks (4 s)
    assert len(rig.asr.log) - n_cancel <= 2
    assert time.time() - t0 < 2.0
    assert rig.client.get(f"/v1/sessions/{sid}").status_code == 404
    assert rig.registry.bytes_in_use == 0


# ------------------------------------------------------------------ H5 scheduling bounds
def _order_runner(promote_after_s: float):
    rig = Rig()
    return rig, JobRunner(rig.analyzer, promote_after_s=promote_after_s)


def test_a_close_never_waits_behind_queued_chunks_of_other_sessions():
    _rig, runner = _order_runner(60)
    order, gate = [], threading.Event()
    runner.submit_task(2, lambda: gate.wait(10))  # the one task running now
    time.sleep(0.05)
    for k in range(6):
        runner.submit_task(2, lambda k=k: order.append(f"chunk{k}"))
    runner.submit_task(0, lambda: order.append("close"))
    gate.set()
    deadline = time.time() + 5
    while len(order) < 7 and time.time() < deadline:
        time.sleep(0.01)
    assert order[0] == "close" and len(order) == 7


def test_a_queued_job_is_promoted_ahead_of_chunks_after_the_documented_wait():
    rig, runner = _order_runner(0.2)
    order, gate = [], threading.Event()
    real = rig.analyzer.analyze_bytes
    rig.analyzer.analyze_bytes = lambda *a, **k: (order.append("job"), real(*a, **k))[1]
    runner.submit_task(2, lambda: gate.wait(10))
    time.sleep(0.05)
    for k in range(5):
        runner.submit_task(2, lambda k=k: order.append(f"chunk{k}"))
    job = runner.submit(canonical_wav(make_pcm(5.0)), AnalysisOptions())
    time.sleep(0.5)  # longer than the promotion delay while the worker is still busy
    gate.set()
    deadline = time.time() + 10
    while (
        runner.public(job["job_id"])["status"] in ("queued", "running")
        and time.time() < deadline
    ):
        time.sleep(0.02)
    assert (
        runner.public(job["job_id"])["status"] == "failed"
    )  # 5 s: audio_too_short, as offline
    assert order[0] == "job"  # ran before every chunk queued ahead of it
    # one run only: the original low-priority entry finds the job already done
    time.sleep(0.2)
    assert order.count("job") == 1


# ------------------------------------------------------------------ M3 cache key carries the profile
def test_cache_namespace_includes_the_requested_profile_name():
    rig = Rig(profiles=("p0", "p1_disfluent_prompt"))
    real = rig.analyzer.asr_for(AnalysisOptions(asr_profile="p0"))
    a = session_namespace(AnalysisOptions(asr_profile="p0"), real)
    b = session_namespace(AnalysisOptions(asr_profile="p1_disfluent_prompt"), real)
    assert a != b and a.startswith("p0|") and b.startswith("p1_disfluent_prompt|")
    cache, seg = ChunkCache(), np.ones(16000, dtype=np.float32)
    cache.put(segment_key(a, 0, seg), [], [])
    assert segment_key(b, 0, seg) not in cache and segment_key(a, 0, seg) in cache


# ------------------------------------------------------------------ H8 loop re-decode (fake ASR; logic only)
class LoopyASR(HashASR):
    """Emulates the ``*_noloop`` ladder: chunks flagged as looped record a re-decode; rung 1 is a pure function of the
    audio (reproducible), rung 2 draws from a counter (the real one samples at T=0.4 and is NOT bit-reproducible)."""

    def __init__(self, rung: int):
        super().__init__(with_redecode=False)
        self.rung = rung
        self.counter = 0

    def transcribe(self, audio):
        words = super().transcribe(audio)
        if not words or words[0].text[1] not in "01234567":
            return words
        self.redecode_events.append(
            {
                "attempt": self.rung,
                "reproducible": self.rung == 1,
                "words_before": len(words),
            }
        )
        if self.rung == 2:
            self.counter += 1
            last = words[-1]  # the tail of a chunk is always inside its kept region
            words[-1] = type(last)(
                f"r{self.counter}", f"r{self.counter}", last.start, last.end, last.prob, ""
            )
        return words


@pytest.mark.parametrize("seed", [3, 5, 11])
def test_loop_redecode_rung1_is_equivalent_and_rung2_is_flagged_not_reproducible(seed):
    pcm = make_pcm(45.0, seed=seed)
    for rung in (1, 2):
        asr = LoopyASR(rung)
        real_ns = "x"
        cache = ChunkCache()
        from mrcd.service.incremental import plan
        from mrcd.service.incremental.cached_asr import decode_with

        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        offline = asr.transcribe_chunked(audio, chunk_s=8.0)
        off_events = list(asr.redecode_events)
        assert off_events, "the fixture must exercise the re-decode ladder"
        for _k, start in reversed(plan.complete_chunks(len(audio))):
            seg = audio[start : start + plan.CHUNK_SAMPLES]
            cache.put(segment_key(real_ns, start, seg), *decode_with(asr, seg, start))
        cached = CachedASR(asr, cache, real_ns)
        incr = cached.transcribe_chunked(audio, chunk_s=8.0)
        key = lambda ws: [(w.text, w.start, w.end, w.prob) for w in ws]  # noqa: E731
        if rung == 1:
            assert key(incr) == key(offline)  # reproducible rung: bit-identical
            assert all(e["reproducible"] for e in off_events)
        else:
            # sampled rung: the two decodes legitimately differ; the service flags it, which is the documented limit
            assert all(e["reproducible"] is False for e in off_events)
            assert key(incr) != key(offline)


# ------------------------------------------------------------------ scheduling classes used by the session itself
def test_session_tasks_use_the_documented_priorities():
    rig = Rig()
    calls: list[int] = []
    original = rig.registry.submit_task
    rig.registry.submit_task = lambda prio, fn: (calls.append(prio), original(prio, fn))[1]
    pcm = make_pcm(30.0)
    st = rig.run(pcm)
    assert st["status"] == "completed"
    assert calls.count(0) == 1  # the finalize
    assert calls.count(2) >= 2 and set(calls) <= {0, 1, 2}  # open-session chunks
    assert calls.index(0) > max(i for i, p in enumerate(calls) if p == 2)  # finalize came last, ran first


# ------------------------------------------------------------------ cache key carries the audio content
def test_the_cache_never_answers_for_different_audio_at_the_same_position_and_length():
    asr = HashASR()
    a = np.full(16000 * 8, 0.1, dtype=np.float32)
    b = np.full(16000 * 8, 0.2, dtype=np.float32)
    assert segment_key("ns", 0, a) != segment_key("ns", 0, b)
    cache = ChunkCache()
    cache.put(segment_key("ns", 0, a), asr.transcribe(a), [])
    cached = CachedASR(asr, cache, "ns")
    asr.log.clear()
    assert [w.text for w in cached.transcribe(b)] == [w.text for w in asr.transcribe(b)]
    assert cached.decode_calls == 1  # b was a miss and was really decoded


# ------------------------------------------------------------------ H8 guard: non-reproducible profiles are refused
@pytest.mark.parametrize("profile", ["p0_noloop", "p1_noloop"])
def test_profiles_with_the_sampled_loop_redecode_cannot_open_an_incremental_session(profile):
    rig = Rig(profiles=("p0", profile))
    r = rig.client.post(f"/v1/sessions?asr_profile={profile}")
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "profile_not_supported_incremental"
    assert rig.client.post("/v1/sessions?asr_profile=p0").status_code == 201
