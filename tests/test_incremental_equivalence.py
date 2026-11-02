"""B incremental vs offline: exact equivalence cases 1-27 (fake content-sensitive ASR) of DESIGN_B_INCREMENTAL section 3.

"Equal" means ``==`` on the whole ``mrcd.analysis/v1`` document serialised with sorted keys; the only excluded field is
``generated_at``. Real-Whisper equivalence (same cases, real models) is in ``scripts/s8_incremental/equivalence_real.py``.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from mrcd.constants import SAMPLE_RATE
from mrcd.extractors.linguistic import VerbatimASR
from mrcd.service import AnalysisOptions
from mrcd.service.incremental import plan
from mrcd.service.incremental.cache import ChunkCache
from mrcd.service.incremental.cached_asr import CachedASR
from mrcd.service.incremental.session import IncrementalConfig, canonical_wav
from tests.incremental_rig import HashASR, Rig, make_pcm, split, strip_volatile

BIG = IncrementalConfig(max_batch_bytes=64 << 20)


pytestmark = pytest.mark.timeout(180)


def assert_equal_to_offline(rig: Rig, pcm: bytes, st: dict, **opts):
    assert st["status"] == "completed", st
    off = rig.offline(pcm, **opts)
    assert strip_volatile(st["result"]) == strip_volatile(off)


# ------------------------------------------------------------------ cases 1-8, 19 (p0)
DURATIONS = [13.0, 13.6, 14.0, 30.0, 48.0, 31.0, 32.0, 37.0, 36.999, 45.466, 60.0]


@pytest.mark.parametrize("seconds", DURATIONS)
def test_cases_1_to_8_exact_document_and_each_chunk_decoded_once(seconds):
    rig = Rig()
    pcm = make_pcm(seconds)
    st = rig.run(pcm, query="?include_text=true", batches=split(pcm, 2560))
    assert_equal_to_offline(rig, pcm, st, include_text=True)
    n = len(pcm) // 2
    # every chunk the offline decodes was decoded exactly once, and the complete ones were decoded before the close
    assert len(rig.asr.log) == len(plan.final_plan(n))
    sess = rig.registry.get(st["sid"])
    complete = len(plan.complete_chunks(n))
    assert sess.cache_stats["hits"] >= min(complete, len(plan.final_plan(n)))


def test_case_8_long_599_9_seconds():
    rig = Rig()
    pcm = make_pcm(599.9)
    st = rig.run(pcm, query="?include_text=true", batches=split(pcm, 32000))
    assert_equal_to_offline(rig, pcm, st, include_text=True)


@pytest.mark.parametrize("seconds", [13.6, 14.0, 36.999, 37.0, 45.466])
def test_decode_calls_equal_the_number_of_fragments(seconds):
    rig = Rig()
    pcm = make_pcm(seconds)
    rig.run(pcm, batches=split(pcm, 2560))
    assert len(rig.asr.log) == len(
        plan.final_plan(len(pcm) // 2)
    )  # nothing decoded twice, nothing skipped


def test_partial_boundary_36_999_discards_the_last_fragment_like_the_offline():
    n = int(36.999 * SAMPLE_RATE)
    assert [s for _k, s, _l in plan.final_plan(n)][-1] == 5 * 96000 and len(
        plan.final_plan(n)
    ) == 6
    assert len(plan.final_plan(int(37.0 * SAMPLE_RATE))) == 7


# ------------------------------------------------------------------ case 9 and 10 (engine level)
@pytest.mark.parametrize("seconds", [7.5, 8.0, 8.001, 13.0, 650.0])
def test_cases_9_10_cached_asr_equals_transcribe_chunked_at_engine_level(seconds):
    asr = HashASR()
    audio = (
        np.frombuffer(make_pcm(seconds, seed=3), dtype="<i2").astype(np.float32)
        / 32768.0
    )
    direct = asr.transcribe_chunked(audio, chunk_s=8.0)
    cache = ChunkCache()
    from mrcd.service.incremental.cache import segment_key
    from mrcd.service.incremental.cached_asr import asr_namespace, decode_with

    ns = asr_namespace(asr)
    for _k, start in plan.complete_chunks(len(audio)):
        seg = audio[start : start + plan.CHUNK_SAMPLES]
        cache.put(segment_key(ns, start, seg), *decode_with(asr, seg, start))
    cached = CachedASR(asr, cache, ns).transcribe_chunked(audio, chunk_s=8.0)
    assert [(w.text, repr(w.start), repr(w.end), repr(w.prob)) for w in cached] == [
        (w.text, repr(w.start), repr(w.end), repr(w.prob)) for w in direct
    ]


# ------------------------------------------------------------------ case 11: partitions
@pytest.mark.parametrize("mode", ["160ms", "2s", "random", "one_block"])
def test_case_11_partitions(mode):
    rig = Rig(BIG)
    pcm = make_pcm(45.466)
    batches = {
        "160ms": split(pcm, 2560),
        "2s": split(pcm, 32000),
        "random": split(pcm, rng_seed=20261101),
        "one_block": split(pcm, len(pcm)),
    }[mode]
    st = rig.run(pcm, batches=batches, query="?include_text=true")
    assert_equal_to_offline(rig, pcm, st, include_text=True)


# ------------------------------------------------------------------ cases 12-13: order and duplicates
def test_case_12_out_of_order_batches():
    rig = Rig()
    pcm = make_pcm(45.466)
    batches = split(pcm, 32000)
    order = list(range(len(batches)))
    import random

    random.Random(5).shuffle(order)
    # keep within the reorder window (64 s): every batch is within the first 45 s here
    st = rig.run(pcm, batches=batches, order=order, query="?include_text=true")
    assert_equal_to_offline(rig, pcm, st, include_text=True)
    assert len(rig.asr.log) == len(plan.final_plan(len(pcm) // 2))


def test_case_13_duplicates_and_duplicate_close():
    rig = Rig()
    pcm = make_pcm(30.0)
    st = rig.run(pcm, batches=split(pcm, 32000), dup=True, query="?include_text=true")
    assert_equal_to_offline(rig, pcm, st, include_text=True)
    assert len(rig.asr.log) == len(plan.final_plan(len(pcm) // 2))


# ------------------------------------------------------------------ cases 14-16: conflict, gap, retries
def test_case_14_same_seq_other_bytes_is_a_conflict_and_does_not_corrupt():
    rig = Rig()
    pcm = make_pcm(20.0)
    batches = split(pcm, 32000)
    sid = rig.open("?include_text=true")
    assert rig.put(sid, 0, *batches[0]).status_code == 200
    bad = rig.put(sid, 0, batches[0][0], b"\x01\x00" * 16000)
    assert bad.status_code == 409 and bad.json()["error"]["code"] == "sequence_conflict"
    for i, (off, data) in enumerate(batches[1:], start=1):
        assert rig.put(sid, i, off, data).status_code == 200
    assert rig.close(sid, pcm, len(batches) - 1).status_code == 202
    st = rig.wait(sid)
    assert_equal_to_offline(rig, pcm, st, include_text=True)


def test_case_15_gap_then_resend_and_unresolved_gap():
    rig = Rig()
    pcm = make_pcm(20.0)
    batches = split(pcm, 32000)
    sid = rig.open("?include_text=true")
    for i, (off, data) in enumerate(batches):
        if i != 3:
            assert rig.put(sid, i, off, data).status_code == 200
    r = rig.close(sid, pcm, len(batches) - 1)
    assert r.status_code == 409
    err = r.json()["error"]
    assert err["code"] == "audio_gap" and err["details"]["missing_seqs"] == [3]
    assert err["details"]["first_missing_offset"] == batches[3][0]
    assert rig.put(sid, 3, *batches[3]).status_code == 200
    assert rig.close(sid, pcm, len(batches) - 1).status_code == 202
    assert_equal_to_offline(rig, pcm, rig.wait(sid), include_text=True)


def test_case_16_lost_responses_are_retried_idempotently_and_the_pcm_matches():
    rig = Rig()
    pcm = make_pcm(40.0)
    batches = split(pcm, 32000)
    sid = rig.open("?include_text=true")
    import random

    rnd = random.Random(9)
    for i, (off, data) in enumerate(batches):
        for _ in range(
            rnd.choice([1, 2, 3])
        ):  # the response is "lost": the same (seq, offset, sha) is resent
            assert rig.put(sid, i, off, data).status_code == 200
    sess = rig.registry.get(sid)
    assert (
        hashlib.sha256(bytes(sess.buf)).hexdigest() == hashlib.sha256(pcm).hexdigest()
    )
    assert rig.close(sid, pcm, len(batches) - 1).status_code == 202
    assert_equal_to_offline(rig, pcm, rig.wait(sid), include_text=True)


def test_close_with_wrong_total_or_sha_is_rejected():
    rig = Rig()
    pcm = make_pcm(14.0)
    batches = split(pcm, 32000)
    sid = rig.open()
    for i, (off, data) in enumerate(batches):
        rig.put(sid, i, off, data)
    r = rig.client.post(
        f"/v1/sessions/{sid}/close",
        json={
            "total_samples": len(pcm) // 2,
            "wav_sha256": "0" * 64,
            "last_seq": len(batches) - 1,
        },
    )
    assert r.status_code == 409 and r.json()["error"]["code"] == "close_mismatch"
    r = rig.client.post(
        f"/v1/sessions/{sid}/close",
        json={
            "total_samples": len(pcm) // 2 + 1,
            "wav_sha256": "0" * 64,
            "last_seq": len(batches) - 1,
        },
    )
    assert r.status_code == 409 and r.json()["error"]["code"] == "close_mismatch"


# ------------------------------------------------------------------ case 17: restart
def test_case_17_service_restart_means_session_not_found_and_offline_fallback_equals():
    rig = Rig()
    pcm = make_pcm(20.0)
    batches = split(pcm, 32000)
    sid = rig.open()
    rig.put(sid, 0, *batches[0])
    rig2 = Rig()  # "restarted" process: no sessions
    r = rig2.put(sid, 1, *batches[1])
    assert r.status_code == 404 and r.json()["error"]["code"] == "session_not_found"
    assert rig2.close(sid, pcm, 1).status_code == 404
    # fallback: the offline path with the full buffer
    job = rig2.client.post(
        "/v1/jobs?include_text=true", content=canonical_wav(pcm)
    ).json()
    import time

    for _ in range(500):
        done = rig2.client.get(f"/v1/jobs/{job['job_id']}").json()
        if done["status"] in ("completed", "failed"):
            break
        time.sleep(0.01)
    assert strip_volatile(done["result"]) == strip_volatile(
        rig2.offline(pcm, include_text=True)
    )


# ------------------------------------------------------------------ case 18: cancellation
def test_case_18_cancel_zeroes_the_audio_and_later_use_is_not_found():
    rig = Rig()
    pcm = make_pcm(40.0)
    batches = split(pcm, 32000)
    sid = rig.open()
    for i, (off, data) in enumerate(batches[:5]):
        rig.put(sid, i, off, data)
    sess = rig.registry.get(sid)
    seen = {}

    class Spy(bytearray):
        def clear(self):
            seen["all_zero_before_clear"] = not any(self)
            super().clear()

    sess.buf = Spy(sess.buf)
    assert rig.client.delete(f"/v1/sessions/{sid}").status_code == 204
    assert rig.client.delete(f"/v1/sessions/{sid}").status_code == 204  # idempotent
    assert seen == {"all_zero_before_clear": True} and len(sess.cache) == 0
    assert rig.put(sid, 5, *batches[5]).status_code == 404
    assert rig.close(sid, pcm, 5).status_code == 404


# ------------------------------------------------------------------ case 19-20: profiles, redecode events
def test_case_19_p0_and_p1_both_equal():
    rig = Rig(profiles=("p0", "p1_disfluent_prompt"))
    pcm = make_pcm(45.466)
    for prof in ("p0", "p1_disfluent_prompt"):
        st = rig.run(pcm, query=f"?asr_profile={prof}&include_text=true")
        assert_equal_to_offline(rig, pcm, st, include_text=True, asr_profile=prof)


def test_case_20_redecode_events_and_their_offsets_are_equal():
    rig = Rig(with_redecode=True)
    pcm = make_pcm(45.466)
    rig.engine.events_log.clear()
    off = rig.offline(pcm, include_text=True)
    st = rig.run(pcm, query="?include_text=true")
    assert strip_volatile(st["result"]) == strip_volatile(off)
    offline_events, inc_events = rig.engine.events_log[0], rig.engine.events_log[-1]
    assert offline_events and offline_events == inc_events
    assert [e["chunk_offset_s"] for e in inc_events] == sorted(
        e["chunk_offset_s"] for e in inc_events
    )


# ------------------------------------------------------------------ case 21: warm-up
def test_case_21_warmup_does_not_change_the_document():
    from mrcd.service.incremental.warmup import synthetic_signal, warm_up

    rig = Rig()
    pcm = make_pcm(30.0)
    before = rig.run(pcm, query="?include_text=true")
    warm_up(rig.analyzer, [AnalysisOptions()])
    after = rig.run(pcm, query="?include_text=true")
    assert strip_volatile(before["result"]) == strip_volatile(after["result"])
    assert (
        np.array_equal(synthetic_signal(), synthetic_signal())
        and len(synthetic_signal()) == 13 * SAMPLE_RATE
    )


# ------------------------------------------------------------------ case 22: interleaving
def test_case_22_three_interleaved_sessions_and_a_job():
    rig = Rig()
    pcms = [make_pcm(s, seed=i) for i, s in enumerate((30.0, 45.466, 22.0))]
    allb = [split(p, 32000) for p in pcms]
    sids = [rig.open("?include_text=true") for _ in pcms]
    jobbed = None
    for step in range(max(len(b) for b in allb)):
        for j, b in enumerate(allb):
            if step < len(b):
                assert rig.put(sids[j], step, *b[step]).status_code == 200
        if step == 1:
            jobbed = rig.client.post(
                "/v1/jobs?include_text=true", content=canonical_wav(pcms[0])
            ).json()
    for j, p in enumerate(pcms):
        assert rig.close(sids[j], p, len(allb[j]) - 1).status_code == 202
    for j, p in enumerate(pcms):
        assert_equal_to_offline(rig, p, rig.wait(sids[j]), include_text=True)
    assert jobbed["status"] in ("queued", "running", "completed")


# ------------------------------------------------------------------ case 23: options
@pytest.mark.parametrize(
    "query,opts",
    [
        ("?include_text=true", dict(include_text=True)),
        ("?include_out_of_scope=true", dict(output_scope="all")),
        ("?decision=true", dict(decision=True)),
        (
            "?include_text=true&decision=true&include_out_of_scope=true",
            dict(include_text=True, decision=True, output_scope="all"),
        ),
    ],
)
def test_case_23_options(query, opts):
    rig = Rig()
    pcm = make_pcm(30.0)
    st = rig.run(pcm, query=query)
    assert_equal_to_offline(rig, pcm, st, **opts)


def test_case_23_head_must_be_preloaded():
    rig = Rig()
    r = rig.client.post("/v1/sessions?head=vc")
    assert (
        r.status_code == 422
        and r.json()["error"]["code"] == "profile_not_supported_incremental"
    )
    rig.analyzer.preload(AnalysisOptions(head="vc"))
    pcm = make_pcm(20.0)
    st = rig.run(pcm, query="?head=vc")
    assert_equal_to_offline(rig, pcm, st, head="vc")


# ------------------------------------------------------------------ case 24: errors
def test_case_24_too_short_has_the_same_error_code_as_offline():
    rig = Rig()
    pcm = make_pcm(9.0)
    st = rig.run(pcm)
    off = rig.offline(pcm)
    assert (
        st["status"] == "failed"
        and st["result"]["error"]["code"] == off["error"]["code"] == "audio_too_short"
    )
    assert strip_volatile(st["result"]) == strip_volatile(off)


def test_case_24_too_long_is_rejected_with_audio_too_long():
    rig = Rig(IncrementalConfig(max_session_seconds=20.0))
    sid = rig.open()
    pcm = make_pcm(30.0)
    batches = split(pcm, 32000)
    codes = [
        rig.put(sid, i, *b).json().get("error", {}).get("code")
        for i, b in enumerate(batches)
    ]
    assert "audio_too_long" in codes


# ------------------------------------------------------------------ cases 25-26: mutants
def test_case_25_shifted_plan_is_only_a_cache_miss(monkeypatch):
    real = plan.complete_chunks
    monkeypatch.setattr(
        plan,
        "complete_chunks",
        lambda n: [(k, s + 1) for k, s in real(n) if s + 1 + plan.CHUNK_SAMPLES <= n],
    )
    rig = Rig()
    pcm = make_pcm(45.466)
    st = rig.run(pcm, query="?include_text=true")
    assert_equal_to_offline(rig, pcm, st, include_text=True)
    assert rig.registry.get(st["sid"]).cache_stats["misses"] > 0


def test_case_26_a_merge_mutant_is_detected(monkeypatch):
    def evil_get(self, key):
        with self._lock:
            items = list(self._items.values())
            self.hits += 1
            if not items:
                return None
            words, events = items[0]  # always the words of ANOTHER fragment
            from dataclasses import replace

            return [replace(w) for w in words], [dict(e) for e in events]

    monkeypatch.setattr(ChunkCache, "get", evil_get)
    rig = Rig()
    pcm = make_pcm(45.466)
    st = rig.run(pcm, query="?include_text=true")
    off = rig.offline(pcm, include_text=True)
    assert st["status"] == "completed" and strip_volatile(
        st["result"]
    ) != strip_volatile(off)


# ------------------------------------------------------------------ case 27: process state
def test_case_27_startup_order_is_preload_then_warmup():
    from mrcd.service import server

    calls = []
    rig = Rig(profiles=())
    rig.analyzer.preload = lambda o: calls.append(("preload", o.asr_profile))
    orig = server.warm_up
    server.warm_up = lambda a, p: calls.append(("warmup", [o.asr_profile for o in p]))
    try:
        server.prepare(
            rig.app,
            rig.analyzer,
            (
                AnalysisOptions(asr_profile="p0"),
                AnalysisOptions(asr_profile="p1_disfluent_prompt"),
            ),
            True,
        )
    finally:
        server.warm_up = orig
    assert calls == [
        ("preload", "p0"),
        ("preload", "p1_disfluent_prompt"),
        ("warmup", ["p0", "p1_disfluent_prompt"]),
    ]
    assert rig.client.get("/v1/health").json()["warmed_up"] is True


def test_service_refuses_incremental_without_cpu_preload_and_warmup():
    from mrcd.service.cli import main

    with pytest.raises(SystemExit):
        main(["serve", "--incremental", "--port", "1"])
    with pytest.raises(SystemExit):
        main(
            [
                "serve",
                "--incremental",
                "--device",
                "cpu",
                "--preload",
                "p0",
                "--port",
                "1",
            ]
        )


def test_incremental_disabled_by_default():
    rig = Rig(incremental=False)
    r = rig.client.post("/v1/sessions")
    assert r.status_code == 404 and r.json()["error"]["code"] == "incremental_disabled"
    assert rig.client.get("/v1/health").json()["incremental"] is False


def test_unknown_or_gpu_profile_is_not_supported():
    rig = Rig()
    r = rig.client.post("/v1/sessions?asr_profile=p0_cuda")
    assert (
        r.status_code == 422
        and r.json()["error"]["code"] == "profile_not_supported_incremental"
    )


def test_session_limit():
    rig = Rig(IncrementalConfig(max_sessions=2))
    rig.open()
    rig.open()
    r = rig.client.post("/v1/sessions")
    assert r.status_code == 429 and r.json()["error"]["code"] == "session_limit"


def test_plan_matches_the_offline_chunk_loop():
    # property: for many lengths, final_plan == the (start, length) pairs transcribe_chunked decodes
    class Rec(VerbatimASR):
        def __init__(self):
            self.segs = []

        def transcribe(self, audio):
            self.segs.append(len(audio))
            return []

    for n in (
        13 * SAMPLE_RATE,
        128000,
        128001,
        224000,
        224001,
        592000,
        592001,
        591999,
        727456,
        9_599_000,
    ):
        rec = Rec()
        rec.transcribe_chunked(np.zeros(n, np.float32), chunk_s=8.0)
        assert rec.segs == [length for _k, _s, length in plan.final_plan(n)], n
