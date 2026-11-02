"""B incremental: privacy tests P1-P6 and progress-message tests M1-M3 (DESIGN_B_INCREMENTAL section 3)."""

from __future__ import annotations

import pytest

import builtins
import json
import logging
import os
import re
import tempfile
import threading
import time

from mrcd.service.contract import SESSION_FORBIDDEN_KEYS, validate_session_document
from mrcd.service.incremental.session import IncrementalConfig
from tests.incremental_rig import Rig, make_pcm, split


pytestmark = pytest.mark.timeout(120)


class Spy(bytearray):
    """Records whether the bytes were already zero when the buffer was emptied."""

    def __init__(self, data=b""):
        super().__init__(data)
        self.zero_before_clear = None

    def clear(self):
        self.zero_before_clear = not any(self)
        super().clear()


def spy_on(sess):
    sess.buf = Spy(sess.buf)
    return sess.buf


def parse_sse(text: str) -> list[dict]:
    out = []
    for block in text.strip().split("\n\n"):
        fields = dict(
            line.split(": ", 1)
            for line in block.splitlines()
            if ": " in line and not line.startswith(":")
        )
        if "event" in fields:
            out.append(
                {
                    "id": int(fields["id"]),
                    "type": fields["event"],
                    "data": json.loads(fields["data"]),
                }
            )
    return out


# ------------------------------------------------------------------ P1 no files, P2 no content in logs
def test_p1_p2_no_files_written_and_no_content_in_logs(monkeypatch, caplog, capfd):
    rig = Rig()
    writes = []
    real_open, real_os_open = builtins.open, os.open

    def guarded_open(file, mode="r", *a, **k):
        if any(c in str(mode) for c in "wax+"):
            writes.append(("open", str(file)))
        return real_open(file, mode, *a, **k)

    def guarded_os_open(path, flags, *a, **k):
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT):
            writes.append(("os.open", str(path)))
        return real_os_open(path, flags, *a, **k)

    def no_tempfile(*a, **k):
        writes.append(("tempfile", ""))
        raise AssertionError("tempfile used")

    caplog.set_level(logging.DEBUG)
    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(os, "open", guarded_os_open)
    for name in ("NamedTemporaryFile", "TemporaryFile", "mkstemp", "mkdtemp"):
        monkeypatch.setattr(tempfile, name, no_tempfile)
    words = []
    for i in range(5):
        pcm = make_pcm(20.0, seed=i)
        st = rig.run(pcm, query="?include_text=true")
        assert st["status"] == "completed"
        words += (
            st["result"]["transcript"]["text"].split()
            if isinstance(st["result"]["transcript"], dict)
            else []
        )
    monkeypatch.undo()
    assert writes == []
    logged = caplog.text + "".join(capfd.readouterr())
    for needle in ("RIFF", "WAVE", "pcm"):
        assert needle not in logged
    assert not any(w in logged for w in words[:50])


# ------------------------------------------------------------------ P3 discard in every ending
def test_p3_discard_after_completed_failed_cancelled_expired_and_shutdown():
    rig = Rig(IncrementalConfig(idle_timeout_s=0.25))
    # completed
    pcm = make_pcm(20.0)
    sid = rig.open()
    sess = rig.registry.get(sid)
    buf = spy_on(sess)
    for i, (o, d) in enumerate(split(pcm, 32000)):
        rig.put(sid, i, o, d)
    rig.close(sid, pcm, len(split(pcm, 32000)) - 1)
    assert rig.wait(sid)["status"] == "completed"
    assert buf.zero_before_clear is True and len(buf) == 0 and len(sess.cache) == 0
    # failed (too short)
    short = make_pcm(9.0)
    sid = rig.open()
    sess = rig.registry.get(sid)
    buf = spy_on(sess)
    for i, (o, d) in enumerate(split(short, 32000)):
        rig.put(sid, i, o, d)
    rig.close(sid, short, len(split(short, 32000)) - 1)
    assert rig.wait(sid)["status"] == "failed"
    assert buf.zero_before_clear is True and len(buf) == 0 and len(sess.cache) == 0
    # cancelled
    sid = rig.open()
    sess = rig.registry.get(sid)
    buf = spy_on(sess)
    rig.put(sid, 0, *split(pcm, 32000)[0])
    rig.client.delete(f"/v1/sessions/{sid}")
    assert buf.zero_before_clear is True and len(buf) == 0
    # expired
    sid = rig.open()
    sess = rig.registry.get(sid)
    buf = spy_on(sess)
    rig.put(sid, 0, *split(pcm, 32000)[0])
    time.sleep(0.9)
    assert rig.client.get(f"/v1/sessions/{sid}").json()["status"] == "expired"
    assert buf.zero_before_clear is True and len(buf) == 0 and len(sess.cache) == 0
    # shutdown
    rig2 = Rig()
    sid = rig2.open()
    sess = rig2.registry.get(sid)
    buf = spy_on(sess)
    rig2.put(sid, 0, *split(pcm, 32000)[0])
    rig2.registry.shutdown()
    assert buf.zero_before_clear is True and len(buf) == 0


# ------------------------------------------------------------------ P4 events carry counts only
def test_p4_events_and_state_documents_carry_no_content():
    rig = Rig()
    pcm = make_pcm(30.0)
    sid = rig.open("?include_text=true")
    batches = split(pcm, 32000)
    out = {}
    t = threading.Thread(
        target=lambda: out.update(
            text=rig.client.get(f"/v1/sessions/{sid}/events").text
        )
    )
    t.start()
    time.sleep(0.2)
    for i, (o, d) in enumerate(batches):
        rig.put(sid, i, o, d)
        assert (
            validate_session_document(rig.client.get(f"/v1/sessions/{sid}").json())
            == []
        )
    rig.close(sid, pcm, len(batches) - 1)
    t.join(30)
    events = parse_sse(out["text"])
    assert events and events[-1]["type"] == "completed"
    allowed = {
        "status",
        "received_s",
        "chunks_decoded",
        "chunks_complete",
        "stage",
        "session_id",
        "result_available",
    }
    # EVERY event, the terminal one included, is counts/state only: no document, no text, no words.
    for e in events:
        assert not set(SESSION_FORBIDDEN_KEYS) & set(e["data"])
        assert "result" not in e["data"]
        assert set(e["data"]) <= allowed
    assert events[-1]["data"]["status"] == "completed"
    assert events[-1]["data"]["result_available"] is True
    # the document (with its text, include_text=true) is only reachable through the authenticated endpoint
    assert not re.search(r"w[0-9a-f]{6}", out["text"])
    assert rig.client.get(f"/v1/sessions/{sid}").json()["result"]["status"] == "completed"


# ------------------------------------------------------------------ P6 expiry
def test_p6_idle_session_is_discarded_after_the_timeout():
    rig = Rig(IncrementalConfig(idle_timeout_s=0.3))
    pcm = make_pcm(20.0)
    sid = rig.open()
    rig.put(sid, 0, *split(pcm, 32000)[0])
    assert rig.client.get(f"/v1/sessions/{sid}").json()["status"] == "open"
    time.sleep(0.3 + 1.0)
    assert rig.client.get(f"/v1/sessions/{sid}").json()["status"] == "expired"
    r = rig.put(sid, 1, *split(pcm, 32000)[1])
    assert r.status_code == 410 and r.json()["error"]["code"] == "session_expired"


# ------------------------------------------------------------------ M1-M3 progress
def test_m1_event_order_and_single_completed():
    rig = Rig()
    pcm = make_pcm(45.466)
    sid = rig.open()
    batches = split(pcm, 32000)
    out = {}
    t = threading.Thread(
        target=lambda: out.update(
            text=rig.client.get(f"/v1/sessions/{sid}/events").text
        )
    )
    t.start()
    time.sleep(0.2)
    for i, (o, d) in enumerate(batches):
        rig.put(sid, i, o, d)
        time.sleep(0.02)
    rig.close(sid, pcm, len(batches) - 1)
    t.join(30)
    ev = parse_sse(out["text"])
    kinds = [e["type"] for e in ev]
    assert (
        kinds[0] == "state"
        and kinds.count("completed") == 1
        and kinds[-1] == "completed"
    )
    assert kinds.index("finalizing") < kinds.index("completed")
    assert all(
        k in ("asr_progress", "state") for k in kinds[1 : kinds.index("finalizing")]
    )
    decoded = [e["data"]["chunks_decoded"] for e in ev if e["type"] == "asr_progress"]
    assert decoded == sorted(decoded) and decoded[-1] >= 5
    assert [e["id"] for e in ev] == sorted({e["id"] for e in ev})


def test_m2_late_subscription_gets_a_counts_only_completed_event():
    rig = Rig()
    pcm = make_pcm(20.0)
    st = rig.run(pcm, query="?include_text=true")
    ev = parse_sse(rig.client.get(f"/v1/sessions/{st['sid']}/events").text)
    assert [e["type"] for e in ev] == ["completed"]
    assert ev[0]["data"]["status"] == "completed" and "result" not in ev[0]["data"]
    assert st["result"]["status"] == "completed"  # the document comes from GET /v1/sessions/{id}


def test_m3_last_event_id_resumes_without_loss_or_repetition():
    rig = Rig()
    pcm = make_pcm(45.466)
    st = rig.run(pcm, batches=split(pcm, 32000))
    sess = rig.registry.get(st["sid"])
    full = sess.events
    assert len(full) > 4
    cut = full[2]["id"]
    ev = parse_sse(
        rig.client.get(
            f"/v1/sessions/{st['sid']}/events", headers={"Last-Event-ID": str(cut)}
        ).text
    )
    assert [e["id"] for e in ev] == [e["id"] for e in full if e["id"] > cut]
    assert ev[-1]["type"] == "completed"
    # already saw everything: nothing is repeated
    last = full[-1]["id"]
    assert (
        parse_sse(
            rig.client.get(
                f"/v1/sessions/{st['sid']}/events", headers={"Last-Event-ID": str(last)}
            ).text
        )
        == []
    )


def test_openapi_exposes_the_session_endpoints():
    rig = Rig()
    paths = rig.client.get("/openapi.json").json()["paths"]
    assert "/v1/sessions" in paths and "/v1/sessions/{session_id}/events" in paths


def test_a_closing_session_is_never_expired_while_its_analysis_is_running():
    """The idle timeout applies to open sessions only: a long analysis must not discard its own result."""
    rig = Rig(IncrementalConfig(idle_timeout_s=0.3))
    rig.asr.delay_s = 0.5  # every real decode takes 0.5 s, so the close outlasts the idle timeout
    pcm = make_pcm(20.0)
    batches = split(pcm, 32000)
    sid = rig.open("?include_text=true")
    for i, (o, d) in enumerate(batches):
        rig.put(sid, i, o, d)
    rig.close(sid, pcm, len(batches) - 1)
    state = rig.wait(sid)
    assert state["status"] == "completed" and state["result"]["status"] == "completed"
