"""End-to-end application guarantees with fake detectors (real ffmpeg, SQLite, HTTP)."""

import hashlib
import json
import subprocess
import time
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from app import exports, media, worker
from app.api import create_app
from app.store import Store
from core.dataset import read_human_events

pytestmark = pytest.mark.skipif(
    subprocess.run(["ffmpeg", "-version"], capture_output=True).returncode, reason="ffmpeg no disponible"
)


def fake_mrcd(job):
    job.stage("detect")
    return {
        "events": [
            {
                "event_id": f"{job.run_id}-e0",
                "label": "filler_word",
                "start_ms": 2000,
                "end_ms": 2500,
                "decision": "event",
                "score": 0.9,
                "score_kind": "softmax_uncalibrated",
            },
            {
                "event_id": f"{job.run_id}-e1",
                "label": "neutral_pause",
                "start_ms": 6000,
                "end_ms": 7000,
                "decision": "event",
                "score": 0.6,
                "score_kind": "softmax_uncalibrated",
            },
        ],
        "words": [{"text": "eh", "start": 2.0, "end": 2.5}],
        "segments": {s["id"]: "succeeded" for s in job.recording["data"]["segments"]},
    }


def failing_rules(job):
    raise RuntimeError("regla rota")


SYSTEMS = {"mrcd": fake_mrcd, "rules": failing_rules}


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("MRCD_DATA", str(tmp_path))
    store = Store(f"sqlite:///{(tmp_path / 'app.db').as_posix()}")
    client = TestClient(create_app(store))
    tokens = {}
    for name, role in [
        ("admin", "admin"),
        ("ana", "annotator"),
        ("beto", "annotator"),
        ("carla", "adjudicator"),
        ("rita", "researcher"),
        ("uma", "user"),
    ]:
        tokens[name] = store.create(
            "user",
            {"name": name, "role": role, "token_sha256": hashlib.sha256(name.encode()).hexdigest()},
            status="active",
        )["id"]

    def call(who, method, path, **kw):
        return client.request(
            method, "/api" + path, headers={"Authorization": f"Bearer {who}", **kw.pop("headers", {})}, **kw
        )

    return store, call, tokens, tmp_path


def run_worker(store):
    while worker.work_once(store, SYSTEMS):
        pass


def upload(call, who, path, **params):
    r = call(who, "POST", "/recordings", params={"filename": path.name, **params}, content=path.read_bytes())
    assert r.status_code == 202, r.text
    return r.json()["id"]


def test_ingest_keeps_original_and_derives_16k_mono(env):
    store, call, _, tmp = env
    stereo = np.stack([np.sin(np.arange(44100 * 3) / 20), np.zeros(44100 * 3)], 1) * 0.3
    sf.write(tmp / "in.wav", stereo, 44100)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(tmp / "in.wav"), "-c:a", "libopus", str(tmp / "in.webm")], check=True
    )
    for name in ("in.wav", "in.webm"):
        rid = upload(call, "uma", tmp / name, speaker_id="spk1")
        run_worker(store)
        rec = store.get(rid)
        assert rec["status"] == "ready", rec["data"].get("error")
        info = rec["data"]["media"]
        assert info["analysis"]["sample_rate"] == 16000 and info["analysis"]["channels"] == 1
        assert abs(info["analysis"]["duration_ms"] - 3000) <= 30
        assert media.sha256(worker.media_dir(rec) / rec["data"]["original_name"]) == rec["data"]["sha256"]
    assert upload(call, "uma", tmp / "in.wav") == store.find("recording")[0]["id"]  # same file, no duplicate


def test_time_offsets_and_export_round_trip(tmp_path):
    segs = media.segments(65_000)
    assert [s["id"] for s in segs] == ["s000", "s001", "s002"] and segs[2]["core_end_ms"] == 65_000
    assert media.owner(segs, 29_900, 30_300) == "s001" and media.owner(segs, 0, 10) == "s000"
    events = [
        {"event_id": "a", "start_ms": 0, "end_ms": 250, "label": "filler_word"},
        {"event_id": "b", "start_ms": 29_990, "end_ms": 30_010, "label": "block"},
        {"event_id": "c", "start_ms": 64_900, "end_ms": 65_000, "label": "neutral_pause"},
    ]
    (tmp_path / "x.eaf").write_bytes(exports.eaf(events, "x.wav", "test"))
    back = read_human_events(tmp_path / "x.eaf")
    assert [(x["category"], x["start_ms"], x["end_ms"]) for x in back] == [
        (x["label"], x["start_ms"], x["end_ms"]) for x in events
    ]
    assert "29990,30010,block" in exports.events_csv(events)
    ET.fromstring(exports.eaf(events, "x.wav", "t"))


def test_full_path_blind_review_adjudication_snapshot_evaluation(env):
    store, call, users, tmp = env
    t = np.arange(16000 * 20) / 16000
    sf.write(tmp / "talk.wav", (0.2 * np.sin(2 * np.pi * 150 * t)).astype(np.float32), 16000)
    rid = upload(call, "rita", tmp / "talk.wav", speaker_id="spk_a")
    run_worker(store)

    runs = call("rita", "POST", "/analysis-runs", json={"recording_id": rid, "systems": ["mrcd", "rules"]}).json()
    again = call("rita", "POST", "/analysis-runs", json={"recording_id": rid, "systems": ["mrcd"]}).json()
    assert again["runs"][0]["id"] == runs["runs"][0]["id"]  # same identity -> no duplicate run
    run_worker(store)
    mrcd_run, rules_run = (call("rita", "GET", f"/analysis-runs/{r['id']}").json() for r in runs["runs"])
    assert mrcd_run["status"] == "succeeded" and rules_run["status"] == "failed"  # independent failures
    assert rules_run["data"]["error"]["stage"] == "load_audio" or rules_run["data"]["error"]["type"] == "RuntimeError"
    worker.execute(store, store.get(mrcd_run["id"]), SYSTEMS)  # re-execution replaces, never appends
    assert len(store.get(mrcd_run["id"])["data"]["result"]["events"]) == 2
    report = call("rita", "GET", f"/exports/{mrcd_run['id']}", params={"format": "md"}).text
    assert "Disfluencias: **1**" in report and "pausas (no disfluencia): **1**" in report

    tasks = {}
    for who in ("ana", "beto"):
        r = call(
            "rita",
            "POST",
            "/review/tasks",
            json={
                "recording_id": rid,
                "start_ms": 0,
                "end_ms": 10_000,
                "mode": "blind",
                "split": "dev",
                "assignee": users[who],
            },
        )
        tasks[who] = r.json()["id"]
    # blinding: no predictions, run access or exports for annotators
    view = call("ana", "GET", f"/review/tasks/{tasks['ana']}").json()
    assert view["suggestions"] == [] and "run_id" not in json.dumps(view["task"])
    assert call("ana", "GET", f"/analysis-runs/{mrcd_run['id']}").status_code == 403
    assert call("ana", "GET", f"/exports/{mrcd_run['id']}").status_code == 403
    assert call("ana", "GET", f"/review/tasks/{tasks['beto']}").status_code == 403
    audio = call("ana", "GET", f"/review/tasks/{tasks['ana']}/audio")
    assert audio.headers["x-media-offset-ms"] == "0" and len(audio.content) > 44

    for who, label in (("ana", "filler_word"), ("beto", "repetition")):
        ann = call(who, "GET", f"/review/tasks/{tasks[who]}").json()["annotation"]
        body = {"events": [{"start_ms": 2000, "end_ms": 2400, "label": label}], "client_op": "op1"}
        r = call(who, "PATCH", f"/annotations/{ann['id']}", json=body, headers={"If-Match": str(ann["rev"])})
        assert r.status_code == 200
        retry = call(who, "PATCH", f"/annotations/{ann['id']}", json=body, headers={"If-Match": str(ann["rev"])})
        assert retry.json()["rev"] == r.json()["rev"]  # idempotent retry
        stale = call(
            who,
            "PATCH",
            f"/annotations/{ann['id']}",
            json={**body, "client_op": "op2"},
            headers={"If-Match": str(ann["rev"])},
        )
        assert stale.status_code == 409  # a second tab cannot overwrite
        gaps = call(who, "POST", f"/review/tasks/{tasks[who]}/submit")
        assert gaps.status_code == 422 and gaps.json()["detail"]["unreviewed_gaps_ms"]
        call(
            who,
            "PATCH",
            f"/annotations/{ann['id']}",
            json={**body, "client_op": "op3", "coverage": [[0, 10_000]]},
            headers={"If-Match": str(r.json()["rev"])},
        )
        assert call(who, "POST", f"/review/tasks/{tasks[who]}/submit").json() == {"status": "submitted"}
        locked = call(who, "PATCH", f"/annotations/{ann['id']}", json=body, headers={"If-Match": "99"})
        assert locked.status_code == 409

    bad = call(
        "admin",
        "POST",
        "/adjudication-cases",
        json={"task_a": tasks["ana"], "task_b": tasks["beto"], "adjudicator": users["ana"]},
    )
    assert bad.status_code == 422
    cid = call(
        "admin",
        "POST",
        "/adjudication-cases",
        json={"task_a": tasks["ana"], "task_b": tasks["beto"], "adjudicator": users["carla"]},
    ).json()["id"]
    case = call("carla", "GET", f"/review/adjudication/{cid}").json()
    assert {e["label"] for s in "AB" for e in case["tracks"][s]["events"]} == {"filler_word", "repetition"}
    assert "ana" not in json.dumps(case) and call("carla", "GET", f"/analysis-runs/{mrcd_run['id']}").status_code == 403
    based = {s: case["tracks"][s]["rev"] for s in "AB"}
    r = call(
        "carla",
        "POST",
        "/adjudications",
        json={
            "case_id": cid,
            "based_on": based,
            "reasons": {"row0": "B"},
            "events": [{"start_ms": 2000, "end_ms": 2450, "label": "filler_word"}],
        },
    )
    assert r.status_code == 200
    assert len(store.history(f"ann_{tasks['ana']}")) >= 3  # every revision is kept

    snap = call("admin", "POST", "/dataset-snapshots", json={"name": "dev-v1", "splits": ["dev"]}).json()
    assert snap["status"] == "frozen"
    assert (
        call("admin", "POST", "/dataset-snapshots", json={"name": "dev-v1", "splits": ["dev"]}).json()["id"]
        == snap["id"]
    )
    with pytest.raises(Exception):
        store.update(snap["id"], {"items": []})
    result = call(
        "rita",
        "POST",
        "/evaluation-runs",
        json={"snapshot_id": snap["id"], "bootstrap": 20, "systems": {"M0": [mrcd_run["id"]]}},
    ).json()
    m0 = result["data"]["results"]["M0"]
    assert m0["per_class"]["filler_word"]["tp"] == 1 and m0["micro"]["pause"]["fp"] == 1
    assert result["data"]["prior_evaluations_on_snapshot"] == 0


def test_split_leakage_and_test_set_rules(env):
    store, call, users, tmp = env
    sf.write(tmp / "a.wav", np.zeros(16000 * 12, np.float32), 16000)
    sf.write(tmp / "b.wav", np.ones(16000 * 12, np.float32) * 0.01, 16000)
    ra, rb = (
        upload(call, "rita", tmp / "a.wav", speaker_id="spk_x"),
        upload(call, "rita", tmp / "b.wav", speaker_id="spk_x"),
    )
    run_worker(store)
    assert (
        call(
            "rita",
            "POST",
            "/review/tasks",
            json={"recording_id": ra, "start_ms": 0, "end_ms": 5000, "mode": "assisted", "split": "test"},
        ).status_code
        == 400
    )
    for rid, split in ((ra, "train"), (rb, "test")):
        store.create(
            "adjudication",
            {
                "recording_id": rid,
                "start_ms": 0,
                "end_ms": 5000,
                "split": split,
                "speaker_id": "spk_x",
                "guideline_version": "g",
                "mode": "blind",
                "events": [],
                "contextual": [],
            },
            status="final",
        )
    r = call("admin", "POST", "/dataset-snapshots", json={"name": "all"})
    assert r.status_code == 409 and "speaker:spk_x" in r.json()["detail"]["leakage"]
    # known participant under a second ID is still caught once aliased in the project
    store.create("project", {"speaker_aliases": {"spk_y": "spk_x"}}, id="proj")
    for doc in store.find("adjudication"):
        rec = store.get(doc["data"]["recording_id"])
        store.update(rec["id"], {**rec["data"], "project": "proj"})
        if doc["data"]["split"] == "test":
            store.update(doc["id"], {**doc["data"], "speaker_id": "spk_y"})
    r = call("admin", "POST", "/dataset-snapshots", json={"name": "all"})
    assert r.status_code == 409 and "speaker:spk_x" in r.json()["detail"]["leakage"]


def test_cancel_and_crash_recovery(env):
    store, call, users, tmp = env
    sf.write(tmp / "c.wav", np.zeros(16000 * 15, np.float32), 16000)
    rid = upload(call, "uma", tmp / "c.wav")
    run_worker(store)
    run_id = call("uma", "POST", "/analysis-runs", json={"recording_id": rid}).json()["runs"][0]["id"]
    assert call("uma", "POST", f"/analysis-runs/{run_id}/cancel").json()["status"] == "cancelled"
    assert call("uma", "POST", "/analysis-runs", json={"recording_id": rid}).json()["runs"][0]["id"] != run_id
    stuck = store.create(
        "run", {"system": "mrcd", "identity": "x", "heartbeat": time.time() - 10_000}, parent=rid, status="running"
    )
    worker.recover(store)
    assert store.get(stuck["id"])["status"] == "queued" and store.get(stuck["id"])["data"]["attempts"] == 1
    assert call("ana", "POST", "/analysis-runs", json={"recording_id": rid}).status_code == 403
    # a run already claimed by the worker is only asked to stop, then ends cancelled (never finished)
    claimed_id = call("uma", "POST", "/analysis-runs", json={"recording_id": rid, "config": {"v": 2}}).json()["runs"][0]["id"]
    claimed = store.update(claimed_id, status="running")  # as the worker's claim would
    assert call("uma", "POST", f"/analysis-runs/{claimed_id}/cancel").json()["status"] == "cancel_requested"
    worker.execute(store, claimed, SYSTEMS)
    assert store.get(claimed_id)["status"] == "cancelled"
    assert call("uma", "GET", f"/recordings/{rid}/clip", params={"start_ms": 99_000_000, "end_ms": 0}).status_code == 422
    broken = store.create("run", {"system": "missing", "identity": "y"}, parent="rec_gone", status="queued")
    run_worker(store)  # a broken job is logged and left for recovery; the queue keeps draining
    assert store.get(broken["id"])["status"] == "running"


def test_pilot_import_keeps_statuses_and_counts(env, tmp_path):
    from app import cli

    store, _, _, _ = env
    root = tmp_path / "pilot"
    (root / "audio").mkdir(parents=True)
    (root / "resultados" / "piloto").mkdir(parents=True)
    wav = root / "audio" / "rec-1-16k.wav"
    sf.write(wav, np.zeros(16000 * 70, np.float32), 16000, subtype="PCM_16")
    segs = [
        {
            "id": f"s00{i}",
            "core_start": 30.0 * i,
            "core_end": 30.0 * i + 30,
            "audio_start": max(0, 30.0 * i - 5),
            "audio_end": 30.0 * i + 35,
            "pilot": i < 2,
        }
        for i in range(2)
    ]
    (root / "manifest.json").write_text(
        json.dumps(
            {"recording_id": "rec-1", "audio_sha256": media.sha256(wav), "original_sha256": "abc", "segments": segs}
        )
    )
    ev = lambda label, a, s: {"label": label, "start": a, "end": a + 0.5, "decision": "event", "segment_id": s}  # noqa: E731
    files = {
        "mrcd": {
            "events": [ev("neutral_pause", 1, "s000"), ev("filler_word", 31, "s001")],
            "records": [{"segment_id": "s000", "seconds": 1.5}, {"segment_id": "s001", "seconds": 2.0}],
            "initialization_seconds": 0.5,
        },
        "rules_partial": {
            "events": [ev("filler_word", 2, "s000")],
            "incomplete": True,
            "records": [{"segment_id": "s000", "seconds": 1.0}],
        },
    }
    for name, data in files.items():
        (root / "resultados" / "piloto" / f"{name}.json").write_text(json.dumps(data))
    args = type("A", (), {"dir": str(root), "speaker": "spk", "project": None, "owner": None})
    cli.import_pilot(store, args)
    cli.import_pilot(store, args)
    runs = {r["data"]["system"]: r for r in store.find("run")}
    assert (
        len(store.find("run")) == 2 and runs["mrcd"]["status"] == "succeeded" and runs["rules"]["status"] == "partial"
    )
    assert runs["rules"]["data"]["result"]["segments"] == {"s000": "succeeded", "s001": "not_run"}
    assert runs["mrcd"]["data"]["result"]["timings"]["total_s"] == 3.5
    assert [e["start_ms"] for e in runs["mrcd"]["data"]["result"]["events"]] == [1000, 31000]


def test_snapshot_windows_only_use_reviewed_regions_and_mask_unknown(env, monkeypatch):
    import importlib.util
    from pathlib import Path

    from core.constants import LABELS
    from core.engine import PreparedRecording
    from core.extractors.acoustic import extract_acoustic

    store, call, _, tmp = env
    sf.write(tmp / "w.wav", (0.1 * np.sin(np.arange(16000 * 30) / 10)).astype(np.float32), 16000)
    rid = upload(call, "rita", tmp / "w.wav", speaker_id="spk_w")
    run_worker(store)
    spec = importlib.util.spec_from_file_location("sw", Path(__file__).parents[1] / "scripts" / "snapshot_windows.py")
    sw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sw)
    monkeypatch.setattr(sw, "prepared", lambda rec, audio, cfg: PreparedRecording(
        extract_acoustic(audio, use_silero=False), [], len(audio) / 16000, {}))
    items = [{"start_ms": 0, "end_ms": 10_000, "speaker_id": "spk_w", "events": [
        {"label": "filler_word", "decision": "event", "start_ms": 2000, "end_ms": 2500},
        {"label": None, "decision": "uncertain", "start_ms": 6000, "end_ms": 6500}]}]
    out = sw.recording_windows(store.get(rid), items, "small")
    assert out["start_ms"].max() + 3000 <= 10_000
    assert not any(s < 6500 and s + 3000 > 6000 for s in out["start_ms"])
    assert (out["y"] == LABELS.index("filler_word")).any() and not out["has_video"].any()


def test_web_administration_creates_people_tasks_and_cases(env):
    store, call, users, tmp = env
    sf.write(tmp / "adm.wav", (0.1 * np.sin(np.arange(16000 * 20) / 9)).astype(np.float32), 16000)
    rid = upload(call, "rita", tmp / "adm.wav", speaker_id="spk_adm")
    run_worker(store)
    assert call("ana", "GET", "/admin/overview").status_code == 403
    assert call("rita", "POST", "/admin/users", json={"name": "x", "role": "user"}).status_code == 403
    made = call("admin", "POST", "/admin/users", json={"name": "dora", "role": "annotator"}).json()
    assert call(made["token"], "GET", "/me").json()["role"] == "annotator"  # the shown token works
    bad = call("admin", "POST", "/admin/tasks", json={"recording_id": rid, "annotators": [users["ana"]], "split": "test",
                                                      "mode": "assisted"})
    assert bad.status_code == 422 and "prueba" in bad.json()["detail"]
    r = call("admin", "POST", "/admin/tasks", json={"recording_id": rid, "annotators": [users["ana"], made["id"]],
                                                    "split": "dev", "region_s": 10})
    assert r.json()["created"] == 4  # 20 s / 10 s regions x 2 people
    over = call("admin", "GET", "/admin/overview").json()
    assert {t["owner"] for t in over["tasks"]} == {"ana", "dora"}
    assert call("admin", "POST", "/admin/cases", json={"adjudicator": users["carla"]}).json()["created"] == []
    assert {x["recording_id"] for x in call("rita", "GET", "/experiments").json()["runs"]} <= {rid}


def test_asr_settings_are_part_of_cache_key_run_identity_and_record(env):
    """int8 vs float32 ASR give different words across platforms, so a result computed
    with other ASR settings must never be reused or returned as current."""
    from app.systems import asr_config
    from core.engine import ASR_DEFAULTS, cache_key

    assert asr_config({}) == {"whisper_size": "small", "asr_device": "cpu", "asr_compute_type": "float32"}
    assert cache_key(media="m", **asr_config({})) != cache_key(media="m", **asr_config({"asr_compute_type": "int8"}))
    store, call, _, tmp = env
    sf.write(tmp / "id.wav", (0.1 * np.sin(np.arange(16000 * 15) / 7)).astype(np.float32), 16000)
    rid = upload(call, "rita", tmp / "id.wav")
    run_worker(store)
    default = call("rita", "POST", "/analysis-runs", json={"recording_id": rid}).json()["runs"][0]["id"]
    assert store.get(default)["data"]["config"] == ASR_DEFAULTS  # effective settings are stored
    int8 = call("rita", "POST", "/analysis-runs", json={"recording_id": rid, "config": {"asr_compute_type": "int8"}})
    assert int8.json()["runs"][0]["id"] != default


def test_public_campaign_join_resume_progress_and_video(env, tmp_path):
    from app.ops import create_campaign

    store, call, users, tmp = env
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=10:duration=20",
                    "-f", "lavfi", "-i", "sine=frequency=220:duration=20", "-shortest", "-c:v", "libx264",
                    "-c:a", "aac", str(tmp / "talk.mp4")], check=True)
    rid = upload(call, "rita", tmp / "talk.mp4", speaker_id="spk_v")
    run_worker(store)
    create_campaign(store, "piloto-test", rid, [(0, 10_000), (10_000, 20_000)], title="Prueba", max_participants=2)
    anon = TestClient(create_app(store))  # no token: the campaign page is public
    info = anon.get("/api/campaigns/piloto-test").json()
    assert info["regions"] == 2 and info["video"] is True and info["minutes"] == 0.3
    assert anon.post("/api/campaigns/piloto-test/join", json={"name": "Lu", "email": "lu@x.com", "consent": False}).status_code == 422
    assert anon.post("/api/campaigns/piloto-test/join", json={"name": "Lu", "email": "no-email", "consent": True}).status_code == 422
    first = anon.post("/api/campaigns/piloto-test/join", json={"name": "Lu", "email": "Lu@X.com", "consent": True}).json()
    again = anon.post("/api/campaigns/piloto-test/join", json={"name": "Lucía", "email": "lu@x.com", "consent": True}).json()
    assert call(first["token"], "GET", "/me").status_code == 401  # rejoining rotates the token
    tasks = call(again["token"], "GET", "/review/tasks").json()["tasks"]
    assert len(tasks) == 2 and all(t["campaign"] == "piloto-test" for t in tasks)  # resumed, not duplicated
    view = call(again["token"], "GET", f"/review/tasks/{tasks[0]['id']}").json()
    assert view["video"] is True and view["suggestions"] == []  # blind
    clip = call(again["token"], "GET", f"/review/tasks/{tasks[0]['id']}/video")
    assert clip.status_code == 200 and clip.headers["content-type"] == "video/mp4" and len(clip.content) > 1000
    second = anon.post("/api/campaigns/piloto-test/join", json={"name": "Max", "email": "max@x.com", "consent": True}).json()
    full = anon.post("/api/campaigns/piloto-test/join", json={"name": "Zoe", "email": "zoe@x.com", "consent": True})
    assert full.status_code == 422  # max_participants
    for tok, label in ((again["token"], "filler_word"), (second["token"], "repetition")):
        t = call(tok, "GET", "/review/tasks").json()["tasks"][0]
        ann = call(tok, "GET", f"/review/tasks/{t['id']}").json()["annotation"]
        call(tok, "PATCH", f"/annotations/{ann['id']}", headers={"If-Match": str(ann["rev"])},
             json={"events": [{"start_ms": 2000, "end_ms": 2500, "label": label}], "coverage": [[0, 10_000]]})
        assert call(tok, "POST", f"/review/tasks/{t['id']}/submit").json() == {"status": "submitted"}
    assert call(again["token"], "GET", "/campaigns/piloto-test/progress").status_code == 403
    prog = call("rita", "GET", "/campaigns/piloto-test/progress").json()
    assert [(p["name"], p["submitted"], p["total"]) for p in prog["participants"]] == [("Lucía", 1, 2), ("Max", 1, 2)]
    assert prog["agreement"][0]["existence"] == 1.0 and prog["agreement"][0]["raw"] == 0.0  # same span, other class


def test_submit_requires_marks_to_touch_the_fragment(env):
    store, call, users, tmp = env
    sf.write(tmp / "edge.wav", (0.1 * np.sin(np.arange(16000 * 30) / 9)).astype(np.float32), 16000)
    rid = upload(call, "rita", tmp / "edge.wav")
    run_worker(store)
    tid = call("rita", "POST", "/review/tasks", json={"recording_id": rid, "start_ms": 10_000, "end_ms": 20_000,
                                                      "split": "dev", "assignee": users["ana"]}).json()["id"]
    ann = call("ana", "GET", f"/review/tasks/{tid}").json()["annotation"]
    marks = [{"start_ms": 6_000, "end_ms": 6_500, "label": "filler_word"},   # context only -> rejected
             {"start_ms": 9_800, "end_ms": 10_300, "label": "repetition"},   # crosses the edge -> fine
             {"start_ms": 21_000, "end_ms": 21_400, "label": "block"}]       # context only -> rejected
    r = call("ana", "PATCH", f"/annotations/{ann['id']}", headers={"If-Match": str(ann["rev"])},
             json={"events": marks, "coverage": [[10_000, 20_000]]})
    bad = call("ana", "POST", f"/review/tasks/{tid}/submit")
    assert bad.status_code == 422 and bad.json()["detail"]["unreviewed_gaps_ms"] == []
    assert sorted(e["start_ms"] for e in bad.json()["detail"]["outside_context"]) == [6_000, 21_000]
    call("ana", "PATCH", f"/annotations/{ann['id']}", headers={"If-Match": str(r.json()["rev"])},
         json={"events": [marks[1]], "coverage": [[10_000, 20_000]]})
    assert call("ana", "POST", f"/review/tasks/{tid}/submit").json() == {"status": "submitted"}
