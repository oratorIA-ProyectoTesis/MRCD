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
            "case_revision": case["case"].get("rev", store.get(cid)["rev"]),
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


@pytest.mark.parametrize("use_alias", [False, True])
def test_split_leakage_and_test_set_rules(env, use_alias):
    store, call, users, tmp = env
    sf.write(tmp / "a.wav", np.zeros(16000 * 12, np.float32), 16000)
    sf.write(tmp / "b.wav", np.ones(16000 * 12, np.float32) * 0.01, 16000)
    ra, rb = (
        upload(call, "rita", tmp / "a.wav", speaker_id="spk_x"),
        upload(call, "rita", tmp / "b.wav", speaker_id="spk_y" if use_alias else "spk_x"),
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
    if use_alias:
        store.create("project", {"speaker_aliases": {"spk_y": "spk_x"}}, id="proj")
        for rid in (ra, rb):
            rec = store.get(rid)
            store.update(rid, {**rec["data"], "project": "proj"})
    for rid, split in ((ra, "train"), (rb, "test")):
        store.create(
            "adjudication",
            {
                "recording_id": rid,
                "start_ms": 0,
                "end_ms": 5000,
                "split": split,
                "speaker_id": store.get(rid)["data"]["speaker_id"],
                "guideline_version": "g",
                "mode": "blind",
                "events": [],
                "contextual": [],
            },
            status="final",
        )
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
    claimed_id = call("uma", "POST", "/analysis-runs", json={"recording_id": rid, "config": {"v": 2}}).json()["runs"][
        0
    ]["id"]
    claimed = store.update(claimed_id, status="running")  # as the worker's claim would
    assert call("uma", "POST", f"/analysis-runs/{claimed_id}/cancel").json()["status"] == "cancel_requested"
    worker.execute(store, claimed, SYSTEMS)
    assert store.get(claimed_id)["status"] == "cancelled"
    assert (
        call("uma", "GET", f"/recordings/{rid}/clip", params={"start_ms": 99_000_000, "end_ms": 0}).status_code == 422
    )
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
    monkeypatch.setattr(
        sw,
        "prepared",
        lambda rec, audio, cfg: PreparedRecording(
            extract_acoustic(audio, use_silero=False), [], len(audio) / 16000, {}
        ),
    )
    items = [
        {
            "start_ms": 0,
            "end_ms": 10_000,
            "speaker_id": "spk_w",
            "events": [
                {"label": "filler_word", "decision": "event", "start_ms": 2000, "end_ms": 2500},
                {"label": None, "decision": "uncertain", "start_ms": 6000, "end_ms": 6500},
            ],
        }
    ]
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
    bad = call(
        "admin",
        "POST",
        "/admin/tasks",
        json={"recording_id": rid, "annotators": [users["ana"]], "split": "test", "mode": "assisted"},
    )
    assert bad.status_code == 422 and "prueba" in bad.json()["detail"]
    r = call(
        "admin",
        "POST",
        "/admin/tasks",
        json={"recording_id": rid, "annotators": [users["ana"], made["id"]], "split": "dev", "region_s": 10},
    )
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
    assert store.get(default)["data"]["config"] == {**ASR_DEFAULTS, "engine_kind": "legacy_window_v1"}  # exact effective settings
    int8 = call("rita", "POST", "/analysis-runs", json={"recording_id": rid, "config": {"asr_compute_type": "int8"}})
    assert int8.json()["runs"][0]["id"] != default


def test_public_campaign_join_resume_progress_and_video(env, tmp_path):
    from app.ops import create_campaign

    store, call, users, tmp = env
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=320x240:rate=10:duration=20",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=220:duration=20",
            "-shortest",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            str(tmp / "talk.mp4"),
        ],
        check=True,
    )
    rid = upload(call, "rita", tmp / "talk.mp4", speaker_id="spk_v")
    run_worker(store)
    create_campaign(store, "piloto-test", rid, [(0, 10_000), (10_000, 20_000)], title="Prueba", max_participants=2)
    anon = TestClient(create_app(store))  # no token: the campaign page is public
    info = anon.get("/api/campaigns/piloto-test").json()
    assert info["regions"] == 2 and info["video"] is True and info["minutes"] == 0.3
    assert (
        anon.post(
            "/api/campaigns/piloto-test/join", json={"name": "Lu", "email": "lu@x.com", "consent": False}
        ).status_code
        == 422
    )
    assert (
        anon.post(
            "/api/campaigns/piloto-test/join", json={"name": "Lu", "email": "no-email", "consent": True}
        ).status_code
        == 422
    )
    first = anon.post(
        "/api/campaigns/piloto-test/join", json={"name": "Lu", "email": "Lu@X.com", "consent": True, "consent_version": info["consent_version"]}
    ).json()
    again = anon.post(
        "/api/campaigns/piloto-test/resume", headers={"Authorization": f"Bearer {first['token']}"}
    ).json()
    assert again["campaign"] == "piloto-test" and "token" not in again
    assert call(first["token"], "GET", "/me").status_code == 200  # returning never revokes the token
    tasks = call(first["token"], "GET", "/review/tasks").json()["tasks"]
    assert len(tasks) == 2 and all(t["campaign"] == "piloto-test" for t in tasks)  # resumed, not duplicated
    view = call(first["token"], "GET", f"/review/tasks/{tasks[0]['id']}").json()
    assert view["video"] is True and view["suggestions"] == []  # blind
    clip = call(first["token"], "GET", f"/review/tasks/{tasks[0]['id']}/video")
    assert clip.status_code == 200 and clip.headers["content-type"] == "video/mp4" and len(clip.content) > 1000
    second = anon.post(
        "/api/campaigns/piloto-test/join", json={"name": "Max", "email": "max@x.com", "consent": True, "consent_version": info["consent_version"]}
    ).json()
    full = anon.post("/api/campaigns/piloto-test/join", json={"name": "Zoe", "email": "zoe@x.com", "consent": True, "consent_version": info["consent_version"]})
    assert full.status_code == 422  # max_participants
    for tok, label in ((first["token"], "filler_word"), (second["token"], "repetition")):
        t = call(tok, "GET", "/review/tasks").json()["tasks"][0]
        ann = call(tok, "GET", f"/review/tasks/{t['id']}").json()["annotation"]
        call(
            tok,
            "PATCH",
            f"/annotations/{ann['id']}",
            headers={"If-Match": str(ann["rev"])},
            json={"events": [{"start_ms": 2000, "end_ms": 2500, "label": label}], "coverage": [[0, 10_000]]},
        )
        assert call(tok, "POST", f"/review/tasks/{t['id']}/submit").json() == {"status": "submitted"}
    assert call(first["token"], "GET", "/campaigns/piloto-test/progress").status_code == 403
    prog = call("rita", "GET", "/campaigns/piloto-test/progress").json()
    assert [(p["name"], p["submitted"], p["total"]) for p in prog["participants"]] == [("Lu", 1, 2), ("Max", 1, 2)]
    assert prog["agreement"][0]["existence"] == 1.0 and prog["agreement"][0]["raw"] == 0.0  # same span, other class
    from app.ops import extend_campaign

    extend_campaign(store, "piloto-test", [(10_000, 20_000), (20_000, 30_000)])  # one repeated, one new
    tasks = call(first["token"], "GET", "/review/tasks").json()["tasks"]  # same token keeps working
    assert sorted(t["start_ms"] for t in tasks) == [0, 10_000, 20_000]
    assert sum(t["status"] == "submitted" for t in tasks) == 1  # finished work is kept


def security_campaign(env, slug="secure-test", capacity=4):
    """Synthetic metadata only: no real recordings, models, or participants."""
    from app.ops import create_campaign

    store, _, _, _ = env
    rec = store.create("recording", {
        "sha256": "a" * 64, "speaker_id": "synthetic",
        "media": {"original": {"has_video": False}, "analysis": {"duration_ms": 20_000}},
    }, status="ready")
    create_campaign(store, slug, rec["id"], [(0, 10_000), (10_000, 20_000)], title="Synthetic security test", max_participants=capacity)
    client = TestClient(create_app(store))
    return client, slug


def registration(email="person@example.invalid", version="mrcd-campaign-consent-v1"):
    return {"name": "Synthetic Person", "email": email, "consent": True, "consent_version": version}


def test_campaign_email_alone_cannot_rotate_or_modify_an_account(env):
    store, call, _, _ = env
    anon, slug = security_campaign(env)
    response = anon.post(f"/api/campaigns/{slug}/join", json=registration())
    token = response.json()["token"]
    before = store.find("user", parent=slug)[0]
    attack = anon.post(f"/api/campaigns/{slug}/join", json={**registration(), "name": "Impersonator"})
    assert attack.status_code == 409 and "token" not in attack.json()
    assert store.get(before["id"]) == before
    assert call(token, "GET", "/me").status_code == 200
    assert response.headers["cache-control"] == "no-store"


def test_campaign_return_requires_existing_token_and_preserves_progress(env):
    store, call, _, _ = env
    anon, slug = security_campaign(env)
    token = anon.post(f"/api/campaigns/{slug}/join", json=registration()).json()["token"]
    person = store.find("user", parent=slug)[0]
    tasks = call(token, "GET", "/review/tasks").json()["tasks"]
    store.update(tasks[0]["id"], status="submitted")
    assert anon.post(f"/api/campaigns/{slug}/resume").status_code == 401
    assert call("invalid", "POST", f"/campaigns/{slug}/resume").status_code == 401
    resumed = call(token, "POST", f"/campaigns/{slug}/resume")
    assert resumed.status_code == 200 and resumed.json()["campaign"] == slug
    assert "token" not in resumed.json() and store.get(person["id"]) == person
    again = call(token, "GET", "/review/tasks").json()["tasks"]
    assert len(again) == 2 and next(t for t in again if t["id"] == tasks[0]["id"])["status"] == "submitted"


def test_campaign_return_rejects_cross_campaign_roles_and_inactive_users(env):
    store, call, _, _ = env
    anon, slug = security_campaign(env)
    token = anon.post(f"/api/campaigns/{slug}/join", json=registration()).json()["token"]
    _, other = security_campaign(env, "other-test")
    assert call(token, "POST", f"/campaigns/{other}/resume").status_code == 403
    for role in ("admin", "rita", "ana", "uma"):
        assert call(role, "POST", f"/campaigns/{slug}/resume").status_code == 403
    person = store.find("user", parent=slug)[0]
    store.update(person["id"], status="inactive")
    assert call(token, "POST", f"/campaigns/{slug}/resume").status_code == 401
    assert anon.post(f"/api/campaigns/{slug}/join", json=registration()).status_code == 409
    assert store.get(person["id"])["status"] == "inactive"


def test_campaign_consent_is_current_explicit_and_legacy_is_not_backfilled(env):
    from datetime import datetime

    store, call, _, _ = env
    anon, slug = security_campaign(env)
    info = anon.get(f"/api/campaigns/{slug}").json()
    assert info["consent_version"] == "mrcd-campaign-consent-v1" and info["consent_text"]
    for body in (registration(version="outdated"), {k: v for k, v in registration().items() if k != "consent_version"}, {**registration(), "consent": False}):
        assert anon.post(f"/api/campaigns/{slug}/join", json=body).status_code == 422
    token = anon.post(f"/api/campaigns/{slug}/join", json=registration()).json()["token"]
    person = store.find("user", parent=slug)[0]
    consent = person["data"]["consent"]
    assert consent["version"] == info["consent_version"] and consent["campaign"] == slug
    assert datetime.fromisoformat(consent["accepted_at"]).tzinfo is not None
    assert consent["text_sha256"] == hashlib.sha256(info["consent_text"].encode()).hexdigest()
    legacy = {k: v for k, v in person["data"].items() if k != "consent"}
    store.update(person["id"], legacy)
    assert call(token, "POST", f"/campaigns/{slug}/resume").status_code == 200
    assert "consent" not in store.get(person["id"])["data"]


def test_campaign_reset_requires_admin_manual_identity_check_and_audit_reason(env):
    store, call, users, _ = env
    anon, slug = security_campaign(env)
    old = anon.post(f"/api/campaigns/{slug}/join", json=registration()).json()["token"]
    person = store.find("user", parent=slug)[0]
    task_ids = sorted(t["id"] for t in store.find("task", owner=person["id"]))
    path = f"/admin/campaigns/{slug}/participants/{person['id']}/reset-access"
    body = {"identity_verified": True, "reason": "Verified manually through the team's existing contact channel"}
    for actor in (old, "rita", "ana", "uma"):
        assert call(actor, "POST", path, json=body).status_code == 403
    for bad in ({**body, "identity_verified": False}, {**body, "reason": " "}):
        assert call("admin", "POST", path, json=bad).status_code == 422
    assert call(old, "GET", "/me").status_code == 200
    reset = call("admin", "POST", path, json=body)
    assert reset.status_code == 200 and reset.headers["cache-control"] == "no-store"
    new = reset.json()["token"]
    assert new != old and call(old, "GET", "/me").status_code == 401
    assert call(new, "POST", f"/campaigns/{slug}/resume").status_code == 200
    assert task_ids == sorted(t["id"] for t in store.find("task", owner=person["id"]))
    updated = store.get(person["id"])
    assert updated["data"]["consent"] == person["data"]["consent"]
    assert updated["data"]["access_reset"]["reason"] == body["reason"]
    assert updated["data"]["access_reset"]["by"] == users["admin"]
    assert store.history(person["id"])[-1]["author"] == users["admin"]
    assert new not in json.dumps(store.history(person["id"]))


def test_campaign_reset_cannot_cross_campaign_or_reactivate_users(env):
    store, call, users, _ = env
    anon, slug = security_campaign(env)
    anon.post(f"/api/campaigns/{slug}/join", json=registration())
    person = store.find("user", parent=slug)[0]
    _, other = security_campaign(env, "other-test")
    body = {"identity_verified": True, "reason": "Verified through the existing contact channel"}
    assert call("admin", "POST", f"/admin/campaigns/{other}/participants/{person['id']}/reset-access", json=body).status_code == 422
    assert call("admin", "POST", f"/admin/campaigns/{slug}/participants/{users['rita']}/reset-access", json=body).status_code == 422
    store.update(person["id"], status="inactive")
    assert call("admin", "POST", f"/admin/campaigns/{slug}/participants/{person['id']}/reset-access", json=body).status_code == 422


@pytest.mark.parametrize("same_email,capacity", [(True, 4), (False, 2)])
def test_campaign_concurrent_registration_respects_identity_capacity_and_tasks(env, same_email, capacity):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    store, _, _, _ = env
    _, slug = security_campaign(env, capacity=capacity)
    barrier = Barrier(6)

    def join(index):
        client = TestClient(create_app(store))
        barrier.wait(timeout=10)
        return client.post(f"/api/campaigns/{slug}/join", json=registration(f"person{0 if same_email else index}@example.invalid")).status_code

    with ThreadPoolExecutor(max_workers=6) as pool:
        statuses = list(pool.map(join, range(6)))
    count = 1 if same_email else capacity
    assert statuses.count(200) == count and all(s in (200, 409, 422) for s in statuses)
    people = store.find("user", parent=slug)
    assert len(people) == count and len({p["data"]["email"] for p in people}) == count
    for person in people:
        tasks = store.find("task", owner=person["id"])
        assert len(tasks) == 2 and len({(t["data"]["start_ms"], t["data"]["end_ms"]) for t in tasks}) == 2


def test_campaign_registration_rolls_back_user_and_tasks_on_failure(env, monkeypatch):
    from app import ops

    store, _, _, _ = env
    anon, slug = security_campaign(env)

    original_tasks = ops.campaign_tasks

    def broken_tasks(*args, **kwargs):
        original_tasks(*args, **kwargs)
        raise RuntimeError("synthetic task failure")

    monkeypatch.setattr(ops, "campaign_tasks", broken_tasks)
    with pytest.raises(RuntimeError, match="synthetic task failure"):
        anon.post(f"/api/campaigns/{slug}/join", json=registration())
    assert store.find("user", parent=slug) == [] and store.find("task") == []


def test_campaign_concurrent_extensions_keep_union_and_unique_tasks(env):
    from concurrent.futures import ThreadPoolExecutor
    from app.ops import extend_campaign

    store, _, _, _ = env
    anon, slug = security_campaign(env)
    anon.post(f"/api/campaigns/{slug}/join", json=registration())
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda n: extend_campaign(store, slug, [(n * 1000, (n + 1) * 1000)]), [1, 2, 1, 2]))
    assert len(store.get(slug)["data"]["regions"]) == 4
    person = store.find("user", parent=slug)[0]
    assert len(store.find("task", owner=person["id"])) == 4


def test_campaign_accounts_cannot_claim_global_pool_or_read_outside_tasks_and_examples(env):
    store, call, _, _ = env
    anon, slug = security_campaign(env)
    token = anon.post(f"/api/campaigns/{slug}/join", json=registration()).json()["token"]
    person = store.find("user", parent=slug)[0]
    for task in store.find("task", owner=person["id"]):
        store.update(task["id"], status="submitted")
    pool = store.create("task", {
        "recording_id": "synthetic-other", "start_ms": 0, "end_ms": 1000,
        "mode": "blind", "context_ms": 0, "guideline_version": "synthetic",
    }, status="pool")
    assert call(token, "GET", "/review/tasks/next").json() is None
    assert store.get(pool["id"])["owner"] is None
    assert call("ana", "GET", "/review/tasks/next").json()["id"] == pool["id"]
    outside = store.create("task", pool["data"], owner=person["id"], status="assigned")
    assert call(token, "GET", f"/review/tasks/{outside['id']}").status_code == 403
    assert outside["id"] not in {t["id"] for t in call(token, "GET", "/review/tasks").json()["tasks"]}
    rec = store.find("recording")[0]
    store.update(rec["id"], {**rec["data"], "example": True})
    assert call(token, "GET", f"/recordings/{rec['id']}").status_code == 403


@pytest.mark.parametrize("endpoint", ["/admin/tasks", "/review/tasks"])
def test_generic_assignment_rejects_campaign_accounts_without_orphans_but_regular_tasks_open(env, endpoint):
    store, call, users, _ = env
    anon, slug = security_campaign(env)
    anon.post(f"/api/campaigns/{slug}/join", json=registration())
    participant = store.find("user", parent=slug)[0]
    rec = store.find("recording")[0]
    store.update(rec["id"], {**rec["data"], "segments": media.segments(20_000)})
    before_tasks = store.find("task")
    before_progress = call("admin", "GET", f"/campaigns/{slug}/progress").json()
    body = {"recording_id": rec["id"], "split": "dev"}
    if endpoint == "/admin/tasks":
        body.update(annotators=[users["ana"], participant["id"]], region_s=10)
    else:
        body.update(assignee=participant["id"], start_ms=0, end_ms=10_000)
    rejected = call("admin", "POST", endpoint, json=body)
    assert rejected.status_code == 422
    assert store.find("task") == before_tasks
    assert call("admin", "GET", f"/campaigns/{slug}/progress").json() == before_progress
    # An ordinary annotator remains assignable, can list the task, and can open it.
    if endpoint == "/admin/tasks":
        body["annotators"] = [users["ana"]]
    else:
        body["assignee"] = users["ana"]
    assert call("admin", "POST", endpoint, json=body).status_code == 200
    listed = call("ana", "GET", "/review/tasks").json()["tasks"]
    assert listed and call("ana", "GET", f"/review/tasks/{listed[0]['id']}").status_code == 200
    overview = call("admin", "GET", "/admin/overview").json()
    assert next(u for u in overview["users"] if u["id"] == participant["id"])["campaign"] == slug


def test_campaign_progress_ignores_legacy_unscoped_and_other_campaign_tasks(env):
    store, call, _, _ = env
    anon, slug = security_campaign(env)
    anon.post(f"/api/campaigns/{slug}/join", json=registration())
    participant = store.find("user", parent=slug)[0]
    before = call("admin", "GET", f"/campaigns/{slug}/progress").json()
    for index, campaign in enumerate((None, "another-campaign")):
        task = store.create("task", {
            "start_ms": 30_000 + index * 1000, "end_ms": 31_000 + index * 1000,
            "campaign": campaign,
        }, owner=participant["id"], status="submitted")
        store.create("annotation", {"events": [], "active_ms": 120_000}, id=f"ann_{task['id']}")
    assert call("admin", "GET", f"/campaigns/{slug}/progress").json() == before


def test_campaign_board_copy_does_not_promise_email_recovery():
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "app/static/app.js").read_text()
    assert "volver con tu correo" not in source and "volver con el mismo correo" not in source
    board = source.split("async function campaign(slug)", 1)[1].split("async function", 1)[0]
    assert "personal token" in board and "administrator" in board


def submitted_review_pair(env):
    """Independent synthetic annotations with complete source metadata; no real people/media."""
    store, _, users, _ = env
    rec = store.create("recording", {
        "sha256": "b" * 64, "speaker_id": "synthetic-reviewer",
        "media": {"original": {"has_video": False}, "analysis": {"duration_ms": 20_000, "sha256": "c" * 64}},
    }, status="ready")
    tasks = []
    for who in ("ana", "beto"):
        task = store.create("task", {
            "recording_id": rec["id"], "start_ms": 0, "end_ms": 10_000, "context_ms": 5000,
            "split": "dev", "speaker_id": "synthetic-reviewer", "mode": "blind", "guideline_version": "guia-anotacion-v1",
        }, owner=users[who], parent=rec["id"], status="submitted")
        store.create("annotation", {
            "events": [{"label": "filler_word", "decision": "event", "start_ms": 2000, "end_ms": 2500}],
            "contextual": [], "coverage": [[0, 10_000]],
        }, id=f"ann_{task['id']}", owner=users[who], parent=task["id"], status="submitted")
        tasks.append(task)
    return rec, tasks


@pytest.mark.parametrize("endpoint", ["/adjudication-cases", "/admin/cases"])
def test_campaign_adjudicator_is_rejected_before_case_creation_and_pair_remains_available(env, endpoint):
    store, call, users, _ = env
    anon, slug = security_campaign(env)
    anon.post(f"/api/campaigns/{slug}/join", json=registration())
    campaign_person = store.find("user", parent=slug)[0]
    _, tasks = submitted_review_pair(env)
    body = {"adjudicator": campaign_person["id"]}
    if endpoint == "/adjudication-cases":
        body.update(task_a=tasks[0]["id"], task_b=tasks[1]["id"])
    assert call("admin", "POST", endpoint, json=body).status_code == 422
    assert store.find("case") == []
    valid = call("admin", "POST", "/admin/cases", json={"adjudicator": users["carla"]})
    assert valid.status_code == 200 and len(valid.json()["created"]) == 1
    case_id = valid.json()["created"][0]
    assert call("carla", "GET", "/review/tasks").json()["cases"] == [{"id": case_id, "status": "open"}]
    assert call("carla", "GET", f"/review/adjudication/{case_id}").status_code == 200


def open_synthetic_case(env):
    store, call, users, _ = env
    rec, tasks = submitted_review_pair(env)
    cid = call("admin", "POST", "/adjudication-cases", json={
        "task_a": tasks[0]["id"], "task_b": tasks[1]["id"], "adjudicator": users["carla"],
    }).json()["id"]
    view = call("carla", "GET", f"/review/adjudication/{cid}").json()
    body = {"case_id": cid, "case_revision": store.get(cid)["rev"],
            "based_on": {side: view["tracks"][side]["rev"] for side in "AB"},
            "events": [{"label": "filler_word", "start_ms": 2000, "end_ms": 2500}], "reasons": {"decision": "Synthetic adjudication"}}
    return rec, tasks, cid, body


def test_adjudication_retry_is_idempotent_conflicts_are_rejected_and_decision_is_immutable(env):
    from app.store import Conflict

    store, call, _, _ = env
    _, _, cid, body = open_synthetic_case(env)
    first = call("carla", "POST", "/adjudications", json=body)
    assert first.status_code == 200
    retry = call("carla", "POST", "/adjudications", json=body)
    assert retry.status_code == 200 and retry.json() == first.json()
    assert len(store.find("adjudication")) == 1
    changed = {**body, "events": [{"label": "repetition", "start_ms": 2000, "end_ms": 2500}]}
    assert call("carla", "POST", "/adjudications", json=changed).status_code == 409
    with pytest.raises(Conflict):
        store.update(first.json()["id"], {"events": []})
    assert store.get(cid)["status"] == "closed"


@pytest.mark.parametrize("identical", [True, False])
def test_concurrent_adjudication_has_one_final_and_no_half_closed_case(env, identical):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    store, _, _, _ = env
    _, _, cid, body = open_synthetic_case(env)
    application = create_app(store)
    barrier = Barrier(4)

    def submit(index):
        payload = {**body, "events": [{**body["events"][0], "label": "filler_word" if identical else ("repetition" if index else "filler_word")}]}
        barrier.wait(timeout=10)
        return TestClient(application).post("/api/adjudications", json=payload, headers={"Authorization": "Bearer carla"})

    with ThreadPoolExecutor(max_workers=4) as pool:
        responses = list(pool.map(submit, range(4)))
    winners = [r for r in responses if r.status_code == 200]
    assert winners and len({r.json()["id"] for r in winners}) == 1
    assert all(r.status_code == 200 if identical else r.status_code in (200, 409) for r in responses)
    assert len(store.find("adjudication")) == 1 and store.get(cid)["status"] == "closed"


def test_adjudication_failure_rolls_back_decision_case_and_history(env, monkeypatch):
    store, call, _, _ = env
    _, _, cid, body = open_synthetic_case(env)
    before = store.get(cid)
    history = store.history(cid)
    original = Store.update

    def fail_after_close(self, doc_id, *args, **kwargs):
        result = original(self, doc_id, *args, **kwargs)
        if doc_id == cid and kwargs.get("status") == "closed":
            raise RuntimeError("synthetic closing failure")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(Store, "update", fail_after_close)
        with pytest.raises(RuntimeError, match="synthetic closing failure"):
            call("carla", "POST", "/adjudications", json=body)
    assert store.find("adjudication") == [] and store.get(cid) == before and store.history(cid) == history
    assert call("carla", "POST", "/adjudications", json=body).status_code == 200


@pytest.mark.parametrize("changed", ["annotation", "case"])
def test_adjudication_rejects_stale_case_and_source_revisions_before_write(env, changed):
    store, call, _, _ = env
    _, tasks, cid, body = open_synthetic_case(env)
    doc = store.get(f"ann_{tasks[0]['id']}" if changed == "annotation" else cid)
    store.update(doc["id"], doc["data"])
    assert call("carla", "POST", "/adjudications", json=body).status_code == 409
    assert store.find("adjudication") == [] and store.get(cid)["status"] == "open"


def test_explicit_adjudication_revision_preserves_prior_decision_and_frozen_snapshot(env):
    store, call, _, _ = env
    _, _, cid, body = open_synthetic_case(env)
    first = call("carla", "POST", "/adjudications", json=body).json()
    original = store.get(first["id"])
    frozen = call("admin", "POST", "/dataset-snapshots", json={"name": "before", "splits": ["dev"]}).json()
    correction = {**body, "case_revision": store.get(cid)["rev"], "supersedes": first["id"],
                  "revision_reason": "Explicit correction after synthetic review",
                  "events": [{"label": "repetition", "start_ms": 2000, "end_ms": 2500}]}
    revised = call("carla", "POST", "/adjudications", json=correction)
    assert revised.status_code == 200 and revised.json()["id"] != first["id"]
    assert store.get(first["id"]) == original and store.get(frozen["id"]) == frozen
    assert len(store.find("adjudication")) == 2
    current = call("admin", "POST", "/dataset-snapshots", json={"name": "after", "splits": ["dev"]}).json()
    assert len(current["data"]["items"]) == 1
    assert current["data"]["items"][0]["source"] == revised.json()["id"]
    assert call("carla", "POST", "/adjudications", json=correction).json() == revised.json()


def test_snapshot_rejects_ambiguous_historical_finals_instead_of_selecting_one(env):
    store, call, _, _ = env
    _, _, cid, body = open_synthetic_case(env)
    first = call("carla", "POST", "/adjudications", json=body).json()
    original = store.get(first["id"])
    store.create("adjudication", original["data"], parent=cid, status="final")
    result = call("admin", "POST", "/dataset-snapshots", json={"name": "ambiguous", "splits": ["dev"]})
    assert result.status_code == 409 and "ambiguous" in str(result.json()["detail"]).lower()
    assert store.find("snapshot") == [] and store.get(first["id"]) == original


@pytest.mark.parametrize("field", ["speaker_id", "media_sha256", "guideline_version", "split", "taxonomy_version"])
def test_snapshot_validates_current_source_metadata_against_immutable_decision(env, field):
    store, call, _, _ = env
    rec, _, cid, body = open_synthetic_case(env)
    assert call("carla", "POST", "/adjudications", json=body).status_code == 200
    if field == "speaker_id":
        store.update(rec["id"], {**rec["data"], "speaker_id": "different-speaker"})
    elif field == "media_sha256":
        analysis = {**rec["data"]["media"]["analysis"], "sha256": "d" * 64}
        store.update(rec["id"], {**rec["data"], "media": {**rec["data"]["media"], "analysis": analysis}})
    else:
        case = store.get(cid)
        store.update(cid, {**case["data"], field: "changed"})
    result = call("admin", "POST", "/dataset-snapshots", json={"name": "invalid", "splits": ["dev"]})
    assert result.status_code in (409, 422) and field in str(result.json()["detail"])
    assert store.find("snapshot") == []


@pytest.mark.parametrize("events,contextual", [
    ([{"label": "filler_word", "start_ms": 2000, "end_ms": 25_000}], []),
    ([], [{"expression": "synthetic", "start_ms": 4000, "end_ms": 3000}]),
])
def test_adjudication_validates_media_bounds_and_contextual_intervals(env, events, contextual):
    store, call, _, _ = env
    _, _, cid, body = open_synthetic_case(env)
    result = call("carla", "POST", "/adjudications", json={**body, "events": events, "contextual": contextual})
    assert result.status_code == 422 and store.find("adjudication") == []
    assert store.get(cid)["status"] == "open"


def test_snapshot_preserves_overlapping_phenomena_and_unions_coverage_with_stable_hash(env):
    from concurrent.futures import ThreadPoolExecutor

    store, call, users, _ = env
    rec, tasks, _, body = open_synthetic_case(env)
    body["events"].append({"label": "repetition", "start_ms": 2100, "end_ms": 2600})
    assert call("carla", "POST", "/adjudications", json=body).status_code == 200
    next_tasks = []
    for task in tasks:
        new_task = store.create("task", {**task["data"], "start_ms": 10_000, "end_ms": 20_000},
                                owner=task["owner"], parent=rec["id"], status="submitted")
        store.create("annotation", {"events": [], "contextual": [], "coverage": [[10_000, 20_000]]},
                     id=f"ann_{new_task['id']}", parent=new_task["id"], owner=task["owner"], status="submitted")
        next_tasks.append(new_task)
    cid = call("admin", "POST", "/adjudication-cases", json={
        "task_a": next_tasks[0]["id"], "task_b": next_tasks[1]["id"], "adjudicator": users["carla"]}).json()["id"]
    view = call("carla", "GET", f"/review/adjudication/{cid}").json()
    assert call("carla", "POST", "/adjudications", json={
        "case_id": cid, "case_revision": store.get(cid)["rev"], "based_on": {s: view["tracks"][s]["rev"] for s in "AB"}, "events": [],
    }).status_code == 200
    application = create_app(store)
    def freeze(_):
        return TestClient(application).post("/api/dataset-snapshots", json={"name": "union", "splits": ["dev"]}, headers={"Authorization": "Bearer admin"})
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(freeze, range(3)))
    assert all(r.status_code == 200 for r in results) and len({r.json()["id"] for r in results}) == 1
    snapshot = results[0].json()
    assert snapshot["data"]["coverage"] == {rec["id"]: [[0, 20_000]]}
    assert len(snapshot["data"]["items"][0]["events"]) == 2
    assert snapshot["data"]["reference_kind"] == "human_adjudicated"


def test_submitted_snapshot_does_not_pool_two_independent_annotations_as_ground_truth(env):
    store, call, _, _ = env
    submitted_review_pair(env)
    result = call("admin", "POST", "/dataset-snapshots", json={"name": "independent", "include": "submitted", "splits": ["dev"]})
    assert result.status_code == 409 and "independent" in str(result.json()["detail"]).lower()
    assert store.find("snapshot") == []


def test_adjudication_and_revision_require_explicit_revision_tokens_and_reason(env):
    store, call, _, _ = env
    _, _, cid, body = open_synthetic_case(env)
    assert call("carla", "POST", "/adjudications", json={k: v for k, v in body.items() if k != "case_revision"}).status_code == 422
    first = call("carla", "POST", "/adjudications", json=body).json()
    corrected = {**body, "case_revision": store.get(cid)["rev"], "supersedes": first["id"], "events": []}
    assert call("carla", "POST", "/adjudications", json=corrected).status_code == 422
    assert len(store.find("adjudication")) == 1


def test_snapshot_detects_changed_annotation_revision_after_finalization(env):
    store, call, _, _ = env
    _, tasks, _, body = open_synthetic_case(env)
    call("carla", "POST", "/adjudications", json=body)
    ann = store.get(f"ann_{tasks[0]['id']}")
    store.update(ann["id"], ann["data"])
    result = call("admin", "POST", "/dataset-snapshots", json={"name": "stale", "splits": ["dev"]})
    assert result.status_code == 409 and "based_on" in str(result.json()["detail"])
    assert store.find("snapshot") == []


def test_invalid_missing_media_identity_is_a_diagnostic_not_an_internal_error(env):
    store, call, _, _ = env
    rec, _, cid, body = open_synthetic_case(env)
    store.update(rec["id"], {**rec["data"], "media": {
        **rec["data"]["media"], "analysis": {**rec["data"]["media"]["analysis"], "sha256": None}}})
    result = call("carla", "POST", "/adjudications", json=body)
    assert result.status_code == 422 and "media_sha256" in str(result.json()["detail"])
    assert store.find("adjudication") == [] and store.get(cid)["status"] == "open"


def test_malformed_imported_event_container_does_not_crash_snapshot_validation(env):
    store, call, _, _ = env
    _, _, cid, body = open_synthetic_case(env)
    first = call("carla", "POST", "/adjudications", json=body).json()
    bad = store.create("adjudication", {**store.get(first["id"])["data"], "supersedes": first["id"],
        "decision_revision": 2, "revision_reason": "Synthetic imported correction", "events": None}, parent=cid, status="final")
    case = store.get(cid)
    store.update(cid, {**case["data"], "current_adjudication": bad["id"], "adjudication_revision": 2})
    result = call("admin", "POST", "/dataset-snapshots", json={"name": "invalid-import", "splits": ["dev"]})
    assert result.status_code == 422 and "events" in str(result.json()["detail"])
    assert store.find("snapshot") == []


def test_single_submitted_annotation_snapshot_is_not_called_adjudicated(env):
    store, call, _, _ = env
    _, tasks = submitted_review_pair(env)
    store.update(tasks[1]["id"], status="assigned")
    result = call("admin", "POST", "/dataset-snapshots", json={"name": "single", "include": "submitted", "splits": ["dev"]})
    assert result.status_code == 200
    assert result.json()["data"]["reference_kind"] == "human_single_annotation"
    assert result.json()["data"]["items"][0]["source_kind"] == "human_single_annotation"


def test_reference_coverage_union_handles_overlaps_touching_nested_and_empty():
    from app.references import union_intervals

    assert union_intervals([]) == []
    assert union_intervals([(10, 15), (2, 5), (4, 12), (3, 4), (15, 20)]) == [[2, 20]]


@pytest.mark.parametrize("endpoint", ["/adjudication-cases", "/admin/cases"])
def test_mixed_source_modes_finalize_and_freeze_as_assisted(env, endpoint):
    store, call, users, _ = env
    rec, tasks = submitted_review_pair(env)
    store.update(tasks[1]["id"], {**tasks[1]["data"], "mode": "assisted"})
    request = {"adjudicator": users["carla"]}
    if endpoint == "/adjudication-cases":
        request.update(task_a=tasks[0]["id"], task_b=tasks[1]["id"])
    created = call("admin", "POST", endpoint, json=request)
    assert created.status_code == 200
    cid = created.json()["id"] if endpoint == "/adjudication-cases" else created.json()["created"][0]
    view = call("carla", "GET", f"/review/adjudication/{cid}").json()
    result = call("carla", "POST", "/adjudications", json={
        "case_id": cid, "case_revision": view["case"]["rev"],
        "based_on": {s: view["tracks"][s]["rev"] for s in "AB"}, "events": [],
    })
    assert result.status_code == 200, result.text
    frozen = call("admin", "POST", "/dataset-snapshots", json={"name": "mixed-mode", "splits": ["dev"]})
    assert frozen.status_code == 200, frozen.text
    item = frozen.json()["data"]["items"][0]
    assert item["mode"] == "assisted" and sorted(item["source_modes"].values()) == ["assisted", "blind"]
    assert item["recording_id"] == rec["id"]


def test_case_detects_changed_individual_mode_even_when_aggregate_stays_assisted(env):
    store, call, users, _ = env
    _, tasks = submitted_review_pair(env)
    store.update(tasks[1]["id"], {**tasks[1]["data"], "mode": "assisted"})
    cid = call("admin", "POST", "/adjudication-cases", json={
        "task_a": tasks[0]["id"], "task_b": tasks[1]["id"], "adjudicator": users["carla"]}).json()["id"]
    view = call("carla", "GET", f"/review/adjudication/{cid}").json()
    store.update(tasks[0]["id"], {**tasks[0]["data"], "mode": "assisted"})
    result = call("carla", "POST", "/adjudications", json={
        "case_id": cid, "case_revision": view["case"]["rev"],
        "based_on": {s: view["tracks"][s]["rev"] for s in "AB"}, "events": []})
    assert result.status_code == 409 and "mode" in result.text
    assert store.find("adjudication") == []


def legacy_open_case(env):
    from app import media_dir
    from core.constants import TAXONOMY_VERSION

    store, _, _, _ = env
    rec, tasks, cid, body = open_synthetic_case(env)
    path = media_dir(rec) / "analysis.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.zeros(20 * 16000, dtype=np.float32), 16000, subtype="PCM_16")
    digest = media.sha256(path)
    store.update(rec["id"], {**rec["data"], "media": {**rec["data"]["media"], "analysis": {
        **rec["data"]["media"]["analysis"], "sha256": digest}}})
    case = store.get(cid)
    legacy = {k: v for k, v in case["data"].items() if k not in ("media_sha256", "taxonomy_version", "source_modes")}
    store.update(cid, legacy)
    request = {"expected_case_revision": store.get(cid)["rev"],
        "expected_annotation_revisions": {s: store.get(f"ann_{t}")["rev"] for s, t in zip("AB", legacy["tasks"])},
        "expected_task_revisions": {s: store.get(t)["rev"] for s, t in zip("AB", legacy["tasks"])},
        "media_sha256": digest, "guideline_version": legacy["guideline_version"],
        "taxonomy_version": TAXONOMY_VERSION, "current_source_reviewed": True,
        "reason": "Synthetic administrator reviewed the current source and both tracks"}
    return rec, tasks, cid, body, request, path


def test_legacy_open_case_requires_explicit_auditable_review_then_can_finalize_and_freeze(env):
    from app.store import Conflict

    store, call, users, _ = env
    _, tasks, cid, body, request, _ = legacy_open_case(env)
    before = {t["id"]: store.get(t["id"]) for t in tasks}
    before.update({f"ann_{t['id']}": store.get(f"ann_{t['id']}") for t in tasks})
    blocked = call("carla", "POST", "/adjudications", json={**body, "case_revision": store.get(cid)["rev"]})
    assert blocked.status_code == 422 and "review-provenance" in blocked.text
    reviewed = call("admin", "POST", f"/admin/adjudication-cases/{cid}/review-provenance", json=request)
    assert reviewed.status_code == 200, reviewed.text
    audit = reviewed.json()["provenance_review"]
    assert audit["owner"] == users["admin"] and audit["created"]
    assert audit["data"]["historical_identity_verified"] is False
    assert audit["data"]["verification_scope"] == "current_source_and_annotations_only"
    assert audit["data"]["media_sha256"] == request["media_sha256"]
    assert audit["data"]["previous_identity"] == {"media_sha256": None, "taxonomy_version": None, "source_modes": None}
    assert audit["data"]["reason"] == request["reason"]
    assert all(store.get(doc_id) == doc for doc_id, doc in before.items())
    with pytest.raises(Conflict):
        store.update(audit["id"], {"reason": "tampered"})
    # Stale pre-review request cannot finalize; reloading is mandatory.
    assert call("carla", "POST", "/adjudications", json={**body, "case_revision": request["expected_case_revision"]}).status_code == 409
    finalized = call("carla", "POST", "/adjudications", json={**body, "case_revision": store.get(cid)["rev"]})
    assert finalized.status_code == 200, finalized.text
    frozen = call("admin", "POST", "/dataset-snapshots", json={"name": "legacy-reviewed-now", "splits": ["dev"]})
    assert frozen.status_code == 200, frozen.text
    assert frozen.json()["data"]["items"][0]["provenance_review"] == audit["id"]


@pytest.mark.parametrize("invalid,expected", [
    ("role", 403), ("unconfirmed", 422), ("reason", 422), ("case_revision", 409),
    ("annotation_revision", 409), ("task_revision", 409), ("hash", 409),
    ("file", 409), ("missing_file", 422), ("guide", 409), ("taxonomy", 422),
])
def test_legacy_provenance_review_rejects_unverified_or_stale_requests_without_writes(env, invalid, expected):
    store, call, _, _ = env
    _, tasks, cid, _, request, path = legacy_open_case(env)
    who = "admin"
    if invalid == "role":
        who = "carla"
    elif invalid == "unconfirmed":
        request["current_source_reviewed"] = False
    elif invalid == "reason":
        request["reason"] = " " * 20
    elif invalid == "case_revision":
        request["expected_case_revision"] += 1
    elif invalid == "annotation_revision":
        request["expected_annotation_revisions"]["A"] += 1
    elif invalid == "task_revision":
        request["expected_task_revisions"]["A"] += 1
    elif invalid == "hash":
        request["media_sha256"] = "d" * 64
    elif invalid == "file":
        path.write_bytes(b"changed synthetic source")
    elif invalid == "missing_file":
        path.unlink()
    elif invalid == "guide":
        request["guideline_version"] = "unknown-guide"
    elif invalid == "taxonomy":
        request["taxonomy_version"] = "unknown-taxonomy"
    before_case, before_tasks = store.get(cid), store.find("task")
    result = call(who, "POST", f"/admin/adjudication-cases/{cid}/review-provenance", json=request)
    assert result.status_code == expected, result.text
    assert store.get(cid) == before_case and store.find("task") == before_tasks
    assert store.find("provenance_review") == [] and store.find("adjudication") == []


def test_legacy_final_is_diagnosed_not_silently_migrated_or_rewritten(env):
    store, call, _, _ = env
    _, _, cid, _, request, _ = legacy_open_case(env)
    case = store.get(cid)
    old = store.create("adjudication", {**case["data"], "case_id": cid, "events": [], "contextual": []}, parent=cid, status="final")
    frozen = store.create("snapshot", {"items": [old["data"]], "snapshot_schema": "historical"}, status="frozen")
    store.update(cid, status="closed")
    request["expected_case_revision"] = store.get(cid)["rev"]
    result = call("admin", "POST", f"/admin/adjudication-cases/{cid}/review-provenance", json=request)
    assert result.status_code == 409 and "legacy final" in result.text.lower()
    rejected = call("admin", "POST", "/dataset-snapshots", json={"name": "legacy-final", "splits": ["dev"]})
    assert rejected.status_code in (409, 422) and "legacy" in rejected.text.lower()
    assert store.get(old["id"]) == old and store.get(frozen["id"]) == frozen


@pytest.mark.parametrize("revision,case_revision,reason", [
    (400, 2, "Synthetic correction review"), (2, 400, "Synthetic correction review"),
    (2, 2, None), (2, 2, "short"), (True, 2, "Synthetic correction review"),
])
def test_snapshot_rejects_invalid_revision_provenance(env, revision, case_revision, reason):
    store, call, _, _ = env
    _, _, cid, body = open_synthetic_case(env)
    first = call("carla", "POST", "/adjudications", json=body).json()
    original = store.get(first["id"])
    bad = store.create("adjudication", {**original["data"], "supersedes": first["id"],
        "decision_revision": revision, "revision_reason": reason}, parent=cid, status="final")
    case = store.get(cid)
    store.update(cid, {**case["data"], "current_adjudication": bad["id"], "adjudication_revision": case_revision})
    result = call("admin", "POST", "/dataset-snapshots", json={"name": "invalid-chain", "splits": ["dev"]})
    assert result.status_code == 409 and "revision" in result.text
    assert store.find("snapshot") == [] and store.get(first["id"]) == original


def test_comparison_returns_only_validated_effective_reference_after_correction(env):
    store, call, _, _ = env
    rec, _, cid, body = open_synthetic_case(env)
    store.update(rec["id"], {**rec["data"], "segments": []})
    run = store.create("run", {"comparison": "synthetic-correction", "system": "mrcd", "result": {
        "events": [], "segments": {}}}, parent=rec["id"], status="succeeded")
    first = call("carla", "POST", "/adjudications", json=body).json()
    correction = {**body, "case_revision": store.get(cid)["rev"], "supersedes": first["id"],
        "revision_reason": "Synthetic reviewer removed the false event", "events": []}
    second = call("carla", "POST", "/adjudications", json=correction).json()
    result = call("rita", "GET", "/experiments/comparisons/synthetic-correction")
    assert result.status_code == 200
    assert [a["id"] for a in result.json()["reference"]] == [second["id"]]
    assert result.json()["reference"][0]["data"]["events"] == []
    assert result.json()["reference_status"] == "available" and store.get(first["id"])


def test_comparison_diagnoses_invalid_reference_instead_of_presenting_historical_finals(env):
    store, call, _, _ = env
    rec, _, cid, body = open_synthetic_case(env)
    store.update(rec["id"], {**rec["data"], "segments": []})
    store.create("run", {"comparison": "synthetic-ambiguous", "system": "mrcd", "result": {
        "events": [], "segments": {}}}, parent=rec["id"], status="succeeded")
    first = call("carla", "POST", "/adjudications", json=body).json()
    store.create("adjudication", store.get(first["id"])["data"], parent=cid, status="final")
    result = call("rita", "GET", "/experiments/comparisons/synthetic-ambiguous")
    assert result.status_code == 200 and result.json()["reference"] == []
    assert result.json()["reference_status"] == "unavailable" and "ambiguous" in result.json()["reference_reason"]


def test_legacy_review_keeps_original_video_hash_separate_from_derived_analysis_identity(env):
    from app import media_dir

    store, call, _, tmp = env
    rec, _, cid, body, request, _ = legacy_open_case(env)
    video = tmp / "synthetic-legacy.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
        "testsrc=size=64x64:rate=2:duration=20", "-f", "lavfi", "-i", "sine=frequency=220:duration=20",
        "-shortest", "-c:v", "libx264", "-c:a", "aac", str(video)], check=True)
    original_hash = media.sha256(video)
    rec = store.get(rec["id"])
    rec["data"].update(sha256=original_hash, original_name="original.mp4")
    directory = media_dir(rec)
    directory.mkdir(parents=True, exist_ok=True)
    original = directory / "original.mp4"
    original.write_bytes(video.read_bytes())
    derived = media.derive(original, directory)
    rec = store.update(rec["id"], {**rec["data"], "media": derived})
    audio_hash = derived["analysis"]["sha256"]
    assert derived["original"]["has_video"] and audio_hash != original_hash
    context = call("admin", "GET", f"/admin/adjudication-cases/{cid}/review-provenance")
    assert context.status_code == 200 and context.json()["recording"]["data"]["sha256"] == original_hash
    assert call("carla", "GET", f"/admin/adjudication-cases/{cid}/review-provenance").status_code == 403
    wrong = call("admin", "POST", f"/admin/adjudication-cases/{cid}/review-provenance", json={**request, "media_sha256": original_hash})
    assert wrong.status_code == 409
    reviewed = call("admin", "POST", f"/admin/adjudication-cases/{cid}/review-provenance", json={**request, "media_sha256": audio_hash})
    assert reviewed.status_code == 200, reviewed.text
    audit = reviewed.json()["provenance_review"]["data"]
    assert audit["media_variant"] == "analysis.wav" and audit["media_sha256"] == audio_hash
    assert audit["recorded_original_sha256"] == original_hash
    assert media.sha256(original) == original_hash and audit["historical_identity_verified"] is False
    assert call("carla", "POST", "/adjudications", json={**body, "case_revision": store.get(cid)["rev"]}).status_code == 200
    frozen = call("admin", "POST", "/dataset-snapshots", json={"name": "legacy-video", "splits": ["dev"]})
    assert frozen.status_code == 200 and frozen.json()["data"]["items"][0]["media_sha256"] == audio_hash


def test_open_case_with_invalid_prior_decision_counter_cannot_create_first_final(env):
    store, call, _, _ = env
    _, _, cid, body = open_synthetic_case(env)
    case = store.get(cid)
    store.update(cid, {**case["data"], "adjudication_revision": 400})
    result = call("carla", "POST", "/adjudications", json={**body, "case_revision": store.get(cid)["rev"]})
    assert result.status_code == 409 and "revision" in result.text
    assert store.find("adjudication") == [] and store.get(cid)["status"] == "open"


def test_snapshot_checks_based_on_provenance_of_predecessors_not_only_head(env):
    store, call, _, _ = env
    _, _, cid, _ = open_synthetic_case(env)
    case = store.get(cid)
    based = {s: {"task": t, "rev": store.get(f"ann_{t}")["rev"]} for s, t in zip("AB", case["data"]["tasks"])}
    data = {**case["data"], "case_id": cid, "events": [], "contextual": [],
        "adjudication_schema": "mrcd-adjudication-v2", "based_on": based}
    root = store.create("adjudication", {**data, "decision_revision": 1, "supersedes": None,
        "based_on": {**based, "A": {"task": "unrelated-task", "rev": 1}}}, parent=cid, status="final")
    head = store.create("adjudication", {**data, "decision_revision": 2, "supersedes": root["id"],
        "revision_reason": "Synthetic explicit correction"}, parent=cid, status="final")
    store.update(cid, {**case["data"], "current_adjudication": head["id"], "adjudication_revision": 2}, status="closed")
    result = call("admin", "POST", "/dataset-snapshots", json={"name": "invalid-predecessor", "splits": ["dev"]})
    assert result.status_code == 409 and "based_on" in result.text
    assert store.find("snapshot") == []


def test_current_source_review_rolls_back_audit_if_case_update_fails(env, monkeypatch):
    store, call, _, _ = env
    _, _, cid, _, request, _ = legacy_open_case(env)
    before, history = store.get(cid), store.history(cid)
    update = Store.update
    def fail_after_review(self, doc_id, *args, **kwargs):
        result = update(self, doc_id, *args, **kwargs)
        if doc_id == cid:
            raise RuntimeError("synthetic legacy review failure")
        return result
    with monkeypatch.context() as patch:
        patch.setattr(Store, "update", fail_after_review)
        with pytest.raises(RuntimeError, match="synthetic legacy review failure"):
            call("admin", "POST", f"/admin/adjudication-cases/{cid}/review-provenance", json=request)
    assert store.find("provenance_review") == [] and store.get(cid) == before and store.history(cid) == history
    assert call("admin", "POST", f"/admin/adjudication-cases/{cid}/review-provenance", json=request).status_code == 200


def test_concurrent_legacy_reviews_use_one_current_case_revision(env):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    store, _, _, _ = env
    _, tasks, cid, _, request, _ = legacy_open_case(env)
    before = [store.get(t["id"]) for t in tasks]
    app = create_app(store)
    barrier = Barrier(3)
    def review(_):
        barrier.wait(timeout=10)
        return TestClient(app).post(f"/api/admin/adjudication-cases/{cid}/review-provenance", json=request,
            headers={"Authorization": "Bearer admin"})
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(review, range(3)))
    assert sorted(r.status_code for r in results) == [200, 409, 409]
    assert len(store.find("provenance_review")) == 1 and [store.get(t["id"]) for t in tasks] == before


def reviewed_legacy_final_with_changed_sources(env):
    """The review fixture contains only synthetic participants and local silence."""
    store, call, _, _ = env
    rec, tasks, cid, body, request, path = legacy_open_case(env)
    review = call("admin", "POST", f"/admin/adjudication-cases/{cid}/review-provenance", json=request)
    assert review.status_code == 200, review.text
    first = call("carla", "POST", "/adjudications", json={**body, "case_revision": store.get(cid)["rev"]})
    assert first.status_code == 200, first.text
    frozen = call("admin", "POST", "/dataset-snapshots", json={"name": "before-renewal", "splits": ["dev"]})
    assert frozen.status_code == 200, frozen.text
    tid = tasks[0]["id"]
    assert call("admin", "POST", f"/review/tasks/{tid}/reopen").status_code == 200
    ann = store.get(f"ann_{tid}")
    assert call("ana", "PATCH", f"/annotations/{ann['id']}", headers={"If-Match": str(ann["rev"])},
        json={"events": [], "contextual": [], "coverage": [[0, 10_000]]}).status_code == 200
    assert call("ana", "POST", f"/review/tasks/{tid}/submit").status_code == 200
    case = store.get(cid)
    request.update(expected_case_revision=case["rev"],
        expected_annotation_revisions={s: store.get(f"ann_{t}")["rev"] for s, t in zip("AB", case["data"]["tasks"])},
        expected_task_revisions={s: store.get(t)["rev"] for s, t in zip("AB", case["data"]["tasks"])},
        reason="Synthetic administrator explicitly reviewed corrected sources again")
    correction = {**body, "case_revision": case["rev"], "based_on": request["expected_annotation_revisions"],
        "events": [], "supersedes": first.json()["id"], "revision_reason": "Explicit corrected A/B adjudication after renewed current-source review"}
    return rec, tasks, cid, request, correction, review.json()["provenance_review"], store.get(first.json()["id"]), frozen.json(), path


@pytest.mark.parametrize("changed_media", [False, True])
def test_reviewed_v2_case_can_renew_review_and_append_correction_without_rewriting_history(env, changed_media):
    store, call, _, _ = env
    rec, tasks, cid, request, correction, old_review, old_final, old_snapshot, path = reviewed_legacy_final_with_changed_sources(env)
    if changed_media:
        audio = np.zeros(20 * 16000, dtype=np.float32)
        audio[0] = 0.25
        sf.write(path, audio, 16000, subtype="PCM_16")
        rec = store.get(rec["id"])
        request["media_sha256"] = media.sha256(path)
        store.update(rec["id"], {**rec["data"], "media": {**rec["data"]["media"], "analysis": {
            **rec["data"]["media"]["analysis"], "sha256": request["media_sha256"]}}})
    denied = call("carla", "POST", "/adjudications", json=correction)
    assert denied.status_code == 409 and "review" in denied.text
    assert len(store.find("adjudication")) == 1 and len(store.find("provenance_review")) == 1
    source_before = {doc["id"]: doc for t in tasks for doc in (store.get(t["id"]), store.get(f"ann_{t['id']}"))}
    renewed = call("admin", "POST", f"/admin/adjudication-cases/{cid}/review-provenance", json=request)
    assert renewed.status_code == 200, renewed.text
    audit = renewed.json()["provenance_review"]
    case = store.get(cid)
    assert case["status"] == "closed" and case["data"]["current_adjudication"] == old_final["id"]
    assert case["data"]["provenance_review"] == old_review["id"]
    assert case["data"]["pending_provenance_review"] == audit["id"]
    assert audit["data"]["previous_review"] == old_review["id"] and audit["data"]["revises_adjudication"] == old_final["id"]
    assert audit["data"]["historical_identity_verified"] is False
    # A renewed review alone is not a new decision and cannot recertify the old one.
    blocked = call("admin", "POST", "/dataset-snapshots", json={"name": "pending-renewal", "splits": ["dev"]})
    assert blocked.status_code == 409 and "pending" in blocked.text
    assert call("carla", "POST", "/adjudications", json=correction).status_code == 409  # stale case CAS
    correction["case_revision"] = case["rev"]
    saved = call("carla", "POST", "/adjudications", json=correction)
    assert saved.status_code == 200, saved.text
    assert saved.json()["revision"] == 2
    assert call("carla", "POST", "/adjudications", json=correction).json() == saved.json()
    head = store.get(saved.json()["id"])
    case = store.get(cid)
    assert head["data"]["provenance_review"] == audit["id"] and head["data"]["media_sha256"] == request["media_sha256"]
    assert case["data"]["provenance_review"] == audit["id"] and not case["data"].get("pending_provenance_review")
    frozen = call("admin", "POST", "/dataset-snapshots", json={"name": "renewed-correction", "splits": ["dev"]})
    assert frozen.status_code == 200, frozen.text
    assert frozen.json()["data"]["items"][0]["source"] == head["id"]
    assert len(store.find("adjudication")) == 2 and len(store.find("provenance_review")) == 2
    assert all(store.get(doc_id) == doc for doc_id, doc in source_before.items())
    assert store.get(old_review["id"]) == old_review and store.get(old_final["id"]) == old_final and store.get(old_snapshot["id"]) == old_snapshot


def test_pending_renewal_becomes_stale_if_sources_change_again_and_requires_new_explicit_review(env):
    store, call, _, _ = env
    _, tasks, cid, request, correction, old_review, old_final, _, _ = reviewed_legacy_final_with_changed_sources(env)
    renewed = call("admin", "POST", f"/admin/adjudication-cases/{cid}/review-provenance", json=request)
    assert renewed.status_code == 200, renewed.text
    pending = renewed.json()["provenance_review"]
    annotation = store.get(f"ann_{tasks[1]['id']}")
    store.update(annotation["id"], annotation["data"], author="synthetic-admin")
    case = store.get(cid)
    current = {s: store.get(f"ann_{t}")["rev"] for s, t in zip("AB", case["data"]["tasks"])}
    rejected = call("carla", "POST", "/adjudications", json={**correction, "case_revision": case["rev"], "based_on": current})
    assert rejected.status_code == 409 and "review" in rejected.text
    # Stale expected A/B tokens cannot create another audit.
    request["expected_case_revision"] = case["rev"]
    stale = call("admin", "POST", f"/admin/adjudication-cases/{cid}/review-provenance", json=request)
    assert stale.status_code == 409 and len(store.find("provenance_review")) == 2
    request["expected_annotation_revisions"] = current
    fresh = call("admin", "POST", f"/admin/adjudication-cases/{cid}/review-provenance", json=request)
    assert fresh.status_code == 200, fresh.text
    latest = fresh.json()["provenance_review"]
    assert latest["data"]["previous_review"] == pending["id"]
    correction.update(case_revision=store.get(cid)["rev"], based_on=current)
    assert call("carla", "POST", "/adjudications", json=correction).status_code == 200
    assert store.get(pending["id"]) == pending and store.get(old_review["id"]) == old_review and store.get(old_final["id"]) == old_final


def test_concurrent_closed_v2_renewals_have_one_cas_winner_and_no_decision_write(env):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    store, _, _, _ = env
    _, _, cid, request, _, old_review, old_final, _, _ = reviewed_legacy_final_with_changed_sources(env)
    app = create_app(store)
    barrier = Barrier(3)
    def renew(_):
        barrier.wait(timeout=10)
        return TestClient(app).post(f"/api/admin/adjudication-cases/{cid}/review-provenance", json=request,
            headers={"Authorization": "Bearer admin"})
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(renew, range(3)))
    assert sorted(r.status_code for r in results) == [200, 409, 409]
    assert len(store.find("provenance_review")) == 2 and len(store.find("adjudication")) == 1
    assert store.get(cid)["data"]["current_adjudication"] == old_final["id"]
    assert store.get(cid)["data"]["provenance_review"] == old_review["id"]


@pytest.mark.parametrize("stage", ["renewal", "correction"])
def test_closed_v2_renewal_and_correction_roll_back_without_changing_saved_history(env, monkeypatch, stage):
    store, call, _, _ = env
    _, _, cid, request, correction, old_review, old_final, old_snapshot, _ = reviewed_legacy_final_with_changed_sources(env)
    if stage == "correction":
        assert call("admin", "POST", f"/admin/adjudication-cases/{cid}/review-provenance", json=request).status_code == 200
        correction["case_revision"] = store.get(cid)["rev"]
    case, history = store.get(cid), store.history(cid)
    reviews, decisions = store.find("provenance_review"), store.find("adjudication")
    update = Store.update
    def fail_after_update(self, doc_id, *args, **kwargs):
        result = update(self, doc_id, *args, **kwargs)
        if doc_id == cid:
            raise RuntimeError("synthetic v2 renewal/correction failure")
        return result
    with monkeypatch.context() as patch:
        patch.setattr(Store, "update", fail_after_update)
        with pytest.raises(RuntimeError, match="synthetic v2 renewal/correction failure"):
            if stage == "renewal":
                call("admin", "POST", f"/admin/adjudication-cases/{cid}/review-provenance", json=request)
            else:
                call("carla", "POST", "/adjudications", json=correction)
    assert store.get(cid) == case and store.history(cid) == history
    assert store.find("provenance_review") == reviews and store.find("adjudication") == decisions
    assert store.get(old_review["id"]) == old_review and store.get(old_final["id"]) == old_final and store.get(old_snapshot["id"]) == old_snapshot


def test_current_review_does_not_make_an_unversioned_historical_final_eligible_for_renewal(env):
    store, call, _, _ = env
    _, _, cid, _, request, _ = legacy_open_case(env)
    reviewed = call("admin", "POST", f"/admin/adjudication-cases/{cid}/review-provenance", json=request)
    assert reviewed.status_code == 200
    case = store.get(cid)
    legacy = store.create("adjudication", {**case["data"], "case_id": cid, "events": [], "contextual": []}, parent=cid, status="final")
    store.update(cid, {**case["data"], "current_adjudication": legacy["id"], "adjudication_revision": 1}, status="closed")
    request["expected_case_revision"] = store.get(cid)["rev"]
    before = store.get(cid)
    denied = call("admin", "POST", f"/admin/adjudication-cases/{cid}/review-provenance", json=request)
    assert denied.status_code == 409 and "unverified historical" in denied.text
    assert store.get(cid) == before and len(store.find("provenance_review")) == 1 and store.get(legacy["id"]) == legacy


def evaluation_http_fixture(env):
    store, _, _, _ = env
    rec = store.create("recording", {"speaker_id": "canonical-evaluation-speaker", "segments": [
        {"id": "s0", "core_start_ms": 0, "core_end_ms": 30_000},
        {"id": "s1", "core_start_ms": 30_000, "core_end_ms": 60_000}]}, status="ready")
    snap = store.create("snapshot", {"name": "synthetic-evaluation", "sha256": "e" * 64,
        "snapshot_schema": "mrcd-snapshot-v2", "reference_kind": "human_adjudicated",
        "taxonomy_version": "mrcd-taxonomy-v1", "coverage": {rec["id"]: [[0, 60_000]]},
        "recording_speakers": {rec["id"]: "canonical-evaluation-speaker"}, "items": [{
            "recording_id": rec["id"], "speaker_id": "canonical-evaluation-speaker", "split": "dev", "start_ms": 0, "end_ms": 60_000,
            "events": [{"start_ms": 1000, "end_ms": 1400, "label": "filler_word"},
                       {"start_ms": 40_000, "end_ms": 40_400, "label": "filler_word"},
                       {"start_ms": 10_000, "end_ms": 12_000, "label": None, "decision": "not_evaluable"}],
            "contextual": []}]}, status="frozen")
    run = store.create("run", {"system": "mrcd", "result": {"segments": {"s0": "succeeded", "s1": "failed"},
        "events": [{"start_ms": 1000, "end_ms": 1400, "label": "filler_word"},
                   {"start_ms": 10_500, "end_ms": 11_000, "label": "filler_word"}]}}, parent=rec["id"], status="partial")
    return rec, snap, run


def test_http_evaluation_uses_human_masks_partial_cores_and_frozen_speakers(env):
    store, call, _, _ = env
    rec, snap, run = evaluation_http_fixture(env)
    before = store.get(snap["id"])
    store.update(rec["id"], {**rec["data"], "speaker_id": "later-recording-metadata-must-not-relabel-reference"})
    response = call("rita", "POST", "/evaluation-runs", json={"snapshot_id": snap["id"], "systems": {"M0": [run["id"]]}, "bootstrap": 10})
    assert response.status_code == 200, response.text
    data = response.json()["data"]
    result = data["results"]["M0"]
    assert result["evaluable_ms"] == 28_000 and result["micro"]["disfluency"]["tp"] == 1
    assert result["micro"]["disfluency"]["fn"] == result["micro"]["disfluency"]["fp"] == 0
    assert result["exclusions"]["reference"]["count"] == result["exclusions"]["prediction"]["count"] == 1
    assert data["reference_view"]["recording_speakers"] == {rec["id"]: "canonical-evaluation-speaker"}
    assert data["protocol"]["version"] == "mrcd-evaluation-v2" and data["protocol"]["boundary_policy"] == "fully_contained_v2"
    assert data["bootstrap_ci"]["M0"]["n_speakers"] == 1 and store.get(snap["id"]) == before


@pytest.mark.parametrize("duplicate", ["same-id", "same-system-recording", "across-names"])
def test_http_evaluation_rejects_ambiguous_selection_before_writing(env, duplicate):
    store, call, _, _ = env
    _, snap, run = evaluation_http_fixture(env)
    if duplicate == "same-system-recording":
        another = store.create("run", run["data"], parent=run["parent"], status="partial")
        systems = {"M0": [run["id"], another["id"]]}
    elif duplicate == "across-names":
        systems = {"M0": [run["id"]], "alias": [run["id"]]}
    else:
        systems = {"M0": [run["id"], run["id"]]}
    response = call("rita", "POST", "/evaluation-runs", json={"snapshot_id": snap["id"], "systems": systems, "bootstrap": 0})
    assert response.status_code == 422 and store.find("evaluation") == []


def test_http_failed_run_is_unavailable_and_no_common_pair_has_no_fake_delta(env):
    store, call, _, _ = env
    rec, snap, run = evaluation_http_fixture(env)
    failed = store.create("run", {"system": "rules", "result": {"segments": {}, "events": []}}, parent=rec["id"], status="failed")
    response = call("rita", "POST", "/evaluation-runs", json={"snapshot_id": snap["id"],
        "systems": {"M0": [run["id"]], "rules": [failed["id"]]}, "bootstrap": 5})
    assert response.status_code == 200, response.text
    data = response.json()["data"]
    assert data["results"]["rules"]["reason"] == "no_evaluable_coverage"
    assert data["results"]["rules"]["micro"]["disfluency"]["fn"] is None
    assert data["paired_results"]["rules-M0"]["delta_f1"] is None
    assert data["paired_results"]["rules-M0"]["reason"] == "no_evaluable_coverage"


def test_snapshot_freezes_canonical_speaker_map_for_future_evaluation(env):
    from core import evaluation

    store, call, _, _ = env
    rec, _, _, body = open_synthetic_case(env)
    project = store.create("project", {"speaker_aliases": {"synthetic-reviewer": "canonical-1"}, "splits": {"canonical-1": "dev"}})
    store.update(rec["id"], {**rec["data"], "project": project["id"]})
    assert call("carla", "POST", "/adjudications", json=body).status_code == 200
    snap = call("admin", "POST", "/dataset-snapshots", json={"name": "canonical-map", "splits": ["dev"]}).json()
    assert snap["data"]["recording_speakers"] == {rec["id"]: "canonical-1"}
    store.update(project["id"], {"speaker_aliases": {"synthetic-reviewer": "changed-later"}})
    view = evaluation.reference_view(snap["data"], {rec["id"]: "changed-later"})
    assert view["recording_speakers"] == {rec["id"]: "canonical-1"}


def render_evaluation_html(data):
    from app import ROOT

    source = (ROOT / "app/static/app.js").read_text()
    function = "function evalTable(x) {" + source.split("function evalTable(x) {", 1)[1].split("\nasync function comparison", 1)[0]
    js = "const SYSTEM={}, CLASSES=[['filler_word']]; const esc=x=>String(x); const num=x=>x==null?'—':String(x); const pct=num; const badge=esc;\n"
    return subprocess.run(["node", "-e", js + function + "\nconsole.log(evalTable(" + json.dumps(data) + "));"], capture_output=True, text=True, check=True).stdout


def test_evaluation_client_renders_unavailable_null_counts_reasons_and_separate_paired_results():
    data = {"created": "2026-10-22T00:00:00Z", "snapshot_id": "synthetic", "results": {
        "failed": {"micro": {"disfluency": {"precision": None, "recall": None, "f1": None}}, "per_class": {"filler_word": {"f1": None, "support": None}},
            "evaluable_ms": 0, "reason": "no_evaluable_coverage", "n_ref": None, "n_hyp": None, "n_uncertain": None, "false_alarms_per_min": None}},
        "paired_results": {"failed-good": {"systems": ["good", "failed"], "evaluable_ms": 0, "delta_f1": None, "reason": "no_evaluable_coverage",
            "results": {}, "bootstrap_ci": {"low": None, "high": None, "requested": 5, "valid": 0, "omitted": 5, "n_speakers": 0}}}}
    rendered = render_evaluation_html(data)
    assert ">null<" not in rendered and "n=0" not in rendered
    assert "no_evaluable_coverage" in rendered and "failed-good" in rendered


def test_symmetric_comparison_intersects_explicit_inference_availability(env):
    store, call, _, _ = env
    rec = store.create("recording", {"segments": [{"id": "s0", "core_start_ms": 0, "core_end_ms": 6000}]}, status="ready")
    for name, spans in (("a", [[0, 2000]]), ("b", [[4000, 6000]])):
        store.create("run", {"system": name, "comparison": "synthetic-explicit-coverage", "result": {
            "segments": {"s0": "succeeded"}, "coverage": spans,
            "events": [{"label": "filler_word", "start_ms": 1000, "end_ms": 1500}]}}, parent=rec["id"], status="succeeded")
    result = call("rita", "GET", "/experiments/comparisons/synthetic-explicit-coverage")
    assert result.status_code == 200, result.text
    data = result.json()
    assert data["symmetric_agreement"]["a~b"]["filler_word"] is None
    assert data["symmetric_coverage"]["a~b"]["evaluable_ms"] == 0
    assert data["symmetric_coverage"]["a~b"]["reason"] == "no_evaluable_coverage"


@pytest.mark.parametrize("protocol", [{"coverage": {}}, {"version": "invented"}, {"iou_threshold": 2}, {"tolerance_ms": -1}])
def test_http_evaluation_rejects_protected_or_invalid_protocol_before_write(env, protocol):
    store, call, _, _ = env
    _, snap, run = evaluation_http_fixture(env)
    result = call("rita", "POST", "/evaluation-runs", json={"snapshot_id": snap["id"],
        "systems": {"M0": [run["id"]]}, "bootstrap": 0, "protocol": protocol})
    assert result.status_code == 422 and store.find("evaluation") == []


@pytest.mark.parametrize("first_output", [
    {"decision": "uncertain"}, {"decision": "not_evaluable"}, {"decision": "event", "is_disfluent": False},
])
def test_symmetric_model_comparison_never_treats_model_decisions_as_human_masks(env, first_output):
    from core.evaluation import coverage_duration

    store, call, _, _ = env
    rec = store.create("recording", {"segments": [{"id": "s0", "core_start_ms": 0, "core_end_ms": 6000}]}, status="ready")
    at = lambda start: {"label": "filler_word", "start_ms": start, "end_ms": start + 500}
    outputs = {"a": [{**at(1000), **first_output}, at(3000)], "b": [at(1000), at(3000)]}
    results = []
    for comparison, order in (("synthetic-a-first", ("a", "b")), ("synthetic-b-first", ("b", "a"))):
        for name in order:
            store.create("run", {"system": name, "comparison": comparison, "result": {
                "events": outputs[name], "segments": {"s0": "succeeded"}, "coverage": [[0, 6000]]}},
                parent=rec["id"], status="succeeded")
        response = call("rita", "GET", f"/experiments/comparisons/{comparison}")
        assert response.status_code == 200, response.text
        result = response.json()
        key = "~".join(order)
        assert result["symmetric_agreement"][key]["filler_word"] == pytest.approx(2 / 3)
        coverage = result["symmetric_coverage"][key]
        assert coverage["coverage"] == {rec["id"]: [[0, 6000]]}
        assert coverage["evaluable_ms"] == coverage_duration(coverage["coverage"]) == 6000
        results.append(result["symmetric_agreement"][key])
    assert results[0] == results[1]


def test_evaluation_saved_http_history_renderer_roundtrip_preserves_v2_and_legacy_identity(env):
    store, call, _, _ = env
    rec, snapshot, run = evaluation_http_fixture(env)
    failed = store.create("run", {"system": "rules", "result": {"segments": {}, "events": []}}, parent=rec["id"], status="failed")
    response = call("rita", "POST", "/evaluation-runs", json={"snapshot_id": snapshot["id"],
        "systems": {"M0": [run["id"]], "rules": [failed["id"]]}, "bootstrap": 3})
    assert response.status_code == 200, response.text
    saved = response.json()
    immutable_before, history_before = store.get(saved["id"]), store.history(saved["id"])
    legacy = store.create("evaluation", {"snapshot_id": snapshot["id"], "systems": {"legacy": [run["id"]]},
        "results": {"legacy": {"micro": {"disfluency": {"precision": None, "recall": None, "f1": None}},
            "per_class": {"filler_word": {"f1": None, "support": 0}}, "n_ref": 0, "n_hyp": 0, "n_uncertain": 0,
            "false_alarms_per_min": None}}, "bootstrap_ci": {"legacy": {"low": None, "high": None, "n_boot": 0, "n_speakers": 0}},
        "prior_evaluations_on_snapshot": 0}, status="final")
    history_response = call("rita", "GET", "/experiments")
    assert history_response.status_code == 200, history_response.text
    history = {entry["id"]: entry for entry in history_response.json()["evaluations"]}
    reloaded = history[saved["id"]]
    for key in ("protocol", "paired_results", "results", "bootstrap_ci", "availability", "missing_runs", "reference_view", "snapshot_sha256", "paired_ci_aliases_omitted"):
        assert reloaded[key] == saved["data"][key]
    fresh_html = render_evaluation_html({**saved["data"], "created": saved["created"], "prior": saved["data"]["prior_evaluations_on_snapshot"]})
    reloaded_html = render_evaluation_html(reloaded)
    assert fresh_html == reloaded_html and "Diferencias pareadas ·" in reloaded_html
    assert "mrcd-evaluation-v2" in reloaded_html and "no_evaluable_coverage" in reloaded_html
    assert "protocolo histórico, sin recalcular" not in reloaded_html
    assert "protocol" not in history[legacy["id"]] and "paired_results" not in history[legacy["id"]]
    legacy_html = render_evaluation_html(history[legacy["id"]])
    assert "protocolo histórico, sin recalcular" in legacy_html and "Diferencias pareadas ·" not in legacy_html
    assert store.get(saved["id"]) == immutable_before and store.history(saved["id"]) == history_before
    assert store.get(legacy["id"]) == legacy


def test_submit_requires_marks_to_touch_the_fragment(env):
    store, call, users, tmp = env
    sf.write(tmp / "edge.wav", (0.1 * np.sin(np.arange(16000 * 30) / 9)).astype(np.float32), 16000)
    rid = upload(call, "rita", tmp / "edge.wav")
    run_worker(store)
    tid = call(
        "rita",
        "POST",
        "/review/tasks",
        json={"recording_id": rid, "start_ms": 10_000, "end_ms": 20_000, "split": "dev", "assignee": users["ana"]},
    ).json()["id"]
    ann = call("ana", "GET", f"/review/tasks/{tid}").json()["annotation"]
    marks = [
        {"start_ms": 6_000, "end_ms": 6_500, "label": "filler_word"},  # context only -> rejected
        {"start_ms": 9_800, "end_ms": 10_300, "label": "repetition"},  # crosses the edge -> fine
        {"start_ms": 21_000, "end_ms": 21_400, "label": "block"},
    ]  # context only -> rejected
    r = call(
        "ana",
        "PATCH",
        f"/annotations/{ann['id']}",
        headers={"If-Match": str(ann["rev"])},
        json={"events": marks, "coverage": [[10_000, 20_000]]},
    )
    bad = call("ana", "POST", f"/review/tasks/{tid}/submit")
    assert bad.status_code == 422 and bad.json()["detail"]["unreviewed_gaps_ms"] == []
    assert sorted(e["start_ms"] for e in bad.json()["detail"]["outside_context"]) == [6_000, 21_000]
    call(
        "ana",
        "PATCH",
        f"/annotations/{ann['id']}",
        headers={"If-Match": str(r.json()["rev"])},
        json={"events": [marks[1]], "coverage": [[10_000, 20_000]]},
    )
    assert call("ana", "POST", f"/review/tasks/{tid}/submit").json() == {"status": "submitted"}
