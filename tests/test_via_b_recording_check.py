"""Tests of the optional technical-validation kit (synthetic data only; no real recordings, no engine output)."""

import json
import wave
from pathlib import Path

import numpy as np
import pytest

from mrcd.av.gate import FaceSeries
from mrcd.recording_check import freeze as fz
from mrcd.recording_check import stats, technical, verify_recording as vr

SR = 16000


def test_segment_grid_and_wilson_and_halfwidth():
    assert stats.segments(10.0) == [(0.0, 3.0), (3.0, 6.0), (6.0, 9.0)]
    assert stats.segments(10.6)[-1] == (9.0, 10.6)
    w = stats.wilson(8, 10)
    assert w["point"] == 0.8 and w["lo"] < 0.8 < w["hi"]
    assert stats.wilson(0, 0)["point"] is None  # no data is not a zero rate
    assert stats.cluster_halfwidth(0.8, 10, 100, 0.3) == pytest.approx(0.137, abs=2e-3)


def test_speaker_bootstrap_deterministic_and_brackets_point():
    counts = {f"s{i}": np.array([8.0 + (i % 3), 10.0]) for i in range(10)}
    stat = lambda v: v[0] / v[1]  # noqa: E731
    a = stats.speaker_bootstrap(counts, stat, B=500, seed=1)
    assert a == stats.speaker_bootstrap(counts, stat, B=500, seed=1)
    assert a["lo"] <= a["point"] <= a["hi"]
    assert stats.speaker_bootstrap({}, stat)["point"] is None


def _signal(seconds=60.0, clap_at=(3.0, 55.0), peak=0.3):
    rng = np.random.default_rng(0)
    x = 0.003 * rng.normal(size=int(SR * seconds))
    for t in clap_at:
        i = int(t * SR)
        x[i : i + 80] += 0.9 * np.hanning(80)
    t = np.arange(len(x)) / SR
    x += peak * 0.2 * np.sin(2 * np.pi * 180 * t) * (np.sin(2 * np.pi * 0.5 * t) > 0)
    return x


def test_level_stats_and_clipping_detected():
    x = _signal()
    lv = vr.level_stats(x, SR)
    assert lv["available"] and -3 < lv["peak_dbfs"] < 0 and lv["noise_floor_dbfs"] < -45
    loud = 3.0 * np.sin(2 * np.pi * 180 * np.arange(SR * 5) / SR)
    clipped = np.clip(loud, -1, 1)
    assert (
        vr.level_stats(clipped, SR)["clipped_fraction"]
        > vr.REQUIREMENTS["max_clipped_fraction"]
    )
    assert vr.level_stats(np.zeros(0), SR)["available"] is False


def test_claps_found_and_missing_clap_is_none_not_zero():
    c = vr.find_claps(_signal(), SR, search_s=45)
    assert c["start"] == pytest.approx(3.0, abs=0.02) and c["end"] == pytest.approx(
        55.0, abs=0.02
    )
    quiet = 0.003 * np.random.default_rng(1).normal(size=SR * 60)
    c2 = vr.find_claps(quiet, SR, search_s=45)
    assert c2["start"] is None and c2["reason"].startswith("clap_not_found")


def test_clap_offset_sign_and_resolution():
    fps = 25.0
    motion = np.full(500, 0.05)
    motion[3 * 25 + 3] = 5.0  # visual event 3 frames (120 ms) after the sound
    r = vr.clap_offset_ms(3.0, motion, fps)
    assert r["offset_ms"] == pytest.approx(120.0) and r[
        "resolution_ms"
    ] == pytest.approx(40.0)
    assert vr.clap_offset_ms(None, motion, fps)["offset_ms"] is None
    flat = np.full(500, 0.05)
    assert vr.clap_offset_ms(3.0, flat, fps)["reason"] == "no_visual_transient"


def _face_series(n, fps=25.0, present=True, lag=2, seed=0):
    rng = np.random.default_rng(seed)
    env = np.convolve(rng.normal(size=n), np.ones(5) / 5, mode="same")
    env = (env - env.mean()) / env.std()
    lip = (np.roll(env, lag) + 0.1 * rng.normal(size=n))[None]
    pres = np.full((1, n), present)
    shape = np.tile([0.4, 0.5, 0.3, 0.7, 0.2], (1, n, 1)) + 0.002 * rng.normal(
        size=(1, n, 5)
    )
    if not present:
        lip = np.full_like(lip, np.nan)
    return FaceSeries(fps, lip, pres, pres.copy(), shape, env)


def test_gate_summary_counts_only_and_missing_face_is_a_measured_zero_with_reason():
    n = 25 * 120
    good = vr.gate_summary(_face_series(n), n / 25.0)
    assert good["admitted_fraction"] > 0.8 and good["video_sync_reason"] == "ok"
    none = vr.gate_summary(_face_series(n, present=False), n / 25.0)
    assert (
        none["admitted"] == 0 and "face_not_visible_enough" in none["primary_reasons"]
    )


def test_verdicts_unknown_is_never_a_pass():
    rep = {
        "duration_s": 200.0,
        "audio": {"sample_rate": 48000},
        "levels": {"available": False},
    }
    v = vr.verdicts(rep)
    assert v["duration"] is True and v["audio_sample_rate"] is True
    rep["video"] = {"height": 720, "fps": 25.0}
    v = vr.verdicts(rep)
    assert (
        v["admitted_fraction"] is None and v["clap_offset"] is None
    )  # skipped checks stay unknown
    rep["verdicts"] = v
    assert not all(x is True for x in v.values())


def _write_wav(path: Path, x: np.ndarray, sr=SR):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


def test_verify_file_on_audio_only_stores_no_content(tmp_path):
    p = tmp_path / "VB_S01_PRES1.wav"
    _write_wav(p, _signal())
    rep = vr.verify_file(p, run_face=False)
    assert rep["stores_content"] is False and len(rep["sha256"]) == 64
    assert rep["audio"]["sample_rate"] == SR and rep["audio"]["channels"] == 1
    assert rep["duration_s"] == pytest.approx(60.0, abs=0.1)
    assert rep["verdicts"]["duration"] is False  # 60 s < 120 s minimum
    assert rep["all_checks_passed"] is False
    blob = json.dumps(rep, default=float)
    assert "samples" not in blob and "transcript" not in blob
    assert max(len(str(v)) for v in rep.values()) < 2000  # numbers only, nothing bulky


def _report(rec, adm, tot, ok_sync=True, face=0.95):
    return {
        "file": rec + ".mp4",
        "duration_s": 200.0,
        "verdicts": {
            "audio_peak_level": True,
            "audio_clipping": True,
            "audio_noise_floor": True,
            "audio_sample_rate": True,
        },
        "levels": {"peak_dbfs": -6.0, "noise_floor_dbfs": -60.0},
        "clap_sync": {"start_offset_ms": 40.0},
        "face_gate": {
            "admitted": adm,
            "segments": tot,
            "admitted_fraction": adm / tot,
            "video_sync_reason": "ok"
            if ok_sync
            else "video_offset_out_of_range_or_imprecise",
            "video_offset_ms": 40.0,
            "face_present_fraction": face,
            "primary_reasons": {"admitted": adm, "face_not_visible_enough": tot - adm},
        },
    }


def test_summarize_pools_by_speaker_decides_and_never_claims_accuracy():
    reports = [_report(f"VB_S{k:02d}_PRES1", 60 + k, 66) for k in range(1, 11)]
    runs = [
        {
            "recording_id": r["file"][:-4],
            "status": "ok",
            "wall_s": 50.0,
            "audio_s": 200.0,
            "events_total": 12,
        }
        for r in reports
    ]
    res = technical.summarize(reports, runs)
    assert (
        res["accuracy_claim_allowed"] is False and "No human labels" in res["statement"]
    )
    assert (
        res["design"]["n_speakers"] == 10
        and res["design"]["meets_min_speakers"] is True
    )
    t1 = res["T1_gate_admission"]
    assert t1["admitted_fraction"]["point"] == pytest.approx(
        sum(60 + k for k in range(1, 11)) / 660
    )
    assert t1["decision"].startswith("cumple")
    assert res["T5_robustness"]["real_time_factor_wall_over_audio"][
        "p50"
    ] == pytest.approx(0.25)
    flat = json.dumps(res, default=float).lower()
    for forbidden in ('"f1"', '"precision"', '"recall"', "accuracy_value"):
        assert forbidden not in flat


def test_summarize_missing_measurements_are_unavailable_not_zero_and_failures_lower_t5():
    reports = [_report(f"VB_S{k:02d}_PRES1", 20, 66) for k in range(1, 4)]
    reports.append(
        {"file": "VB_S04_PRES1.mp4", "duration_s": 200.0, "verdicts": {}, "levels": {}}
    )  # no face gate
    res = technical.summarize(reports)
    assert res["T1_gate_admission"]["recordings_without_gate_result"] == [
        "VB_S04_PRES1"
    ]
    assert res["T1_gate_admission"]["decision"] == "no cumple"
    assert res["T5_robustness"]["available"] is False
    runs = [
        {"recording_id": f"VB_S0{k}_PRES1", "status": "ok" if k > 1 else "failed"}
        for k in range(1, 5)
    ]
    t5 = technical.summarize(reports, runs)["T5_robustness"]
    assert t5["analysis_success"]["point"] == 0.75 and t5["decision"] == "no cumple"
    assert t5["failures"] == [{"recording_id": "VB_S01_PRES1", "status": "failed"}]
    with pytest.raises(ValueError):
        technical.parse_recording_id(
            "Maria_Perez_1"
        )  # real names never pass the pseudonymous id check


def test_freeze_detects_any_change(tmp_path):
    (tmp_path / "a.md").write_text("pre-registration", encoding="utf-8")
    (tmp_path / "b.py").write_text("x = 1\n", encoding="utf-8")
    out = tmp_path / "freeze.json"
    rec = fz.freeze(out, root=tmp_path, files=["a.md", "b.py"])
    assert fz.verify_freeze(out, root=tmp_path) == [] and rec["config_hash"]
    (tmp_path / "b.py").write_bytes(
        b"x = 1\r\n"
    )  # CRLF checkout of the same text stays intact
    assert fz.verify_freeze(out, root=tmp_path) == []
    (tmp_path / "b.py").write_text("x = 2\n", encoding="utf-8")
    assert any("b.py" in m for m in fz.verify_freeze(out, root=tmp_path))
    with pytest.raises(fz.FreezeError):
        fz.require_frozen(out, root=tmp_path)
    assert fz.verify_freeze(tmp_path / "missing.json", root=tmp_path)[0].startswith(
        "freeze record missing"
    )
    with pytest.raises(fz.FreezeError):
        fz.freeze(tmp_path / "x.json", root=tmp_path, files=["nope.md"])


def test_committed_freeze_record_is_intact_if_present():
    path = fz.REPO_ROOT / "docs" / "via-b" / "PREREGISTRO_TECNICO.freeze.json"
    if not path.is_file():
        pytest.skip("preregistration not frozen yet")
    assert fz.verify_freeze(path) == [], (
        "frozen file changed: write a dated amendment and a new freeze tag"
    )


def test_verify_file_video_only_reports_geometry_and_leaves_audio_checks_out(tmp_path):
    cv2 = pytest.importorskip("cv2")
    p = tmp_path / "VB_S02_ESP1.mp4"
    w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (160, 120))
    if not w.isOpened():
        pytest.skip("no mp4 writer available")
    for k in range(75):
        f = np.zeros((120, 160, 3), np.uint8)
        f[40:80, (k * 2) % 120 : (k * 2) % 120 + 30] = 255
        w.write(f)
    w.release()
    rep = vr.verify_file(p, run_face=False)
    assert rep["audio"] is None and rep["video"]["height"] == 120
    assert rep["video"]["fps"] == pytest.approx(25.0)
    assert rep["verdicts"]["video_resolution"] is False  # 120 px < 720
    assert (
        rep["verdicts"]["admitted_fraction"] is None
    )  # face gate skipped: unknown, never a pass
    assert rep["all_checks_passed"] is False
