"""Microphone-check kit (docs/microphone-check/mic_check.py), exercised with a simulated microphone only.

No test opens a real microphone: sounddevice is made to raise if anything imports it.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
KIT = REPO / "docs" / "microphone-check"


@pytest.fixture(scope="module")
def kit():
    spec = importlib.util.spec_from_file_location(
        "mic_check_under_test", KIT / "mic_check.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(autouse=True)
def no_real_microphone(monkeypatch):
    class Blocker:
        def __getattr__(self, name):
            raise AssertionError(
                "a test tried to use sounddevice (the physical microphone)"
            )

    monkeypatch.setitem(sys.modules, "sounddevice", Blocker())


def cfg():
    return json.loads((KIT / "config.json").read_text(encoding="utf-8"))


def feed(
    meter,
    seconds_silence_start=3,
    speech=4,
    seconds_silence_end=3,
    level=0.2,
    clip=False,
):
    sr = 16000
    parts = [np.zeros(sr * seconds_silence_start, np.float32) + 1e-4]
    t = np.arange(sr * speech) / sr
    sp = (level * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    if clip:
        sp[:200] = 1.0
    parts += [sp, np.zeros(sr * seconds_silence_end, np.float32) + 1e-4]
    audio = np.concatenate(parts)
    for i in range(0, len(audio), 1600):
        meter.push(audio[i : i + 1600])


def test_meter_reports_edges_level_and_a_pass_verdict(kit):
    m = kit.LevelMeter()
    feed(m)
    th = cfg()["thresholds"]
    rep = m.report(th)
    assert rep["measured"] and rep["signal_present"]
    assert (
        2.5 <= rep["leading_silence_s"] <= 3.2
        and 2.5 <= rep["trailing_silence_s"] <= 3.2
    )
    assert rep["clipped_samples"] == 0 and rep["peak_dbfs"] == pytest.approx(
        -14.0, abs=0.5
    )
    checks = kit.verdicts(rep, wall_s=10.0, dropped_blocks=0, th=th)
    assert {c["status"] for c in checks.values()} == {"PASS"}


def test_clipping_and_dropped_blocks_are_flagged(kit):
    m = kit.LevelMeter()
    feed(m, clip=True)
    th = cfg()["thresholds"]
    rep = m.report(th)
    assert rep["clipped_samples"] >= 200
    checks = kit.verdicts(rep, wall_s=10.0, dropped_blocks=9, th=th)
    assert checks["clipping"]["status"] in ("WARN", "FAIL")
    assert checks["dropped_blocks"]["status"] == "FAIL"


def test_silent_device_is_a_signal_failure_not_a_pass(kit):
    m = kit.LevelMeter()
    m.push(np.zeros(16000 * 5, np.float32))
    th = cfg()["thresholds"]
    checks = kit.verdicts(m.report(th), wall_s=5.0, dropped_blocks=0, th=th)
    assert checks["signal"]["status"] == "FAIL"
    empty = kit.verdicts(kit.LevelMeter().report(th), None, 0, th)
    assert empty["signal"]["status"] == "FAIL"


def test_raw_audio_is_not_kept_unless_opted_in(kit):
    assert kit.MeteredMicrophone().recorder is None
    assert kit.MeteredMicrophone(keep_audio=True).recorder is not None


def test_run_folder_is_new_and_never_reused(kit, tmp_path, monkeypatch):
    class Fixed:
        @staticmethod
        def now(tz=None):
            from datetime import datetime, timezone

            return datetime(2026, 10, 31, 12, 0, 0, tzinfo=timezone.utc)

    monkeypatch.setattr(kit, "datetime", Fixed)
    first = kit.new_run_dir(tmp_path)
    assert first.name == "20261031T120000Z"
    with pytest.raises(FileExistsError):
        kit.new_run_dir(tmp_path)


def test_rows_drop_the_text_unless_requested(kit):
    rows = [{"timestamp": 1.0, "clase": "fluent", "texto_ventana": "private words"}]
    assert "texto_ventana" not in kit.minimal_rows(rows, keep_text=False)[0]
    assert kit.minimal_rows(rows, keep_text=True)[0]["texto_ventana"] == "private words"


def test_results_template_is_marked_not_performed():
    text = (KIT / "RESULTS_TEMPLATE.md").read_text(encoding="utf-8")
    assert "**STATE: NO REALIZADA**" in text
    assert "REALIZADA - PASS" in text  # the allowed values after the user completes it
    cfg_text = (KIT / "config.json").read_text(encoding="utf-8")
    assert '"keep_audio": false' in cfg_text and '"keep_text": false' in cfg_text


def _write_wav(path, seconds=20):
    t = np.arange(16000 * seconds) / 16000
    x = (0.2 * np.sin(2 * np.pi * 200 * t) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(x.tobytes())


def _patched_engine(monkeypatch):
    """Replace LiveEngine by one built from fake parts so the test needs no checkpoint and no Whisper."""
    import scripts.live_demo as live_demo
    from mrcd.extractors.linguistic import Word
    from tests.test_live_stream_partitions import LABELS, fake_score

    class SecretASR:
        words: list = []

        def merge_recent(self, previous, audio, begin):
            return [Word("secretword", "secretword", begin + 1.0, begin + 1.4, 0.9, "")]

    real_cls = live_demo.LiveEngine

    def factory(checkpoint, whisper, mic=None, config=None, **kw):
        return real_cls.for_parts(
            fake_score,
            LABELS,
            mic=mic,
            asr=SecretASR(),
            config=config,
            ling_dim=12,
            tick_s=0.01,
        )

    monkeypatch.setattr(live_demo, "LiveEngine", factory)


@pytest.mark.parametrize("opt_in", [False, True])
def test_full_simulated_run_writes_a_new_folder_and_respects_privacy(
    kit, tmp_path, monkeypatch, opt_in
):
    _patched_engine(monkeypatch)
    wav_path = tmp_path / "clip.wav"
    _write_wav(wav_path)
    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_bytes(b"x")
    out_root = tmp_path / "out"
    args = [
        "--simulate-wav",
        str(wav_path),
        "--simulate-speed",
        "20",
        "--duration-s",
        "30",
        "--i-consent",
        "--out-root",
        str(out_root),
        "--ckpt",
        str(ckpt),
    ]
    if opt_in:
        args += ["--keep-audio", "--keep-text"]
    assert kit.main(args) == 0
    (run,) = list(out_root.iterdir())
    names = {p.name for p in run.iterdir()}
    assert {
        "metadata.json",
        "level_report.json",
        "engine_rows.json",
        "share_summary.json",
        "run.log",
        "RESULTS.md",
    } <= names
    assert ("session.wav" in names) is opt_in
    share = json.loads((run / "share_summary.json").read_text(encoding="utf-8"))
    assert share["result_state"].startswith("SIMULATED INPUT")
    assert share["metadata"]["simulated_input"] is True and share["error"] is None
    assert share["n_rows"] > 0 and "texto_ventana" not in json.dumps(share)
    assert "secretword" not in (run / "share_summary.json").read_text(encoding="utf-8")
    assert "secretword" not in (run / "run.log").read_text(encoding="utf-8")
    rows_text = (run / "engine_rows.json").read_text(encoding="utf-8")
    assert ("secretword" in rows_text) is opt_in
    assert "**STATE: NO REALIZADA**" in (run / "RESULTS.md").read_text(
        encoding="utf-8"
    )  # never self-promoted
    meta = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
    assert (
        meta["status"] == "NOT_A_PHYSICAL_MICROPHONE_RESULT"
        and meta["keep_text"] is opt_in
    )
    # a second run goes to a different folder
    import time

    time.sleep(1.1)
    assert kit.main(args) == 0
    assert len(list(out_root.iterdir())) == 2


def test_consent_is_required_without_a_terminal(kit, tmp_path, monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    ckpt = tmp_path / "c.pt"
    ckpt.write_bytes(b"x")
    with pytest.raises(SystemExit):
        kit.main(
            [
                "--simulate-wav",
                str(tmp_path / "x.wav"),
                "--ckpt",
                str(ckpt),
                "--out-root",
                str(tmp_path / "o"),
            ]
        )
    assert not (tmp_path / "o").exists()
