#!/usr/bin/env python3
"""Microphone check for the MRCD live path (to be RUN BY THE USER).

Uses the normal live path: ``scripts/live_demo.LiveEngine`` (ring buffer -> SessionFeatureTracker -> Whisper ->
checkpoint -> LiveScorer) with a ``MicrophoneStream`` subclass that also meters the signal level. It writes ONLY
into a NEW timestamped folder and refuses to reuse one.

Privacy defaults (verified in the code, see README "Privacy"):
* raw audio is NOT kept: ``MicrophoneStream(record=False)`` means no ``SessionRecorder``; the only audio held in memory is
  the 13 s ring buffer, which is never written. ``--keep-audio`` is the explicit opt-in (it writes ``session.wav``).
* transcript text is dropped from the saved rows (``texto_ventana``). ``--keep-text`` is the explicit opt-in.
* the level report holds aggregate numbers only (peak, noise floor, clipping, silence), never samples.
* ``share_summary.json`` is the single file meant to be handed over; it contains no rows, no text and no audio.

Nothing in this file has been run against a physical microphone by the authors. ``--simulate-wav`` feeds an existing
WAV through the same code path (including the same metering) without importing ``sounddevice``.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import _pin_omp  # noqa: E402,F401  (must run before numpy / torch / ctranslate2)

sys.path.insert(0, str(ROOT))

import argparse  # noqa: E402
import asyncio  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import platform  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import wave  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import numpy as np  # noqa: E402

from mrcd.constants import SAMPLE_RATE  # noqa: E402
from mrcd.live.mic_stream import MicrophoneStream  # noqa: E402

CONFIG_PATH = HERE / "config.json"
RESULTS_TEMPLATE = HERE / "RESULTS_TEMPLATE.md"
BLOCK_S = 0.1  # metering resolution
FLOOR_DB = -120.0


def db(x: float) -> float:
    return FLOOR_DB if x <= 1e-6 else max(FLOOR_DB, 20.0 * math.log10(x))


# ----------------------------------------------------------------------------- metering
class LevelMeter:
    """Aggregate level statistics. Holds one float per 100 ms block, never audio samples."""

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.block = int(sample_rate * BLOCK_S)
        self._carry = np.zeros(0, np.float32)
        self.rms_blocks: list[float] = []
        self.peak = 0.0
        self.clipped = 0
        self.samples = 0
        self._sum = 0.0
        self._lock = threading.Lock()

    def push(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        with self._lock:
            self.samples += len(x)
            self._sum += float(x.sum())
            if len(x):
                self.peak = max(self.peak, float(np.abs(x).max()))
                self.clipped += int((np.abs(x) >= 0.999).sum())
            buf = np.concatenate([self._carry, x])
            n = len(buf) // self.block
            for i in range(n):
                seg = buf[i * self.block : (i + 1) * self.block]
                self.rms_blocks.append(float(np.sqrt(np.mean(seg**2))))
            self._carry = buf[n * self.block :]

    def report(self, th: dict) -> dict:
        with self._lock:
            rms = np.array(self.rms_blocks, dtype=np.float64)
            samples, peak, clipped = self.samples, self.peak, self.clipped
            dc = self._sum / samples if samples else 0.0
        if not len(rms):
            return {"measured": False, "reason": "no audio blocks were received"}
        rms_db = np.array([db(v) for v in rms])
        floor = float(np.percentile(rms_db, 10))
        gate = max(floor + th["active_above_floor_db"], th["active_min_dbfs"])
        active = np.where(rms_db > gate)[0]
        lead = float(active[0] * BLOCK_S) if len(active) else None
        trail = float((len(rms) - 1 - active[-1]) * BLOCK_S) if len(active) else None
        out = {
            "measured": True,
            "samples": samples,
            "audio_seconds": round(samples / self.sample_rate, 3),
            "peak_dbfs": round(db(peak), 2),
            "clipped_samples": clipped,
            "clipped_fraction": round(clipped / samples, 6) if samples else None,
            "rms_mean_dbfs": round(float(np.mean(rms_db)), 2),
            "noise_floor_dbfs_p10": round(floor, 2),
            "active_gate_dbfs": round(gate, 2),
            "active_fraction": round(float(len(active) / len(rms)), 3),
            "leading_silence_s": None if lead is None else round(lead, 2),
            "trailing_silence_s": None if trail is None else round(trail, 2),
            "dc_offset": round(dc, 5),
            "signal_present": bool(len(active)) and peak > 1e-4,
        }
        return out


def verdicts(level: dict, wall_s: float | None, dropped_blocks: int, th: dict) -> dict:
    """PASS / WARN / FAIL per check. Heuristic smoke thresholds from config.json, NOT acoustic standards."""
    v: dict[str, dict] = {}

    def put(name: str, status: str, why: str) -> None:
        v[name] = {"status": status, "why": why}

    if not level.get("measured"):
        put(
            "signal",
            "FAIL",
            "no audio blocks received: device not opened or stream stalled",
        )
        return v
    if not level["signal_present"]:
        put(
            "signal",
            "FAIL",
            "no signal above the floor (muted device, wrong device or Windows microphone permission denied)",
        )
    else:
        put("signal", "PASS", "signal present")
    p = level["peak_dbfs"]
    if (
        level["clipped_fraction"]
        and level["clipped_fraction"] > th["clip_fail_fraction"]
    ):
        put(
            "clipping",
            "FAIL",
            f"{level['clipped_samples']} clipped samples; lower the input gain",
        )
    elif level["clipped_samples"] > 0:
        put("clipping", "WARN", f"{level['clipped_samples']} clipped samples")
    else:
        put("clipping", "PASS", "no clipped samples")
    put(
        "level",
        "WARN" if p < th["peak_min_dbfs"] else "PASS",
        f"peak {p} dBFS (warn below {th['peak_min_dbfs']})",
    )
    nf = level["noise_floor_dbfs_p10"]
    put(
        "noise_floor",
        "FAIL"
        if nf > th["noise_floor_fail_dbfs"]
        else "WARN"
        if nf > th["noise_floor_warn_dbfs"]
        else "PASS",
        f"p10 block level {nf} dBFS",
    )
    lead, trail = level["leading_silence_s"], level["trailing_silence_s"]
    if lead is None:
        put(
            "silence_edges",
            "WARN",
            "no active speech found; leading/trailing silence not measurable",
        )
    else:
        ok = lead >= th["edge_silence_min_s"] and trail >= th["edge_silence_min_s"]
        put(
            "silence_edges",
            "PASS" if ok else "WARN",
            f"leading {lead}s, trailing {trail}s (want >= {th['edge_silence_min_s']}s each)",
        )
    put(
        "dc_offset",
        "WARN" if abs(level["dc_offset"]) > th["dc_offset_warn"] else "PASS",
        f"{level['dc_offset']}",
    )
    put(
        "dropped_blocks",
        "FAIL"
        if dropped_blocks > th["dropped_blocks_fail"]
        else "WARN"
        if dropped_blocks
        else "PASS",
        f"{dropped_blocks} blocks flagged by the audio driver",
    )
    if wall_s is not None:
        drift = abs(level["audio_seconds"] - wall_s)
        put(
            "clock_drift",
            "WARN" if drift > th["duration_drift_warn_s"] else "PASS",
            f"audio {level['audio_seconds']}s vs wall {wall_s:.1f}s",
        )
    return v


class MeteredMicrophone(MicrophoneStream):
    """``MicrophoneStream`` that also meters the signal. Same callback path; the meter sees every block first."""

    def __init__(self, *, keep_audio: bool = False, **kw):
        super().__init__(record=keep_audio, **kw)
        self.meter = LevelMeter(self.sample_rate)

    def _callback(self, indata, frames, time_info, status) -> None:
        self.meter.push(indata[:, 0])
        super()._callback(indata, frames, time_info, status)

    def reset(self) -> None:
        super().reset()
        self.meter = LevelMeter(self.sample_rate)


class FileMicrophone(MeteredMicrophone):
    """Simulated microphone: replays a mono 16 kHz WAV through the same callback. Never imports sounddevice."""

    def __init__(self, wav_path: Path, *, speed: float = 1.0, keep_audio: bool = False):
        super().__init__(keep_audio=keep_audio)
        with wave.open(str(wav_path), "rb") as w:
            if (
                w.getnchannels() != 1
                or w.getsampwidth() != 2
                or w.getframerate() != SAMPLE_RATE
            ):
                raise SystemExit(
                    f"{wav_path.name}: need mono 16-bit {SAMPLE_RATE} Hz WAV for the simulation"
                )
            self._pcm = (
                np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(
                    np.float32
                )
                / 32768.0
            )
        self.speed = speed
        self._thread: threading.Thread | None = None
        self._halt = threading.Event()
        self.finished = threading.Event()

    def start(self) -> None:
        self._halt.clear()
        self.finished.clear()
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    def _pump(self) -> None:
        step = int(SAMPLE_RATE * 0.05)
        t0 = time.monotonic()
        for n, i in enumerate(range(0, len(self._pcm), step)):
            if self._halt.is_set():
                break
            self._callback(self._pcm[i : i + step].reshape(-1, 1), step, None, None)
            # schedule against the clock, not with sleep(0.05): sleep granularity on Windows would drift
            delay = t0 + (n + 1) * 0.05 / max(self.speed, 1e-6) - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        self.finished.set()

    def stop(self) -> None:
        self._halt.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=5)


# ----------------------------------------------------------------------------- metadata
def device_info(device) -> dict:
    try:
        import sounddevice as sd

        d = sd.query_devices(device, "input")
        host = sd.query_hostapis(d["hostapi"])["name"]
        return {
            "name": d["name"],
            "host_api": host,
            "default_samplerate_hz": d["default_samplerate"],
            "max_input_channels": d["max_input_channels"],
            "requested": device,
        }
    except Exception as exc:  # noqa: BLE001 - reported, never fatal
        return {"error": f"{type(exc).__name__}: {exc}"}


def git_commit() -> str | None:
    import subprocess

    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
            or None
        )
    except Exception:  # noqa: BLE001
        return None


def collect_metadata(
    args, cfg: dict, mic, checkpoint: Path, consent_at: str, started: str
) -> dict:
    from mrcd.reproducibility import engine_environment, file_sha256

    simulated = bool(args.simulate_wav)
    return {
        "tool": "docs/microphone-check/mic_check.py",
        "status": "NOT_A_PHYSICAL_MICROPHONE_RESULT"
        if simulated
        else "executed_by_user_on_this_machine",
        "simulated_input": simulated,
        "started_utc": started,
        "consent_acknowledged_utc": consent_at,
        "requested_duration_s": args.duration_s,
        "sample_rate_hz": SAMPLE_RATE,
        "channels": 1,
        "live_config": args.live_config,
        "whisper_size": args.whisper,
        "checkpoint_file": checkpoint.name,
        "checkpoint_sha256": file_sha256(checkpoint) if checkpoint.exists() else None,
        "git_commit": git_commit(),
        "keep_audio": args.keep_audio,
        "keep_text": args.keep_text,
        "device": {"simulated": args.simulate_wav and Path(args.simulate_wav).name}
        if simulated
        else device_info(args.device),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "environment": engine_environment(),
        "config_file": CONFIG_PATH.name,
    }


# ----------------------------------------------------------------------------- run
class RunLog:
    def __init__(self, path: Path):
        self.path = path

    def __call__(self, msg: str) -> None:
        line = f"{datetime.now(timezone.utc).strftime('%H:%M:%S')}Z {msg}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def new_run_dir(root: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = root / stamp
    run.mkdir(
        parents=True, exist_ok=False
    )  # exist_ok=False: a run never overwrites a previous one
    return run


def minimal_rows(rows: list[dict], keep_text: bool) -> list[dict]:
    out = []
    for r in rows:
        r = dict(r)
        if not keep_text:
            r.pop("texto_ventana", None)
        out.append(r)
    return out


def run_session(
    engine, mic, duration_s: float, log: RunLog, simulated: bool, prompts: list[dict]
) -> float:
    loop = asyncio.new_event_loop()
    try:
        if not engine.start(loop):
            raise RuntimeError("engine.start() returned False")
        t0 = time.monotonic()
        shown: set[int] = set()
        while True:
            elapsed = time.monotonic() - t0
            for i, p in enumerate(prompts):
                at = p["at_s"] if p["at_s"] >= 0 else duration_s + p["at_s"]  # negative = counted from the end
                if i not in shown and elapsed >= at:
                    shown.add(i)
                    log(f"[{elapsed:5.0f}s] {p['say']}")
            if (
                simulated
                and getattr(mic, "finished", None) is not None
                and mic.finished.is_set()
            ):
                break
            if elapsed >= duration_s:
                break
            time.sleep(0.2)
        wall = time.monotonic() - t0
        engine.stop()
        if not engine.wait_idle(timeout=300):
            log("WARNING: the end-of-session pass did not finish within 300 s")
        return wall
    finally:
        loop.close()


def main(argv: list[str] | None = None) -> int:
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--list-devices",
        action="store_true",
        help="print input devices and exit (does not open a stream)",
    )
    p.add_argument(
        "--device",
        default=cfg["device"],
        help="input device index or name (default: system default)",
    )
    p.add_argument("--duration-s", type=float, default=cfg["duration_s"])
    p.add_argument(
        "--out-root",
        default=str(ROOT / cfg["out_root"]),
        help="a NEW timestamped sub-folder is created here",
    )
    p.add_argument("--ckpt", default=str(ROOT / cfg["checkpoint"]))
    p.add_argument("--whisper", default=cfg["whisper_size"])
    p.add_argument("--live-config", default=cfg["live_config"])
    p.add_argument(
        "--keep-audio",
        action="store_true",
        help="OPT-IN: also write session.wav (raw voice)",
    )
    p.add_argument(
        "--keep-text",
        action="store_true",
        help="OPT-IN: keep transcript text in engine_rows.json",
    )
    p.add_argument(
        "--i-consent",
        action="store_true",
        help="skip the interactive consent question (you confirm the checklist)",
    )
    p.add_argument(
        "--simulate-wav",
        help="TEST ONLY: replay this mono 16 kHz WAV instead of the microphone",
    )
    p.add_argument("--simulate-speed", type=float, default=1.0)
    args = p.parse_args(argv)
    if args.device is not None and str(args.device).isdigit():
        args.device = int(args.device)

    if args.list_devices:
        import sounddevice as sd

        print(sd.query_devices())
        return 0
    if args.duration_s < cfg["min_duration_s"]:
        raise SystemExit(
            f"--duration-s must be >= {cfg['min_duration_s']} (the engine needs a warm-up context)"
        )
    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        raise SystemExit(f"checkpoint not found: {ckpt}")

    simulated = bool(args.simulate_wav)
    if not args.i_consent:
        if not sys.stdin.isatty():
            raise SystemExit(
                "consent required: run interactively, or pass --i-consent after reading CHECKLIST.md"
            )
        print(cfg["consent_text"])
        if input("Type YES to continue: ").strip() != "YES":
            raise SystemExit("aborted: consent not confirmed")
    consent_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    run = new_run_dir(Path(args.out_root))
    log = RunLog(run / "run.log")
    log(f"run folder: {run}")
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")

    th = cfg["thresholds"]
    from scripts.live_demo import LiveEngine

    from mrcd.live.session import get_config

    fill = get_config(args.live_config).fill_dropped_audio  # same driver-drop policy as the normal live path
    mic = (
        FileMicrophone(Path(args.simulate_wav), speed=args.simulate_speed, keep_audio=args.keep_audio)
        if simulated
        else MeteredMicrophone(device=args.device, keep_audio=args.keep_audio, fill_dropped=fill)
    )
    meta = collect_metadata(args, cfg, mic, ckpt, consent_at, started)
    (run / "metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    log(
        f"simulated_input={simulated} keep_audio={args.keep_audio} keep_text={args.keep_text}"
    )
    log("loading engine (checkpoint + Whisper) ...")
    engine = LiveEngine(ckpt, args.whisper, mic=mic, config=args.live_config)
    error: str | None = None
    wall = None
    try:
        wall = run_session(
            engine,
            mic,
            args.duration_s,
            log,
            simulated,
            [] if simulated else cfg["prompts"],
        )
    except RuntimeError as exc:
        error = f"{type(exc).__name__}: {exc}"
        log(f"FAILED to run: {error.splitlines()[0]}")
    state_error = getattr(getattr(engine, "_state", None), "error", None)
    if state_error is not None:
        error = error or f"worker: {type(state_error).__name__}: {state_error}"
        log(f"worker error: {type(state_error).__name__}")

    level = mic.meter.report(th)
    checks = verdicts(level, wall, mic.dropped_blocks, th)
    rows = list(engine.session_rows)
    counts: dict[str, int] = {}
    for r in rows:
        if r.get("status") != "warming":
            counts[r.get("clase", "?")] = counts.get(r.get("clase", "?"), 0) + 1
    level_report = {
        "level": level,
        "checks": checks,
        "dropped_blocks": mic.dropped_blocks,
        "dropped_samples": mic.dropped_samples,
        "wall_s": None if wall is None else round(wall, 2),
        "thresholds": th,
        "thresholds_note": cfg["thresholds_note"],
    }
    (run / "level_report.json").write_text(
        json.dumps(level_report, indent=2) + "\n", encoding="utf-8"
    )
    (run / "engine_rows.json").write_text(
        json.dumps(minimal_rows(rows, args.keep_text), indent=1, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    share = {
        "what_this_is": "Minimised summary of a microphone-pipeline check. Contains no audio, no text and no per-window rows.",
        "result_state": "NO REALIZADA until the user completes RESULTS.md"
        if not simulated
        else "SIMULATED INPUT - not a microphone result",
        "metadata": {
            k: meta[k]
            for k in (
                "started_utc",
                "simulated_input",
                "requested_duration_s",
                "live_config",
                "whisper_size",
                "checkpoint_sha256",
                "git_commit",
                "platform",
                "python",
            )
        },
        "device": {
            k: meta["device"].get(k)
            for k in ("name", "host_api", "default_samplerate_hz", "max_input_channels")
        }
        if not simulated
        else meta["device"],
        "level": level,
        "checks": checks,
        "dropped_blocks": mic.dropped_blocks,
        "n_rows": len(rows),
        "rows_by_class": counts,
        "gaps": list(engine.gaps),
        "error": error,
    }
    (run / "share_summary.json").write_text(
        json.dumps(share, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if (
        args.keep_audio
        and getattr(mic, "recorder", None) is not None
        and mic.recorder.duration_s > 0
    ):
        (run / "session.wav").write_bytes(mic.recorder.to_wav_bytes())
        log(
            "WARNING: session.wav contains your raw voice; delete it or keep it private"
        )
    (run / "RESULTS.md").write_text(
        RESULTS_TEMPLATE.read_text(encoding="utf-8"), encoding="utf-8"
    )
    for name, c in checks.items():
        log(f"{c['status']:4s} {name}: {c['why']}")
    log(f"rows={len(rows)} by_class={counts}")
    log(
        f"done. Review {run / 'share_summary.json'} and complete {run / 'RESULTS.md'} (state stays NO REALIZADA until you do)."
    )
    return 0 if error is None else 2


if __name__ == "__main__":
    sys.exit(main())
