"""Single data model of the MRCD C4 diagrams + deterministic local renderer (Mermaid source + SVG).

Run (repo root, standard library only, no network, no browser, no Node):

    .venv/Scripts/python.exe docs/architecture/c4_model.py

Writes c4_1_context / c4_2_containers / c4_3_components / c4_4_code as .mmd (Mermaid flowchart source) and .svg.
The SVG is produced by this script (hand-rolled layout, absolute coordinates stored below); the .mmd files are
generated from the SAME model, but no Mermaid renderer exists on this machine, so the .mmd were NOT render-tested.
Nothing here is random: same model -> byte-identical output.

Conventions
-----------
* kind="current"  : exists in the inspected tree (MRCD) -- blue.
* kind="app"      : exists in ORATORIA-PROYECTO (separate repo, read-only inspection) -- green.
* kind="external" : third party / data not owned by either repo -- grey.
* kind="person"   : human actor -- yellow.
* kind="proposed" : does NOT exist; part of the proposed integration -- dashed orange.
* edge style: "sync" (request/response), "live" (live/streaming path), "offline" (batch path),
  "data" (file/model read), "proposed" (not implemented).
* edge `check="import"`: verify_architecture.py must find an import between the two components' modules.
* edge `via`: optional waypoints (x, y) so a line avoids boxes.
"""

from __future__ import annotations

import html
import textwrap
from pathlib import Path

OUT = Path(__file__).resolve().parent


def N(
    id,
    title,
    tech,
    body,
    x,
    y,
    w=250,
    kind="current",
    group=None,
    modules=(),
    symbols=(),
    app_paths=(),
):
    return dict(
        id=id,
        title=title,
        tech=tech,
        body=body,
        x=x,
        y=y,
        w=w,
        kind=kind,
        group=group,
        modules=list(modules),
        symbols=list(symbols),
        app_paths=list(app_paths),
    )


def E(a, b, label, style="sync", check=None, via=()):
    return dict(a=a, b=b, label=label, style=style, check=check, via=list(via))


def G(id, title, x, y, w, h, trust=False):
    return dict(id=id, title=title, x=x, y=y, w=w, h=h, trust=trust)


DIAGRAMS = {}

# ---- Level 1: system context -----------------------------------------------------------------
DIAGRAMS["c4_1_context"] = dict(
    title="C4 level 1 - System context (as inspected)",
    width=1200,
    height=700,
    groups=[
        G(
            "local",
            "Local workstation (nothing MRCD processes leaves it)",
            20,
            40,
            860,
            640,
            True,
        ),
        G(
            "cloud",
            "Third-party cloud (audio leaves the machine)",
            910,
            40,
            270,
            230,
            True,
        ),
    ],
    nodes=[
        N(
            "speaker",
            "Speaker",
            "Person",
            "Records a speech with microphone and camera in the browser. Does not see MRCD output today.",
            40,
            100,
            210,
            "person",
            "local",
        ),
        N(
            "researcher",
            "Researcher / advisor",
            "Person",
            "Runs MRCD scripts and the live demo and reads results. OE1/OE2 approval is PENDING (no signature).",
            40,
            420,
            210,
            "person",
            "local",
        ),
        N(
            "app",
            "ORATORIA-PROYECTO web app",
            "Software system (separate repo)",
            "Practice sessions and report. Embeds ORATORIA-ENGINE v0 (3 classes). It does NOT call MRCD.",
            380,
            100,
            250,
            "app",
            "local",
        ),
        N(
            "mrcd",
            "MRCD engine (this repo)",
            "Software system, Python 0.3.0",
            "Experimental 7-class window classifier over audio and ASR text. Offline files and live replay. Scores are not calibrated.",
            380,
            420,
            250,
            "current",
            "local",
        ),
        N(
            "corpora",
            "Annotated corpora",
            "Files (git-ignored)",
            "DIHANA, DIME, ESLORA, own clips. Offline evaluation only; licences restrict redistribution.",
            720,
            380,
            140,
            "external",
            "local",
        ),
        N(
            "models",
            "Model files",
            "Files",
            "ckpt_audio_text.pt, Vc head, faster-whisper small cache.",
            720,
            560,
            140,
            "external",
            "local",
        ),
        N(
            "asr",
            "Cloud ASR providers",
            "External (optional)",
            "Speechmatics, Deepgram or managed ASR, only if selected by ORATORIA_ASR_ADAPTER. Default is simulado.",
            940,
            100,
            220,
            "external",
            "cloud",
        ),
    ],
    edges=[
        E("speaker", "app", "Speaks; audio and video captured in the browser"),
        E("app", "asr", "Audio PCM16 (only if a cloud adapter is selected)"),
        E("researcher", "mrcd", "Runs scripts, demo, tests", "offline"),
        E("mrcd", "corpora", "Reads (offline)", "data"),
        E("mrcd", "models", "Loads", "data"),
        E(
            "app",
            "mrcd",
            "PROPOSED integration, not implemented: local file-in / JSON-out",
            "proposed",
        ),
    ],
)

# ---- Level 2: containers ---------------------------------------------------------------------
DIAGRAMS["c4_2_containers"] = dict(
    title="C4 level 2 - Containers (as inspected; proposed integration dashed orange)",
    width=1640,
    height=1180,
    groups=[
        G(
            "browser",
            "Browser on the user device (video never leaves it)",
            20,
            40,
            380,
            420,
            True,
        ),
        G("backend", "App backend (docker compose, local)", 520, 40, 760, 630, True),
        G("cloud", "Third-party cloud", 1310, 40, 300, 200, True),
        G(
            "mrcdp",
            "MRCD local Python process / working tree",
            20,
            720,
            1600,
            440,
            True,
        ),
    ],
    nodes=[
        N(
            "spa",
            "Web SPA",
            "React 19, Vite, AudioWorklet, MediaPipe worker",
            "getUserMedia mono; PCM16 16 kHz 160 ms chunks; vision and prosody summaries (~1/s); report UI. No video upload.",
            50,
            90,
            320,
            "app",
            "browser",
            app_paths=[
                "frontend/src/contexts/speech-analysis/hooks/use-audio-capture.ts"
            ],
        ),
        N(
            "rec",
            "Local recording (research, optional)",
            "MediaRecorder",
            "Records video+audio and only downloads it locally; keeps a local reference.",
            50,
            290,
            320,
            "app",
            "browser",
            app_paths=["frontend/src/contexts/research/hooks/use-local-recording.ts"],
        ),
        N(
            "api",
            "Backend API",
            "FastAPI, WS /ws/session, REST /api/v1/sessions",
            "Authenticates, frames audio, runs the ASR port, closes segments, stores transcript and report.",
            550,
            90,
            700,
            "app",
            "backend",
            app_paths=["backend/src/oratoria/platform/ws/protocol/audio_frame.py"],
        ),
        N(
            "asrport",
            "ASR adapter port",
            "SpeechRecognizer port",
            "ORATORIA_ASR_ADAPTER: simulado (default), gestionado, whisper_http, speechmatics, deepgram, oratoria_engine.",
            550,
            290,
            230,
            "app",
            "backend",
            app_paths=["backend/src/oratoria/bootstrap/asr.py"],
        ),
        N(
            "engine0",
            "ORATORIA-ENGINE v0 (embedded)",
            "Python, separate repo",
            "Detector v0 FP/REP/PRO, uncalibrated. This is what the app hosts today.",
            550,
            490,
            230,
            "app",
            "backend",
        ),
        N(
            "db",
            "PostgreSQL",
            "Database",
            "Sessions, transcript, report. No raw-audio persistence found on the WS path.",
            1000,
            290,
            250,
            "app",
            "backend",
            app_paths=[
                "backend/src/oratoria/infrastructure/persistence/postgres/models.py"
            ],
        ),
        N(
            "cloudasr",
            "ASR provider",
            "External API",
            "Receives audio when a cloud adapter is configured.",
            1330,
            90,
            260,
            "external",
            "cloud",
        ),
        N(
            "svc",
            "MRCD service (dev server + CLI)",
            "mrcd/service, FastAPI loopback :8765 / CLI",
            "File in, JSON out, contract mrcd.analysis/v1. No auth, loopback only. Added in commit 955d27c after the first inspection; the app has no client for it.",
            750,
            770,
            300,
            "current",
            "mrcdp",
            modules=["mrcd/service/"],
        ),
        N(
            "lib",
            "mrcd library",
            "Python package mrcd 0.3.0",
            "DisfluencyEngine.load/analyze/analyze_file -> AnalysisResult(Event, Window). Only supported entry; everything else is internal.",
            750,
            920,
            300,
            "current",
            "mrcdp",
            modules=["mrcd/"],
        ),
        N(
            "demo",
            "Live demo",
            "scripts/live_demo.py, FastAPI :8001",
            "Microphone -> LiveScorer rows -> WebSocket /ws and dashboard. Not an API for other systems.",
            340,
            920,
            300,
            "current",
            "mrcdp",
            modules=["scripts/live_demo.py"],
        ),
        N(
            "scripts",
            "Research and batch scripts",
            "scripts/*.py, argparse",
            "Dataset, training, parity, ASR recovery, audits. No console_scripts and no CLI around analyze_file.",
            1160,
            920,
            340,
            "current",
            "mrcdp",
            modules=["scripts/extract_and_label.py", "scripts/run_ablation.py"],
        ),
        N(
            "artifacts",
            "Model artifacts",
            "Files",
            "models/ckpt_audio_text.pt, Vc head (sha256 pinned), faster-whisper small cache.",
            750,
            1070,
            300,
            "external",
            "mrcdp",
            modules=["models/ckpt_audio_text.pt"],
        ),
        N(
            "results",
            "Results store",
            "results/ (git-ignored)",
            "Reports, logs, JSON. Only force-added files are versioned.",
            1160,
            1070,
            340,
            "external",
            "mrcdp",
        ),
    ],
    edges=[
        E("spa", "api", "WS: PCM16 frames + summaries; REST: report", "live"),
        E("rec", "spa", "local reference only", "data"),
        E("api", "asrport", "audio chunks", "live"),
        E("asrport", "engine0", "adapter oratoria_engine", "live"),
        E("asrport", "cloudasr", "audio (cloud adapter only)", "live", via=[(1000, 255), (1290, 255), (1290, 130)]),
        E("api", "db", "persist"),
        E("api", "svc", "PROPOSED: app client does not exist", "proposed"),
        E("svc", "lib", "in-process call", "sync", "import"),
        E("demo", "lib", "LiveScorer rows", "live", "import"),
        E("scripts", "lib", "analyze_file, feature code", "offline", "import"),
        E("lib", "artifacts", "loads checkpoint, Vc, Whisper", "data"),
        E("scripts", "results", "writes", "data"),
    ],
)

# ---- Level 3: components of the mrcd package -------------------------------------------------
_ROWS = [90, 260, 430, 600, 770, 940]
DIAGRAMS["c4_3_components"] = dict(
    title="C4 level 3 - Components of the mrcd package (edges = observed imports unless noted)",
    width=1840,
    height=1500,
    groups=[
        G("feat", "Feature extraction (shared by offline and live)", 20, 40, 340, 1070),
        G("api", "Public API and constants", 700, 40, 300, 700),
        G("model", "Model, ASR profile and head", 1080, 40, 340, 1070),
        G("kit", "Research kits (not imported by the engine)", 1460, 40, 340, 790),
        G("live", "Live path (testable without a microphone)", 20, 1140, 1420, 340),
    ],
    nodes=[
        N(
            "acoustic",
            "Acoustic extractor",
            "extractors/acoustic.py",
            "Silero VAD, pYIN F0, RMS at 100 fps -> 5-D frames.",
            40,
            _ROWS[0],
            300,
            "current",
            "feat",
            modules=["mrcd/extractors/acoustic.py"],
        ),
        N(
            "prosody",
            "Prosody extractor",
            "extractors/prosody.py",
            "9-D boundary features around the longest silence.",
            40,
            _ROWS[1],
            300,
            "current",
            "feat",
            modules=["mrcd/extractors/prosody.py"],
        ),
        N(
            "windows",
            "Window tensors",
            "dataset.py, windowing.py",
            "acoustic_windows, linguistic_windows (10 s context, +2 columns), center_f0; short-unit planning.",
            40,
            _ROWS[2],
            300,
            "current",
            "feat",
            modules=["mrcd/dataset.py", "mrcd/windowing.py"],
        ),
        N(
            "sync",
            "A/V sync buffer",
            "sync/buffer.py",
            "Aligns audio and kinesic by capture timestamp. Imported by dataset.py; not in the live path.",
            40,
            _ROWS[3],
            300,
            "current",
            "feat",
            modules=["mrcd/sync/buffer.py"],
        ),
        N(
            "kin",
            "Kinesic and face tracking",
            "extractors/kinesic.py, face_tracking.py",
            "MediaPipe 12-D. Reached only by offline video scripts; analyze() rejects video modes.",
            40,
            _ROWS[4],
            300,
            "current",
            "feat",
            modules=["mrcd/extractors/kinesic.py", "mrcd/extractors/face_tracking.py"],
        ),
        N(
            "ling",
            "Linguistic extractor + ASR",
            "extractors/linguistic.py",
            "VerbatimASR (faster-whisper), transcribe_chunked 8 s / 2 s, 10 word features.",
            40,
            _ROWS[5],
            300,
            "current",
            "feat",
            modules=["mrcd/extractors/linguistic.py"],
        ),
        N(
            "pub",
            "Engine facade",
            "engine.py, __init__.py",
            "DisfluencyEngine, Event, Window, AnalysisResult; grid 3 s / 0.5 s; _merge; _run_head.",
            720,
            330,
            260,
            "current",
            "api",
            modules=["mrcd/engine.py", "mrcd/__init__.py"],
        ),
        N(
            "consts",
            "Constants",
            "constants.py",
            "Taxonomy (7 + fluent), SAMPLE_RATE, WINDOW_S, STRIDE_S, CONTEXT_S, FEATURE_SCHEMA_VERSION, H.",
            720,
            130,
            260,
            "current",
            "api",
            modules=["mrcd/constants.py"],
        ),
        N(
            "service",
            "Service surface",
            "service/*",
            "analyzer.py, contract.py (mrcd.analysis/v1), server.py (dev), cli.py. Wraps DisfluencyEngine; not imported by the engine.",
            720,
            560,
            260,
            "current",
            "api",
            modules=["mrcd/service/"],
        ),
        N(
            "fusion",
            "Fusion network",
            "models/fusion.py",
            "CrossModalFusion; audio_text checkpoint has 75,144 parameters; logits over 8 labels.",
            1100,
            _ROWS[0],
            300,
            "current",
            "model",
            modules=["mrcd/models/fusion.py"],
        ),
        N(
            "infer",
            "Checkpoint and inference",
            "inference.py",
            "load_ckpt (fails on schema mismatch), predict = softmax.",
            1100,
            _ROWS[1],
            300,
            "current",
            "model",
            modules=["mrcd/inference.py"],
        ),
        N(
            "head",
            "Vc head (experimental)",
            "adaptation.py",
            "Hash-pinned MLP on pooled features; max over windows vs frozen thresholds; not calibrated.",
            1100,
            _ROWS[2],
            300,
            "current",
            "model",
            modules=["mrcd/adaptation.py"],
        ),
        N(
            "decision",
            "Decision utilities",
            "decision.py",
            "unit_score, three_state, CV thresholds. NOT called by the engine.",
            1100,
            _ROWS[3],
            300,
            "current",
            "model",
            modules=["mrcd/decision.py"],
        ),
        N(
            "profiles",
            "ASR profiles",
            "asr_profiles.py",
            "p0 (default), p1_disfluent_prompt (experimental), fingerprint, anomaly heuristics.",
            1100,
            _ROWS[4],
            300,
            "current",
            "model",
            modules=["mrcd/asr_profiles.py"],
        ),
        N(
            "repro",
            "Reproducibility",
            "reproducibility.py",
            "OMP pin/status/warning, file sha256, engine_environment.",
            1100,
            _ROWS[5],
            300,
            "current",
            "model",
            modules=["mrcd/reproducibility.py"],
        ),
        N(
            "anno",
            "Weak annotation",
            "annotation/auto_labeler.py",
            "Heuristic cascade that produced the TRAINING labels (seconds-based thresholds). Not used at inference.",
            1480,
            _ROWS[0],
            300,
            "current",
            "kit",
            modules=["mrcd/annotation/auto_labeler.py"],
        ),
        N(
            "corp",
            "Corpus importers",
            "corpora/*",
            "DIHANA, DIME, ESLORA importers; absence of a label = unknown.",
            1480,
            _ROWS[1],
            300,
            "current",
            "kit",
            modules=["mrcd/corpora/"],
        ),
        N(
            "ctxeval",
            "Context evaluation",
            "context_eval/*",
            "DIHANA folds 1-4 repetition/revision experiment (exploratory).",
            1480,
            _ROWS[2],
            300,
            "current",
            "kit",
            modules=["mrcd/context_eval/"],
        ),
        N(
            "asrrec",
            "ASR recovery experiment",
            "asr_recovery/*",
            "P0/P1/P2 decoding study (exploratory).",
            1480,
            _ROWS[3],
            300,
            "current",
            "kit",
            modules=["mrcd/asr_recovery/"],
        ),
        N(
            "sasr",
            "Streaming ASR",
            "live/streaming_asr.py",
            "Re-transcribes the last 13 s and merges words by absolute time.",
            40,
            1190,
            300,
            "current",
            "live",
            modules=["mrcd/live/streaming_asr.py"],
        ),
        N(
            "tracker",
            "Session feature tracker",
            "live/session_features.py, pyin_carry.py",
            "Incremental VAD/F0/frames with margins L/R; exact Viterbi resume in v3.",
            395,
            1190,
            250,
            "current",
            "live",
            modules=["mrcd/live/session_features.py", "mrcd/live/pyin_carry.py"],
        ),
        N(
            "scorer",
            "LiveScorer and configs",
            "live/session.py",
            "live_cfg_v1/v2/v3; step and finalize; rows provisional / final.",
            780,
            1190,
            260,
            "current",
            "live",
            modules=["mrcd/live/session.py"],
        ),
        N(
            "mic",
            "Microphone stream",
            "live/mic_stream.py",
            "AudioRingBuffer, SessionRecorder, MicrophoneStream (sounddevice, extra demo). Driven by scripts/live_demo.py.",
            1100,
            1190,
            300,
            "current",
            "live",
            modules=["mrcd/live/mic_stream.py"],
        ),
        N(
            "wbuild",
            "Window builder",
            "live/window_builder.py",
            "window_text; offline-equivalent tensor builder kept for tests.",
            780,
            1350,
            260,
            "current",
            "live",
            modules=["mrcd/live/window_builder.py"],
        ),
    ],
    edges=[
        E("pub", "consts", "constants", "offline", "import"),
        E("service", "pub", "DisfluencyEngine", "sync", "import"),
        E("pub", "acoustic", "extract_acoustic", "offline", "import"),
        E("pub", "prosody", "prosody_windows", "offline", "import"),
        E("pub", "windows", "window tensors", "offline", "import"),
        E("pub", "ling", "VerbatimASR", "offline", "import", via=[(640, 600), (520, 995)]),
        E("pub", "infer", "predict", "offline", "import"),
        E("pub", "head", "opt-in Vc (lazy import)", "offline", "import", via=[(1040, 485)]),
        E("pub", "profiles", "profile + anomalies", "offline", "import", via=[(1000, 480), (1040, 825)]),
        E("pub", "repro", "environment record", "offline", "import", via=[(1000, 480), (1040, 995)]),
        E("infer", "fusion", "builds the net", "offline", "import"),
        E("head", "infer", "BATCH_KEYS", "offline", "import"),
        E("windows", "prosody", "speaker_f0_center", "offline", "import"),
        E("windows", "sync", "kinesic alignment", "offline", "import"),
        E("ling", "profiles", "decode kwargs", "offline", "import"),
        E("sasr", "ling", "VerbatimASR", "live", "import"),
        E("scorer", "tracker", "tensors per tick", "live", "import"),
        E("scorer", "wbuild", "window_text", "live", "import"),
        E(
            "tracker",
            "windows",
            "reuses offline tensor functions",
            "live",
            "import",
            via=[(520, 485)],
        ),
    ],
)

# ---- Level 4: code ---------------------------------------------------------------------------
DIAGRAMS["c4_4_code"] = dict(
    title="C4 level 4 - Code: central types and entry points (every symbol verified by AST)",
    width=1800,
    height=1000,
    groups=[
        G(
            "batch",
            "Batch path: mrcd/engine.py and mrcd/inference.py",
            20,
            40,
            900,
            560,
        ),
        G("livep", "Live path: mrcd/live/*", 950, 40, 830, 560),
        G("aux", "ASR, head and network", 20, 640, 1760, 200),
    ],
    nodes=[
        N(
            "c_engine",
            "DisfluencyEngine",
            "class, mrcd/engine.py",
            "load(...) classmethod; analyze(audio, ...); analyze_file(path); _evaluate; _run_head; _merge; _asr_profile_meta.",
            40,
            90,
            380,
            "current",
            "batch",
            symbols=[
                ("mrcd/engine.py", s)
                for s in (
                    "DisfluencyEngine",
                    "DisfluencyEngine.load",
                    "DisfluencyEngine.analyze",
                    "DisfluencyEngine.analyze_file",
                    "DisfluencyEngine._evaluate",
                    "DisfluencyEngine._run_head",
                    "DisfluencyEngine._merge",
                    "DisfluencyEngine._asr_profile_meta",
                )
            ],
        ),
        N(
            "c_result",
            "AnalysisResult",
            "dataclass, mrcd/engine.py",
            "events, windows, transcript, duration, mode, warning, metadata, head; to_dict(); rate_per_minute().",
            540,
            90,
            360,
            "current",
            "batch",
            symbols=[
                ("mrcd/engine.py", "AnalysisResult"),
                ("mrcd/engine.py", "AnalysisResult.to_dict"),
            ],
        ),
        N(
            "c_event",
            "Event",
            "frozen dataclass",
            "label, start, end, confidence, n_windows, text, modalities, validation, asr_profile.",
            500,
            330,
            190,
            "current",
            "batch",
            symbols=[("mrcd/engine.py", "Event")],
        ),
        N(
            "c_window",
            "Window",
            "frozen dataclass",
            "start, end, label, confidence, probabilities, text, terminal, head_scores.",
            720,
            330,
            190,
            "current",
            "batch",
            symbols=[("mrcd/engine.py", "Window")],
        ),
        N(
            "c_infer",
            "load_ckpt / predict",
            "functions, mrcd/inference.py",
            "Rejects schema mismatch; softmax over 8 labels.",
            40,
            330,
            380,
            "current",
            "batch",
            symbols=[
                ("mrcd/inference.py", "load_ckpt"),
                ("mrcd/inference.py", "predict"),
            ],
        ),
        N(
            "c_cfg",
            "LiveConfig / LIVE_CFG_V1..V3",
            "frozen dataclass + constants",
            "left/right margins, finalize_on_stop, mark_status, terminal_window, rescore_at_close, viterbi_carry.",
            970,
            90,
            340,
            "current",
            "livep",
            symbols=[
                ("mrcd/live/session.py", s)
                for s in (
                    "LiveConfig",
                    "LIVE_CFG_V1",
                    "LIVE_CFG_V2",
                    "LIVE_CFG_V3",
                    "get_config",
                )
            ],
        ),
        N(
            "c_scorer",
            "LiveScorer",
            "class, mrcd/live/session.py",
            "push(audio); step(words) -> provisional rows; finalize(words) -> final rows. Rows are dicts with Spanish keys.",
            1420,
            90,
            340,
            "current",
            "livep",
            symbols=[
                ("mrcd/live/session.py", s)
                for s in (
                    "LiveScorer",
                    "LiveScorer.push",
                    "LiveScorer.step",
                    "LiveScorer.finalize",
                )
            ],
        ),
        N(
            "c_tracker",
            "SessionFeatureTracker",
            "class, mrcd/live/session_features.py",
            "append, finalize, latest_start, window(start, words) -> tensors.",
            970,
            320,
            340,
            "current",
            "livep",
            symbols=[
                ("mrcd/live/session_features.py", s)
                for s in (
                    "SessionFeatureTracker",
                    "SessionFeatureTracker.window",
                    "SessionFeatureTracker.latest_start",
                )
            ],
        ),
        N(
            "c_sasr",
            "StreamingASR",
            "class, mrcd/live/streaming_asr.py",
            "merge_recent, asr_mark, dedupe_words.",
            1420,
            320,
            340,
            "current",
            "livep",
            symbols=[
                ("mrcd/live/streaming_asr.py", s)
                for s in ("StreamingASR", "StreamingASR.merge_recent", "dedupe_words")
            ],
        ),
        N(
            "c_mic",
            "MicrophoneStream",
            "class, mrcd/live/mic_stream.py",
            "sounddevice callback -> AudioRingBuffer; SessionRecorder keeps the whole session.",
            970,
            480,
            340,
            "current",
            "livep",
            symbols=[
                ("mrcd/live/mic_stream.py", s)
                for s in ("MicrophoneStream", "AudioRingBuffer", "SessionRecorder")
            ],
        ),
        N(
            "c_fusion",
            "CrossModalFusion",
            "nn.Module, mrcd/models/fusion.py",
            "forward(batch) -> logits; .head = Dropout + Linear(d*2 -> 8).",
            40,
            700,
            400,
            "current",
            "aux",
            symbols=[
                ("mrcd/models/fusion.py", "CrossModalFusion"),
                ("mrcd/models/fusion.py", "CrossModalFusion.forward"),
            ],
        ),
        N(
            "c_profile",
            "ASRProfile / PROFILES",
            "frozen dataclass, mrcd/asr_profiles.py",
            "p0, p1_disfluent_prompt; decode_kwargs; describe; asr_cache_key; detect_asr_anomalies.",
            500,
            700,
            400,
            "current",
            "aux",
            symbols=[
                ("mrcd/asr_profiles.py", s)
                for s in (
                    "ASRProfile",
                    "PROFILES",
                    "get_profile",
                    "asr_cache_key",
                    "detect_asr_anomalies",
                )
            ],
        ),
        N(
            "c_asr",
            "VerbatimASR",
            "class, mrcd/extractors/linguistic.py",
            "transcribe, transcribe_chunked(8 s, 2 s), model_info.",
            960,
            700,
            400,
            "current",
            "aux",
            symbols=[
                ("mrcd/extractors/linguistic.py", s)
                for s in (
                    "VerbatimASR",
                    "VerbatimASR.transcribe_chunked",
                    "VerbatimASR.model_info",
                )
            ],
        ),
        N(
            "c_head",
            "LoadedHead / load_vc",
            "frozen dataclass, mrcd/adaptation.py",
            "window_scores (sigmoid), unit_decision (max >= frozen threshold); calibrated: false.",
            1420,
            700,
            340,
            "current",
            "aux",
            symbols=[
                ("mrcd/adaptation.py", s)
                for s in (
                    "LoadedHead",
                    "LoadedHead.unit_decision",
                    "load_vc",
                    "VC_SHA256",
                )
            ],
        ),
    ],
    edges=[
        E("c_engine", "c_result", "returns", "offline"),
        E("c_result", "c_event", "events[]", "offline"),
        E("c_result", "c_window", "windows[]", "offline"),
        E("c_engine", "c_infer", "predict()", "offline"),
        E("c_infer", "c_fusion", "builds the net", "offline"),
        E("c_scorer", "c_cfg", "config", "live"),
        E("c_scorer", "c_tracker", "tracker.window()", "live"),
        E("c_scorer", "c_sasr", "words (via the driver)", "live"),
        E("c_tracker", "c_mic", "audio chunks (via the driver)", "live"),
    ],
)

# ---- styling ---------------------------------------------------------------------------------
STYLE = {
    "current": dict(fill="#e3f2fd", stroke="#1565c0", dash=""),
    "app": dict(fill="#e8f5e9", stroke="#2e7d32", dash=""),
    "external": dict(fill="#eeeeee", stroke="#616161", dash=""),
    "person": dict(fill="#fff8e1", stroke="#b28704", dash=""),
    "proposed": dict(fill="#ffffff", stroke="#e65100", dash="7 4"),
}
EDGE = {
    "sync": dict(color="#333333", dash="", name="request / response"),
    "live": dict(color="#1565c0", dash="", name="live / streaming path"),
    "offline": dict(color="#6a1b9a", dash="", name="offline / batch path"),
    "data": dict(color="#616161", dash="2 3", name="file or model read/write"),
    "proposed": dict(color="#e65100", dash="7 4", name="PROPOSED, not implemented"),
}
FS_TITLE, FS_BODY = 12.5, 10.5
CW = 6.0


def _lines(text, width_px, cw=CW):
    return textwrap.wrap(text, max(int(width_px / cw), 8)) or [""]


def _node_geom(n):
    inner = n["w"] - 16
    tl, kl, bl = (
        _lines(n["title"], inner, 7.2),
        _lines(n["tech"], inner, 5.6),
        _lines(n["body"], inner),
    )
    return tl, kl, bl, 10 + len(tl) * 15 + len(kl) * 12 + 4 + len(bl) * 13 + 10


def _border(n, h, tx, ty):
    cx, cy = n["x"] + n["w"] / 2, n["y"] + h / 2
    dx, dy = tx - cx, ty - cy
    if dx == 0 and dy == 0:
        return cx, cy
    sx = (n["w"] / 2) / abs(dx) if dx else float("inf")
    sy = (h / 2) / abs(dy) if dy else float("inf")
    s = min(sx, sy)
    return cx + dx * s, cy + dy * s


def render_svg(key, d):
    nodes = {n["id"]: n for n in d["nodes"]}
    geom = {i: _node_geom(n) for i, n in nodes.items()}
    W, H = d["width"], d["height"] + 150
    o = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Segoe UI, Arial, sans-serif">',
        f"<title>{html.escape(d['title'])}</title>",
        "<defs>",
    ]
    for name, st in EDGE.items():
        o.append(
            f'<marker id="ar_{name}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{st["color"]}"/></marker>'
        )
    o.append("</defs>")
    o.append(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')
    o.append(
        f'<text x="20" y="26" font-size="16" font-weight="700" fill="#111">{html.escape(d["title"])}</text>'
    )
    for g in d["groups"]:
        dash, col = ("10 5", "#b71c1c") if g["trust"] else ("3 3", "#9e9e9e")
        o.append(
            f'<rect x="{g["x"]}" y="{g["y"]}" width="{g["w"]}" height="{g["h"]}" rx="8" fill="none" stroke="{col}" stroke-width="1.6" stroke-dasharray="{dash}"/>'
        )
        suffix = " [trust boundary]" if g["trust"] else ""
        o.append(
            f'<text x="{g["x"] + 10}" y="{g["y"] + 18}" font-size="11.5" font-weight="600" fill="{col}">{html.escape(g["title"] + suffix)}</text>'
        )
    labels = []
    for e in d["edges"]:
        a, b = nodes[e["a"]], nodes[e["b"]]
        ha, hb = geom[a["id"]][3], geom[b["id"]][3]
        ac, bc = (
            (a["x"] + a["w"] / 2, a["y"] + ha / 2),
            (b["x"] + b["w"] / 2, b["y"] + hb / 2),
        )
        via = e["via"]
        p_start = _border(a, ha, *(via[0] if via else bc))
        p_end = _border(b, hb, *(via[-1] if via else ac))
        pts = [p_start, *via, p_end]
        st = EDGE[e["style"]]
        path = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        o.append(
            f'<polyline points="{path}" fill="none" stroke="{st["color"]}" stroke-width="1.7" stroke-dasharray="{st["dash"]}" marker-end="url(#ar_{e["style"]})"/>'
        )
        # label on the longest segment
        seg = max(
            range(len(pts) - 1),
            key=lambda i: (
                (pts[i + 1][0] - pts[i][0]) ** 2 + (pts[i + 1][1] - pts[i][1]) ** 2
            ),
        )
        mx, my = (
            (pts[seg][0] + pts[seg + 1][0]) / 2,
            (pts[seg][1] + pts[seg + 1][1]) / 2,
        )
        labels.append((mx, my, e["label"], st["color"]))
    for n in d["nodes"]:
        tl, kl, bl, h = geom[n["id"]]
        st = STYLE[n["kind"]]
        o.append(
            f'<rect x="{n["x"]}" y="{n["y"]}" width="{n["w"]}" height="{h}" rx="6" fill="{st["fill"]}" stroke="{st["stroke"]}" stroke-width="1.8" stroke-dasharray="{st["dash"]}"/>'
        )
        y = n["y"] + 10
        for ln in tl:
            y += 15
            o.append(
                f'<text x="{n["x"] + 8}" y="{y - 3}" font-size="{FS_TITLE}" font-weight="700" fill="#111">{html.escape(ln)}</text>'
            )
        for ln in kl:
            y += 12
            o.append(
                f'<text x="{n["x"] + 8}" y="{y - 2}" font-size="9.5" font-style="italic" fill="#37474f">{html.escape(ln)}</text>'
            )
        y += 4
        for ln in bl:
            y += 13
            o.append(
                f'<text x="{n["x"] + 8}" y="{y - 3}" font-size="{FS_BODY}" fill="#222">{html.escape(ln)}</text>'
            )
    for mx, my, lab, col in labels:  # labels last: always readable
        wrapped = textwrap.wrap(lab, 26) or [lab]
        lw = max(len(x) for x in wrapped) * 5.4 + 8
        lh = len(wrapped) * 12 + 4
        o.append(
            f'<rect x="{mx - lw / 2:.1f}" y="{my - lh / 2:.1f}" width="{lw:.1f}" height="{lh}" rx="3" fill="#ffffff" fill-opacity="0.95" stroke="{col}" stroke-width="0.6"/>'
        )
        for k, ln in enumerate(wrapped):
            o.append(
                f'<text x="{mx:.1f}" y="{my - lh / 2 + 12 + k * 12:.1f}" font-size="9.5" text-anchor="middle" fill="{col}">{html.escape(ln)}</text>'
            )
    ly = d["height"] + 20
    o.append(
        f'<text x="20" y="{ly}" font-size="12" font-weight="700" fill="#111">Legend</text>'
    )
    x = 20
    for k, nm in (
        ("current", "exists in MRCD tree"),
        ("app", "exists in ORATORIA-PROYECTO (read-only inspection)"),
        ("external", "external / data"),
        ("person", "person"),
        ("proposed", "PROPOSED, does not exist"),
    ):
        st = STYLE[k]
        o.append(
            f'<rect x="{x}" y="{ly + 10}" width="22" height="14" rx="3" fill="{st["fill"]}" stroke="{st["stroke"]}" stroke-width="1.6" stroke-dasharray="{st["dash"]}"/>'
        )
        o.append(
            f'<text x="{x + 28}" y="{ly + 22}" font-size="10.5" fill="#222">{html.escape(nm)}</text>'
        )
        x += 40 + len(nm) * 5.6
    x = 20
    for k, st in EDGE.items():
        o.append(
            f'<line x1="{x}" y1="{ly + 48}" x2="{x + 34}" y2="{ly + 48}" stroke="{st["color"]}" stroke-width="1.7" stroke-dasharray="{st["dash"]}" marker-end="url(#ar_{k})"/>'
        )
        o.append(
            f'<text x="{x + 42}" y="{ly + 52}" font-size="10.5" fill="#222">{html.escape(st["name"])}</text>'
        )
        x += 60 + len(st["name"]) * 5.6
    o.append(
        f'<line x1="20" y1="{ly + 74}" x2="54" y2="{ly + 74}" stroke="#b71c1c" stroke-width="1.6" stroke-dasharray="10 5"/>'
    )
    o.append(
        f'<text x="62" y="{ly + 78}" font-size="10.5" fill="#222">trust boundary (red dashed); grey dotted frame = grouping only</text>'
    )
    o.append(
        f'<text x="20" y="{ly + 100}" font-size="10" fill="#555">Architecture of the inspected tree (commit in README.md). Generated by docs/architecture/c4_model.py; do not edit by hand.</text>'
    )
    o.append("</svg>")
    return "\n".join(o) + "\n"


def render_mmd(key, d):
    cls = {
        "current": "cur",
        "app": "app",
        "external": "ext",
        "person": "per",
        "proposed": "prop",
    }

    def node(n, ind):
        body = n["body"].replace('"', "'")
        return f'{ind}{n["id"]}["<b>{n["title"]}</b><br/><i>{n["tech"]}</i><br/>{body}"]:::{cls[n["kind"]]}'

    o = [
        f"%% {d['title']}",
        "%% Generated by docs/architecture/c4_model.py from the same model as the SVG. NOT render-tested locally (no Mermaid renderer installed).",
        "flowchart LR",
    ]
    for g in d["groups"]:
        o.append(
            f'  subgraph {g["id"]}["{g["title"]}{" [trust boundary]" if g["trust"] else ""}"]'
        )
        o += [node(n, "    ") for n in d["nodes"] if n["group"] == g["id"]]
        o.append("  end")
    o += [node(n, "  ") for n in d["nodes"] if n["group"] is None]
    for e in d["edges"]:
        arrow = {
            "proposed": "-.->",
            "data": "-.->",
            "live": "==>",
            "offline": "-->",
            "sync": "-->",
        }[e["style"]]
        lab = e["label"].replace('"', "'")
        if e["style"] == "proposed" and not lab.startswith("PROPOSED"):
            lab = "PROPOSED: " + lab
        o.append(f'  {e["a"]} {arrow}|"{lab}"| {e["b"]}')
    o += [
        "  classDef cur fill:#e3f2fd,stroke:#1565c0",
        "  classDef app fill:#e8f5e9,stroke:#2e7d32",
        "  classDef ext fill:#eeeeee,stroke:#616161",
        "  classDef per fill:#fff8e1,stroke:#b28704",
        "  classDef prop fill:#ffffff,stroke:#e65100,stroke-dasharray:7 4",
    ]
    return "\n".join(o) + "\n"


def main():
    for key, d in DIAGRAMS.items():
        (OUT / f"{key}.svg").write_text(
            render_svg(key, d), encoding="utf-8", newline="\n"
        )
        (OUT / f"{key}.mmd").write_text(
            render_mmd(key, d), encoding="utf-8", newline="\n"
        )
        print("wrote", key)


if __name__ == "__main__":
    main()
