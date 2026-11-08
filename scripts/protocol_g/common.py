"""Shared constants, paths and helpers of the protocol G harness.

The protocol F harness (``scripts/confirmatoria/run_confirmatoria.py``) is imported unchanged and exposed as ``H``;
the annotation kit v1.1 (``scripts/annotation/agreement.py``, ``markers.py``) is imported unchanged as well.
Every constant cites the protocol section that fixes it (``protocol <section>``); none is assumed by the harness.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
_F = REPO / "scripts" / "confirmatoria" / "run_confirmatoria.py"


def _load_f():
    if "run_confirmatoria" in sys.modules:
        return sys.modules["run_confirmatoria"]
    spec = importlib.util.spec_from_file_location("run_confirmatoria", _F)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_confirmatoria"] = mod
    spec.loader.exec_module(
        mod
    )  # sets OMP_NUM_THREADS=4 / HF_HUB_OFFLINE=1 exactly as protocol F does
    return mod


H = _load_f()

# ----------------------------------------------------------------------------------------------- statistics
SEED = H.SEED  # 13
N_BOOT_CI = H.N_BOOT_CI  # 2000
N_BOOT_TEST = H.N_BOOT_TEST  # 5000
ALPHA = 0.05
MAX_DEGENERATE_SHARE = 0.10  # protocol 10.1, PG3: share of the 5 000 TEST resamples with undefined kappa

# ----------------------------------------------------------------------------------------------- gates (§7)
QG1_KAPPA, QG1_DICE, QG1_MIN_EVENTS = 0.60, 0.50, 5
QG2_KAPPA = 0.40
QG3_KAPPA, QG3_MIN_EVENTS = 0.40, 10  # protocol 7: min events counted in the PASS on the rater-2 subset items
GATE_CLASSES = ("filled_pause", "repetition", "lexical_filler", "prolongation")
CLASS_CODE = {
    "filled_pause": "FP",
    "repetition": "REP",
    "lexical_filler": "LF",
    "prolongation": "PRO",
}

# ----------------------------------------------------------------------------------------------- floors (§8)
FLOOR_ITEMS = 20  # protocol 8: also the descriptive floor (below it: counts only)
FLOOR_RECORDINGS = 4
FLOOR_SPEAKERS = 3
FLOOR_MARKER_REFERENCE = 30  # adjudicated non-uncertain marker spans
FLOOR_MARKER_TRANSCRIBED = 20
INAUDIBLE_SHARE = 0.50

# ----------------------------------------------------------------------------------------------- hypotheses (§10)
SURVIVAL_T = 0.80
FLOOR_PGA1_F1 = H.FLOOR_F1_REPETITION  # 0.30, protocol F A1
FLOOR_PGA2_RECALL = H.FLOOR_RECALL_FILLER  # 0.40, protocol F A2
TOL_S = 0.25  # survival / marker tolerance (+-250 ms); S-TOL: 0.10 and 0.50
EVENT_MIN_WIDTH_S = 0.02
EVENT_IOU = 0.10
FAMILY_SIZES = {"primary": 3, "secondary": 6, "cw2": 3, "ft": 3}

# ----------------------------------------------------------------------------------------------- aligner (§4.3)
ALIGNER_MODEL = "jonatasgrosman/wav2vec2-large-xlsr-53-spanish"
ALIGNER_REVISION = "96d7e9b4e4a78af515a3c6d3cee7c0826045d276"
ALIGNER_WEIGHTS_SHA256 = (
    "41c110e55d2eac8c79486ad87dbe8f9527ed034fe087a6adf03c891eeba914c1"
)
ALIGNER_WINDOW_S, ALIGNER_CONTEXT_S, ALIGNER_STRIDE_S = 20.0, 2.0, 0.02
ALIGNER_CONF_TOKEN = 0.50
ALIGNER_CONF_SHARE = 0.70
P0ANCHOR_MATCH_SHARE = 0.70  # protocol 4.3, S-P0ANCHOR: its own exact-match recording gate
MAX_FAILED_RECORDINGS = 1  # "more than 1 of 19"

# ----------------------------------------------------------------------------------------------- system pins (§5, §14)
EXPECTED_DEFAULT_PROFILE = "o2026_11"
EXPECTED_EVIDENCE = {
    "EVIDENCE_GAP_S": 0.15,
    "EVIDENCE_DUR_S": 0.35,
    "EVIDENCE_PUNCT": [",", "...", "…"],
}
WHISPER_SMALL_MODEL_BIN_SHA256 = (
    "3e305921506d8872816023e4c273e75d2419fb89b24da97b4fe7bce14170d671"
)
CW2_REVISION = "f4334f6e8193f2691212d49b20fa12d370e13896"
CW2_WEIGHTS_MANIFEST_SHA256 = (
    "b22cf7a5208e73405514f32b9e7686019364771d51a8a8683f105596d88e7ac9"
)
#: transformers / torch are REQUIRED in the run environment (the CTC aligner imports them); versions are the ones of the
#: venv that holds the whole run stack (faster-whisper + ctranslate2 + transformers + torch) and are recorded, not pinned
EXPECTED_LIBS = {
    "faster-whisper": "1.2.1",
    "ctranslate2": "4.8.2",
    "transformers": "5.22.1",
    "torch": "2.9.1+cu130",
}
REQUIRED_IMPORTS = ("transformers", "torch")
#: sha256 of the OTHER files of the pinned aligner snapshot (config, vocabulary, feature extractor, alphabet);
#: ``pytorch_model.bin`` is pinned by ALIGNER_WEIGHTS_SHA256. Use ``aligner_expected_files()``.
ALIGNER_AUX_FILES_SHA256 = {
    "alphabet.json": "bc7df743c2cc7d5a579ea61a431d938b4146d4713296c9d295212f32386d49c8",
    "config.json": "39ffe954848d5161270eef4378a17a538bffeefc310482f409ec099f7fceaa5b",
    "preprocessor_config.json": "ca5999a45e98bb76ea87a461ba28a23ad32a5bb9f733b8e0f6546ff38b6c612d",
    "special_tokens_map.json": "bb7068de1150661a10b55f9e4b12a0e77af8bf91f5e45e1b58afaf1d0e17f675",
    "vocab.json": "7aea8bcd8ba1c176f12cc96c1e8f45247548f4de3218106f7352e3d833cfafdc",
}


def aligner_expected_files() -> dict[str, str]:
    """Every file of the pinned aligner snapshot that must exist locally and its sha256 (read at call time)."""
    return {**ALIGNER_AUX_FILES_SHA256, "pytorch_model.bin": ALIGNER_WEIGHTS_SHA256}


SYSTEM_PATHS = H.SYSTEM_PATHS
#: path prefixes (posix, repo-relative) where the leakage terms are allowed to appear
ALLOW_LIST = (
    "docs/annotation_20261108/",
    "results/protocol_g",  # matches results/protocol_g/ and results/protocol_g_draft/ (glob protocol_g*)
    "scripts/protocol_g/",
    "tests/test_protocol_g",
    # frozen protocol-F manifest: audio BYTE hashes and file names only, no text (needed by the F protocol record)
    "results/confirmatoria_20261107/PROTOCOL.sha256.json",
)

# ----------------------------------------------------------------------------------------------- file names
LOCK_NAME = "RUN_G.lock.json"
RESULT_NAME = "RESULT_G.json"
PROTOCOL_NAME = "PROTOCOL_G.md"
STAGE1_NAME = "PROTOCOL_G.stage1.sha256.json"
STAGE2_NAME = "PROTOCOL_G.stage2.sha256.json"
OSF1_NAME = "OSF_REGISTRATION_STAGE1.json"
OSF2_NAME = "OSF_REGISTRATION.json"
GATES_NAME = "REFERENCE_GATES_G.json"
FREEZE_NAME = "SYSTEM_FREEZE_G.json"


@dataclass(frozen=True)
class Paths:
    """Every file the harness reads or writes. ``out`` defaults to ``results/protocol_g``; overridable for tests."""

    out: Path = REPO / "results" / "protocol_g"
    repo: Path = REPO

    @property
    def protocol(self) -> Path:
        return self.out / PROTOCOL_NAME

    @property
    def stage1(self) -> Path:
        return self.out / STAGE1_NAME

    @property
    def stage2(self) -> Path:
        return self.out / STAGE2_NAME

    @property
    def osf1(self) -> Path:
        return self.out / OSF1_NAME

    @property
    def osf2(self) -> Path:
        return self.out / OSF2_NAME

    @property
    def gates(self) -> Path:
        return self.out / GATES_NAME

    @property
    def freeze(self) -> Path:
        return self.out / FREEZE_NAME

    @property
    def lock(self) -> Path:
        return self.out / LOCK_NAME

    @property
    def result(self) -> Path:
        return self.out / RESULT_NAME


class Refusal(SystemExit):
    """A precondition of the protocol is not met. Carries a short machine-readable ``code``."""

    def __init__(self, code: str, message: str):
        super().__init__(f"REFUSED [{code}]: {message}")
        self.code = code


# ----------------------------------------------------------------------------------------------- small helpers
def not_available(reason: str, **extra) -> dict:
    """An absent value. NEVER a number: callers must not coerce it to 0 (protocol rule FR-025)."""
    return {"status": "not_available", "reason": reason, **extra}


def is_na(x) -> bool:
    return isinstance(x, dict) and x.get("status") == "not_available"


def clean(o):
    """JSON-safe copy: numpy -> python, NaN/inf -> None (an undefined number is null, never 0)."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (set, frozenset)):
        return [clean(v) for v in sorted(o)]
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else None
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def dumps(o) -> str:
    return json.dumps(clean(o), indent=1, sort_keys=False, ensure_ascii=False)


def read_sidecar(path: Path) -> str | None:
    side = path.with_name(path.name + ".sha256")
    if not side.is_file():
        return None
    return side.read_text("utf8").split()[0].strip().lower()


def check_sidecar(path: Path) -> None:
    """The ``<file>.sha256`` sidecar written by ``unblind.py --write-hash`` must exist and match the bytes."""
    want = read_sidecar(path)
    if want is None:
        raise Refusal("annotation_unhashed", f"{path.name} has no .sha256 sidecar")
    if H.sha256_file(path) != want:
        raise Refusal(
            "annotation_hash_mismatch", f"{path.name} differs from its .sha256 sidecar"
        )


def rel(path: Path, repo: Path) -> str:
    return path.resolve().relative_to(repo.resolve()).as_posix()


def corpora_root(arg: str | None) -> Path:
    r = arg or os.environ.get("MRCD_CORPORA_ROOT")
    if not r:
        raise SystemExit(
            "set MRCD_CORPORA_ROOT or pass --corpora-root (no default path is built in)"
        )
    return Path(r)
