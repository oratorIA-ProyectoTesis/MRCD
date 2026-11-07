"""Freeze / verify the CHM150 protocol. ``python -I freeze.py`` writes PROTOCOL.sha256.json ONCE (refuses to overwrite)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _pin  # noqa: E402,F401

import datetime  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402

R = _pin.REPO
FROZEN = {
    "protocol": _pin.OUT / "PROTOCOL.md",
    "reference_summary": _pin.OUT / "reference_summary.json",
    "reference_local": _pin.OUT / "reference.json",
    "pin": R / "scripts/external_chm150/_pin.py",
    "freeze": R / "scripts/external_chm150/freeze.py",
    "s0_reference": R / "scripts/external_chm150/s0_reference.py",
    "s1_run": R / "scripts/external_chm150/s1_run.py",
    "s2_eval": R / "scripts/external_chm150/s2_eval.py",
    "importer": R / "mrcd/corpora/chm150.py",
    "transcript_reference": R / "mrcd/context_eval/transcript_reference.py",
    "survival": R / "mrcd/context_eval/survival.py",
    "reference": R / "mrcd/context_eval/reference.py",
    "textfeat": R / "mrcd/context_eval/textfeat.py",
    "decode": R / "mrcd/asr_recovery/decode.py",
    "stats": R / "mrcd/asr_recovery/stats.py",
    "s3_features": R / "mrcd/service/s3_features.py",
    "fusion_text": R / "mrcd/fusion_text.py",
    "rep_ctx_p1_route": R / "mrcd/service/experimental/rep_ctx_p1_route.json",
    "engine_checkpoint": _pin.CKPT,
    "whisper_small_model_bin": _pin.WHISPER / "model.bin",
    "chm150_transcription": _pin.CORPUS_ROOT
    / "data/transcriptions/CHM150.transcription",
}
MANIFEST = _pin.OUT / "PROTOCOL.sha256.json"


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def current() -> dict:
    return {k: _sha(p) for k, p in FROZEN.items()}


def frozen_manifest() -> dict:
    return json.loads(MANIFEST.read_text("utf8"))


def verify_frozen() -> None:
    if not MANIFEST.exists():
        raise SystemExit("protocol not frozen: run freeze.py first")
    want = frozen_manifest()["sha256"]
    got = current()
    bad = sorted(k for k in want if want[k] != got.get(k))
    if bad:
        raise SystemExit(f"frozen files changed after the freeze: {bad}")


def main():
    if MANIFEST.exists():
        raise SystemExit("PROTOCOL.sha256.json already exists: the protocol is frozen")
    man = {
        "frozen_utc": datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "note": "Frozen before any ASR, engine or route output on CHM150 was produced.",
        "paths": {
            k: str(p.relative_to(R)) if p.is_relative_to(R) else p.name
            for k, p in FROZEN.items()
        },
        "sha256": current(),
    }
    MANIFEST.write_text(json.dumps(man, indent=1), "utf8")
    print(json.dumps(man["sha256"], indent=1))


if __name__ == "__main__":
    main()
