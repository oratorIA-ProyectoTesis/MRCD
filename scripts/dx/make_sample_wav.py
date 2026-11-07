"""Regenerate the synthetic sample ``mrcd/service/web/sample_es.wav`` (no recorded voice, no corpus audio).

Synthesis: espeak-ng 1.51 (Debian bookworm package ``espeak-ng`` 1.51+dfsg-10+deb12u2), voice ``es-419``,
150 words per minute, SSML input; then 22 050 Hz -> 16 000 Hz (``scipy.signal.resample_poly(x, 320, 441)``),
peak-normalised to 0.89 and written as mono PCM16. Needs Docker (for espeak-ng) and the repository venv.

    python scripts/dx/make_sample_wav.py [--out mrcd/service/web/sample_es.wav]

The text contains two fillers ("eh") and one literal repetition ("El motor, el motor").
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[2]
SSML = (
    '<speak>Buenos días. <break time="300ms"/> Eh, hoy les presento el proyecto. <break time="400ms"/> '
    'El motor, el motor escucha el audio y busca muletillas. <break time="400ms"/> '
    'Luego, eh, marca cada repetición con su tiempo. <break time="400ms"/> Así se arma el reporte final.</speak>'
)
IMAGE = "debian:bookworm-slim"
SCRIPT = (
    "set -e; apt-get update -qq >/dev/null; apt-get install -y -qq espeak-ng >/dev/null 2>&1; "
    "espeak-ng --version; espeak-ng -v es-419 -s 150 -m -f /work/sample.ssml -w /work/raw.wav"
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out", default=str(ROOT / "mrcd" / "service" / "web" / "sample_es.wav")
    )
    args = ap.parse_args(argv)
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        (work / "sample.ssml").write_text(SSML, encoding="utf-8")
        subprocess.run(
            ["docker", "run", "--rm", "-v", f"{work}:/work", IMAGE, "sh", "-c", SCRIPT],
            check=True,
        )
        with wave.open(str(work / "raw.wav"), "rb") as w:
            assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (
                1,
                2,
                22050,
            )
            x = (
                np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(
                    np.float64
                )
                / 32768.0
            )
    y = resample_poly(x, 320, 441)
    y = y / np.max(np.abs(y)) * 0.89
    pcm = np.clip(np.round(y * 32767), -32768, 32767).astype("<i2")
    with wave.open(args.out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm.tobytes())
    data = Path(args.out).read_bytes()
    print(args.out, f"{len(pcm) / 16000:.2f} s", hashlib.sha256(data).hexdigest())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
