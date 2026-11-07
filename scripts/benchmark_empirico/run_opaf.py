"""Run the OpenOPAF filled-pause detector (Praat scripts of De Jong & Pacilly, GPL v3) on the DIHANA frame audio.

Frozen mapping (PROTOCOL.md section 3.3): the call mirrors OpenOPAF's own AudioExtractor command
(pre-processing None, silence -25 dB, min dip 2 dB, min pause 0.3 s, filled pauses on, language English,
filled-pause threshold 1.5); prediction = nrFP >= 1. Output: unit_id, nsyll, nrFP, tFP (no audio, no text).
Runs in /path/to/_venv_opaf (praat-parselmouth 0.4.7, Praat 6.1.38). Reads only frame audio of folds 1-4.
usage: /path/to/_venv_opaf/Scripts/python.exe run_opaf.py
"""

import csv
import json
import os
import sys
from pathlib import Path

import parselmouth
from parselmouth.praat import run_file

MAIN = Path("/path/to/mrcd")
OUT = Path(__file__).resolve().parents[2] / "results" / "benchmark_empirico_20261107"
SCRIPT = Path("/path/to/_ext/OpenOPAF/application/praat/processing.praat")
WAV = MAIN / "results/baselines_20261030/dihana/wav16k"
FRAME = MAIN / "results/asr_recovery_20261031/subset.json"
ARGS = (
    "None",
    -25,
    2,
    0.3,
    "yes",
    "English",
    1.5,
    "Praat Info window",
    "OverWriteData",
    "no",
)

ids = [u["unit_id"] for u in json.loads(FRAME.read_text("utf8"))["frame"]]
assert len(ids) == 1025
dest = OUT / "runs" / "opaf_flags.csv"
dest.parent.mkdir(parents=True, exist_ok=True)
rows = []
for k, uid in enumerate(ids):
    wav = WAV / f"{uid}.wav"
    try:
        _, info = run_file(
            str(SCRIPT), str(wav), str(WAV) + "/", "unused.csv", *ARGS, capture_output=True
        )
    except parselmouth.PraatError as e:  # ANNEX 1: detector crash = no output for the turn
        rows.append({"unit_id": uid, "nsyll": "NA", "nrFP": "NA", "tFP": "NA", "error": str(e).splitlines()[0][:80]})
        continue
    line = [ln for ln in info.strip().splitlines() if ln and not ln.startswith("name,")][-1]
    c = [x.strip() for x in line.split(",")]
    rows.append({"unit_id": uid, "nsyll": c[1], "nrFP": c[-2], "tFP": c[-1], "error": ""})
    if k % 100 == 0:
        print(k, flush=True)
with open(dest, "w", newline="", encoding="utf8") as f:
    w = csv.DictWriter(f, fieldnames=["unit_id", "nsyll", "nrFP", "tFP", "error"])
    w.writeheader()
    w.writerows(rows)
json.dump(
    {
        "parselmouth": parselmouth.__version__,
        "praat": parselmouth.PRAAT_VERSION,
        "args": list(map(str, ARGS)),
        "python": sys.version.split()[0],
        "omp": os.environ.get("OMP_NUM_THREADS"),
    },
    open(OUT / "runs" / "opaf_env.json", "w"),
    indent=1,
)
print("done", len(rows))
