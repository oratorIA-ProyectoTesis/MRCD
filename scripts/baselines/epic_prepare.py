"""EPIC v2 filler (FPAUSE presence per aligned sentence): the 136-sentence Codex reference set = DEVELOPMENT (already inspected in v2/v3), NOT a test.
Reuses existing clips (wav with real context) + existing labels (source_FPAUSE_present from EPIC notation). Segment times are automatic (WebMAUS).
Label meaning: presence of ' -ehm- ' notation; absence = no filled pause marked (NOT a negative for other disfluencies).
"""

import os
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import soundfile as sf
from bl_common import OUT, write_config, sha256_file

C = Path(r"C:\zArnes-tesis\mrcd-codex\MRCD_Plan_Avance_20261030\reproduction\v2")
D = OUT / "epic"
D.mkdir(parents=True, exist_ok=True)
refs = {
    r["sample_id"]: r for r in json.loads((C / "references.json").read_text("utf8"))
}
inp = json.loads((C / "inputs.json").read_text("utf8"))
man = []
bad = 0
for i in inp:
    i = dict(i)
    if (
        "wav" not in i
    ):  # the 34 `new_epic_*` clips live in mrcd_calibracion_20261029/clips
        i["wav"] = str(
            Path(
                (os.environ.get("MRCD_CALIB_ROOT", r"<CALIB_ROOT>") + r"\clips")
            )
            / (i["sample_id"] + ".wav")
        )
    i["wav_sha256"] = sha256_file(i["wav"])
    info = sf.info(i["wav"])
    if info.samplerate != 16000 or info.channels != 1:
        bad += 1
    r = refs[i["sample_id"]]
    man.append(
        {
            "unit_id": i["sample_id"],
            "wav": i["wav"],
            "wav_sha256": i["wav_sha256"],
            "group": r["actor_id"],
            "split": r["split"],
            "target_start_s": i["target_start_s"],
            "target_end_s": i["target_start_s"] + i["target_duration_s"],
        }
    )
refcsv = [
    {
        "unit_id": r["sample_id"],
        "group": r["actor_id"],
        "split": r["split"],
        "filler_present": r["source_FPAUSE_present"],
        "source_text": r["source_text"],
    }
    for r in refs.values()
]
(D / "manifest.json").write_text(json.dumps(man, indent=1))
with (D / "references.csv").open("w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(refcsv[0]))
    w.writeheader()
    w.writerows(refcsv)
write_config(
    D / "prepare_run",
    "epic_prepare",
    __file__,
    {"n": len(man), "wav_not_16k_mono": bad, "role": "DEVELOPMENT (not a test)"},
    {
        "references.json": sha256_file(C / "references.json"),
        "inputs.json": sha256_file(C / "inputs.json"),
        "manifest.json": sha256_file(D / "manifest.json"),
    },
)
print(len(man), "bad format", bad)
