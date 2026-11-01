"""S6 step 1: enumerate EPIC units (aligned sentences of the 74 local videos) with the source FPAUSE label.

Label = presence of the EPIC filled-pause notation ' -xxx- ' in the source sentence. Absence of the mark is a
negative FOR FILLED PAUSE ONLY (negative coverage documented in MATRIZ_CORPUS.md). Sentence times are automatic
(WebMAUS). The sentence text is read only to derive the label and is NOT written out.
"""

import csv
import json
import re
import zipfile
from pathlib import Path

from s6_common import EPIC, WORK, video_paths

RE_F = re.compile(r"(?:^|\s)-\S+-(?=\s|$)")

meta = json.load(
    zipfile.ZipFile(EPIC / "04_metadata_v2.0.zip").open(
        "04_metadata_v2.0/EPIC_v2.0_metadata.json"
    )
)
src = {m["source_text_id"]: m["actor_id"] for m in meta["sourceText"]}
vids = video_paths()
zal = zipfile.ZipFile(EPIC / "08_alignments_v2.0.zip")
rows = []
for n in sorted(zal.namelist()):
    if not n.endswith("-org-es.json") or "__MACOSX" in n or "/._" in n:
        continue
    stem = Path(n).stem
    if stem not in vids:
        continue
    for s in json.loads(zal.read(n))["sentences"]:
        t0, d = float(s["start"]), float(s["duration"])
        text = s.get("full_text") or ""
        rows.append(
            {
                "unit_id": f"{stem}__{s['id']}",
                "video": stem,
                "speaker": src[stem],
                "t0": round(t0, 3),
                "t1": round(t0 + d, 3),
                "fpause": int(bool(RE_F.search(text))),
            }
        )
with (WORK / "units.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
pos = sum(r["fpause"] for r in rows)
print(
    json.dumps(
        {
            "units": len(rows),
            "fpause": pos,
            "videos": len({r["video"] for r in rows}),
            "speakers": len({r["speaker"] for r in rows}),
        }
    )
)
