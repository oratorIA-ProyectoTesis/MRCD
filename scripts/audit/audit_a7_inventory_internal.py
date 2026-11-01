"""Inventory of the corpus repository folders, SOPHIAS schemas and the internal MRCD data.

Counts files on disk only; makes no claim about content that is not present locally.
"""

import collections
import csv
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from audit_common import MRCD, REPO, save

inv = {}
for d in sorted(p for p in REPO.iterdir() if p.is_dir() and re.match(r"\d\d_", p.name)):
    files = [f for f in d.rglob("*") if f.is_file()]
    ext = collections.Counter(f.suffix.lower() for f in files)
    inv[d.name] = {
        "files": len(files),
        "bytes": sum(f.stat().st_size for f in files),
        "ext_counts": dict(ext.most_common(8)),
    }
inv["03_Spanish_in_Texas"]["transcript_txt_files"] = len(
    list((REPO / "03_Spanish_in_Texas" / "transcripciones").glob("*.txt"))
)
inv["03_Spanish_in_Texas"]["audio_or_video_files"] = len(
    [
        f
        for f in (REPO / "03_Spanish_in_Texas").rglob("*")
        if f.suffix.lower() in {".mp3", ".wav", ".mp4", ".flac"}
    ]
)
inv["04_SOPHIAS"]["recording_files_local"] = len(
    [
        f
        for f in (REPO / "04_SOPHIAS").rglob("*")
        if f.suffix.lower() in {".mp3", ".wav", ".mp4", ".csv", ".json"}
        and "schemas" not in f.parts
    ]
)
inv["15_Bangor_Miami"]["chat_files_local"] = len(
    list((REPO / "15_Bangor_Miami").rglob("*.cha"))
)
inv["11_PRAUTOCAL"]["audio_files_local"] = len(
    [f for f in (REPO / "11_PRAUTOCAL").rglob("*") if f.suffix.lower() == ".wav"]
)
inv["13_PUCP_Cabrera"]["wav_files_local"] = len(
    [f for f in (REPO / "13_PUCP_Cabrera").rglob("*") if f.suffix.lower() == ".wav"]
)

zt = Path(r"C:/zTESIS-2026-02")
dest = {
    "C:/zTESIS-2026-02/corpus-repository_files": len(
        [f for f in (zt / "corpus-repository").rglob("*") if f.is_file()]
    )
    if (zt / "corpus-repository").exists()
    else None,
    "C:/zTESIS-2026-02/SOPHIAS_files": len(
        [f for f in (zt / "SOPHIAS").rglob("*") if f.is_file()]
    )
    if (zt / "SOPHIAS").exists()
    else None,
    "C:/zTESIS-2026-02/SOPHIAS_media_or_data_files": len(
        [
            f
            for f in (zt / "SOPHIAS").rglob("*")
            if f.suffix.lower() in {".mp4", ".wav", ".mp3", ".csv"}
        ]
    )
    if (zt / "SOPHIAS").exists()
    else None,
}
# SOPHIAS schema: events and keys
sch = zt / "SOPHIAS" / "schemas" / "aicofe_events.schema.json"
soph_events = json.loads(sch.read_text(encoding="utf-8")) if sch.exists() else {}

# ---- internal MRCD data ----
D = MRCD / "data"
man = json.loads((D / "dataset_manifest.json").read_text(encoding="utf-8"))[
    "recordings"
]
man4 = json.loads((D / "v4" / "dataset_manifest.json").read_text(encoding="utf-8"))[
    "recordings"
]


def summ(r):
    ec = collections.Counter()
    for x in r:
        ec.update(x.get("event_counts", {}))
    return {
        "recordings": len(r),
        "speakers": len({x["speaker_id"] for x in r}),
        "source_videos": len({x["source_video"] for x in r}),
        "minutes": round(sum(x["duration_s"] for x in r) / 60, 2),
        "event_counts_AUTOMATIC": dict(ec),
        "label_source": dict(collections.Counter(x.get("label_source") for x in r)),
        "licence": dict(collections.Counter(x.get("license") for x in r)),
    }


internal = {"v3_root_manifest": summ(man), "v4_manifest": summ(man4)}

# human tier in annotation_batch_01
tot = collections.Counter()
files = sorted((D / "annotation_batch_01").glob("*.eaf"))
for f in files:
    tree = ET.parse(f)
    for tier in tree.iter("TIER"):
        if tier.get("TIER_ID") == "human_disfluency":
            for v in tier.iter("ANNOTATION_VALUE"):
                if (v.text or "").strip():
                    tot[v.text.strip()] += 1
internal["annotation_batch_01"] = {
    "eaf_files": len(files),
    "human_disfluency_tier_nonempty_values": dict(tot),
}
with open(D / "judge" / "human_check.csv", encoding="utf-8-sig") as fh:
    rows = list(csv.reader(fh))
internal["judge_human_check_csv"] = {
    "rows": len(rows) - 1,
    "rows_with_human_label": sum(1 for r in rows[1:] if len(r) > 4 and r[4].strip()),
}
jc = collections.Counter()
jn = 0
for ln in (D / "judge" / "judgments.jsonl").read_text(encoding="utf-8").splitlines():
    j = json.loads(ln)
    jn += 1
    jc[j["judge"]] += 1
internal["llm_judge_judgments"] = {
    "n": jn,
    "by_label_AUTOMATIC_NOT_HUMAN": dict(jc),
    "models": "gpt-4o (per record)",
}

# case-study interview corpus (directory overridable; names anonymised)
dan = Path(os.environ.get("MRCD_CASE_STUDY_DIR", r"C:/zArnes-tesis/audios/corpus-case-study"))
dd = {}
for f in sorted(dan.glob("*.eaf")):
    tree = ET.parse(f)
    dd[f.name] = {
        t.get("TIER_ID"): sum(
            1 for v in t.iter("ANNOTATION_VALUE") if (v.text or "").strip()
        )
        for t in tree.iter("TIER")
    }
internal["corpus_case_study_eaf"] = dd
internal["corpus_case_study_has_wav"] = (dan / "case_study_interview_001.wav").exists()

# any other EAF under data/ (human tier?)
other = [
    str(p)
    for p in D.rglob("*.eaf")
    if "annotation_batch_01" not in str(p) and "features" not in str(p)
]
internal["other_eaf_outside_batch_and_features"] = other[:10]

save(
    "a7_inventory_internal",
    {
        "repo_folders": inv,
        "destination_checks": dest,
        "sophias_events_schema_description": soph_events.get("description"),
        "internal": internal,
    },
)
