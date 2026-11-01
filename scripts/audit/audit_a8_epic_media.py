"""EPIC v2: which Spanish source videos exist locally, and which map to the 136-segment Codex set."""

import collections
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from audit_common import CODEX, PILOT, REPO, save

base = REPO / "16_EPIC_v2"
vids = sorted(base.rglob("*.mp4"))
stems = {v.stem for v in vids}
info = []
tot_s = 0.0
bad = 0
for v in vids:
    try:
        r = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration:stream=codec_type,width,height",
                "-of",
                "json",
                str(v),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        j = json.loads(r.stdout)
        dur = float(j["format"]["duration"])
        has_v = any(s.get("codec_type") == "video" for s in j["streams"])
        has_a = any(s.get("codec_type") == "audio" for s in j["streams"])
        tot_s += dur
        info.append(
            {
                "file": v.name,
                "duration_s": round(dur, 2),
                "video": has_v,
                "audio": has_a,
            }
        )
    except Exception as e:  # noqa: BLE001
        bad += 1
        info.append({"file": v.name, "error": str(e)[:80]})

# map the Codex 136 reference texts to EPIC recordings by exact sentence-text match
zal = zipfile.ZipFile(PILOT / "source_data" / "EPIC_v2" / "08_alignments_v2.0.zip")
idx = collections.defaultdict(set)
for n in zal.namelist():
    if n.endswith("-org-es.json") and "__MACOSX" not in n:
        stem = Path(n).stem
        for s in json.loads(zal.read(n))["sentences"]:
            idx[(s.get("full_text") or s.get("tokens") or "").strip()].add(stem)
refs = json.loads(
    (CODEX / "reproduction" / "v2" / "references.json").read_text(encoding="utf-8")
)
mapped = 0
amb = 0
unmapped = 0
rec_ids = set()
for r in refs:
    c = idx.get(r["source_text"].strip())
    if not c:
        unmapped += 1
    elif len(c) > 1:
        amb += 1
    else:
        mapped += 1
        rec_ids.add(next(iter(c)))
spanish_listed = json.loads(
    (PILOT / "source_data" / "EPIC_v2" / "spanish_video_inventory.json").read_text(
        encoding="utf-8"
    )
)
out = {
    "local_spanish_source_videos": len(vids),
    "local_media_selected_dir": len([v for v in vids if "media_selected" in v.parts]),
    "total_minutes": round(tot_s / 60, 1),
    "ffprobe_errors": bad,
    "all_have_video_and_audio": all(
        i.get("video") and i.get("audio") for i in info if "error" not in i
    ),
    "spanish_source_videos_published": len(spanish_listed),
    "codex_136_refs_mapped_to_recording": mapped,
    "codex_136_refs_ambiguous_text_match": amb,
    "codex_136_refs_unmapped": unmapped,
    "recordings_covering_codex_136": len(rec_ids),
    "of_which_media_present_locally": len(rec_ids & stems),
    "files": info,
}

# FPAUSE support restricted to recordings whose media exist locally
meta = json.loads(
    zipfile.ZipFile(PILOT / "source_data" / "EPIC_v2" / "04_metadata_v2.0.zip").read(
        "04_metadata_v2.0/EPIC_v2.0_metadata.json"
    )
)
actor_of = {m["source_text_id"]: m["actor_id"] for m in meta["sourceText"]}
RE_F = re.compile(r"(?:^|\s)-\S+-(?=\s|$)")
RE_T = re.compile(r"\S+=(?=\s|$)")
RE_P = re.compile(r"\((?:\.{1,2}|\d+\.\d+)\)")
loc = {"sentences": 0, "fp_sent": 0, "trunc_sent": 0, "pause_sent": 0}
glob_ = {"sentences": 0, "fp_sent": 0}
pos_actor = collections.Counter()
act = set()
for n in zal.namelist():
    if n.endswith("-org-es.json") and "__MACOSX" not in n:
        stem = Path(n).stem
        for s in json.loads(zal.read(n))["sentences"]:
            tx = s.get("full_text") or s.get("tokens") or ""
            glob_["sentences"] += 1
            glob_["fp_sent"] += bool(RE_F.search(tx))
            if stem in stems:
                loc["sentences"] += 1
                act.add(actor_of.get(stem))
                if RE_F.search(tx):
                    loc["fp_sent"] += 1
                    pos_actor[actor_of.get(stem)] += 1
                loc["trunc_sent"] += bool(RE_T.search(tx))
                loc["pause_sent"] += bool(RE_P.search(tx))
out["local_media_support"] = {
    **loc,
    "distinct_actors_local": len(act - {None}),
    "actors_with_fp_sentence": len(pos_actor),
    "fp_sentences_per_actor_top5": pos_actor.most_common(5),
}
out["all_130_originals_support"] = glob_
save("a8_epic_media", out)
