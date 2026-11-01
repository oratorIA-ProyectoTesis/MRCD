"""Independent counts for EPIC v2 (Bologna) support files and EPIC-UdS text."""

import collections
import json
import re
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from audit_common import CODEX, PILOT, REPO, save, sha256

base = PILOT / "source_data" / "EPIC_v2"
if not (base / "08_alignments_v2.0.zip").exists():
    base = REPO / "16_EPIC_v2"
zal = zipfile.ZipFile(base / "08_alignments_v2.0.zip")
zan = zipfile.ZipFile(base / "07_annotations_v2.0.zip")
zme = zipfile.ZipFile(base / "04_metadata_v2.0.zip")
meta = json.loads(zme.read("04_metadata_v2.0/EPIC_v2.0_metadata.json"))
src_meta = {m["source_text_id"]: m for m in meta["sourceText"]}
tgt_meta = {m["target_text_id"]: m for m in meta["targetText"]}

RE_FILLED = re.compile(r"(?:^|\s)-\S+-(?=\s|$)")  # " -ehm- "
RE_TRUNC = re.compile(r"\S+=(?=\s|$)")  # "sin="
RE_MIDW = re.compile(r"\S*\w_\w\S*")  # mid-word pause underdash
RE_PAUSE = re.compile(r"\((\.{1,2}|\d+\.\d+)\)")  # (.) (..) (1.21)


def scan(names, zf):
    st = collections.Counter()
    pause_bins = collections.Counter()
    actors = set()
    events = set()
    per_file_fp = {}
    sents = 0
    sents_fp = 0
    sents_trunc = 0
    sents_pause = 0
    secs = 0.0
    for n in names:
        d = json.loads(zf.read(n))
        stem = Path(n).stem
        m = src_meta.get(stem) or tgt_meta.get(stem)
        if m and "actor_id" in m:
            actors.add(m["actor_id"])
        if m:
            events.add(m.get("event_id"))
        nfp = 0
        for s in d["sentences"]:
            sents += 1
            secs += s.get("duration") or 0
            t = s.get("full_text") or s.get("tokens") or ""
            fp = len(RE_FILLED.findall(t))
            tr = len(RE_TRUNC.findall(t))
            mw = len(RE_MIDW.findall(t))
            ps = RE_PAUSE.findall(t)
            st["filled_pause_marks"] += fp
            st["truncation_marks"] += tr
            st["midword_pause_marks"] += mw
            st["silent_pause_marks"] += len(ps)
            for p in ps:
                pause_bins[
                    "0.2-0.5"
                    if p == "."
                    else "0.5-1.0"
                    if p == ".."
                    else ">1.0(explicit seconds)"
                ] += 1
            sents_fp += fp > 0
            sents_trunc += tr > 0
            sents_pause += len(ps) > 0
            nfp += fp
        per_file_fp[stem] = nfp
    return {
        "files": len(names),
        "sentences_with_time": sents,
        "sentence_seconds_sum": round(secs, 1),
        "mark_counts": dict(st),
        "pause_bins": dict(pause_bins),
        "sentences_with_filled_pause": sents_fp,
        "sentences_with_truncation": sents_trunc,
        "sentences_with_silent_pause": sents_pause,
        "files_with_filled_pause": sum(v > 0 for v in per_file_fp.values()),
        "distinct_actor_ids": len(actors),
        "distinct_events_sessions": len(events - {None}),
    }


names = [n for n in zal.namelist() if n.endswith(".json") and "__MACOSX" not in n]
es_src = [n for n in names if n.endswith("-org-es.json")]
es_tgt = [n for n in names if re.search(r"-int-(en|it)-es\.json$", n)]
res = {
    "base": str(base),
    "zip_sha256": {
        k: sha256(base / k)
        for k in (
            "08_alignments_v2.0.zip",
            "07_annotations_v2.0.zip",
            "04_metadata_v2.0.zip",
        )
    },
    "alignment_files_total": len(names),
    "spanish_source_originals_alignments": scan(es_src, zal),
    "spanish_interpretation_targets_alignments": scan(es_tgt, zal),
    "metadata_manual_annotation_types": meta["annotation"],
    "alignment_metadata": meta["alignment"],
}

# POS-tag level counts (annotations zip) on Spanish originals: custom tags
tagc = collections.Counter()
for n in zan.namelist():
    if n.endswith("-org-es.txt") and "source/" in n:
        for line in zan.read(n).decode("utf-8", errors="replace").splitlines():
            parts = line.split("\t")
            if len(parts) == 3 and parts[2] in {
                "SPAUSE",
                "FPAUSE",
                "TRUNCATION",
                "MIDWORDPAUSE",
                "MISPRONUNCIATION",
                "CORRECTION",
            }:
                tagc[parts[2]] += 1
res["spanish_source_pos_custom_tag_counts"] = dict(tagc)

# local media check
vid = list(base.glob("*.mp4"))
res["local_media_files"] = [p.name for p in vid]
res["local_media_bytes"] = [p.stat().st_size for p in vid]

# Codex 136-segment reference set
refs = json.loads(
    (CODEX / "reproduction" / "v2" / "references.json").read_text(encoding="utf-8")
)
res["codex_reference_set"] = {
    "n": len(refs),
    "positive_FPAUSE": sum(r["source_FPAUSE_present"] == 1 for r in refs),
    "negative": sum(r["source_FPAUSE_present"] == 0 for r in refs),
    "distinct_actor_ids": len({r["actor_id"] for r in refs}),
    "splits": dict(collections.Counter(r["split"] for r in refs)),
    "actors_with_positive": len(
        {r["actor_id"] for r in refs if r["source_FPAUSE_present"] == 1}
    ),
    "negatives_containing_truncation_or_midword": sum(
        r["source_FPAUSE_present"] == 0
        and bool(RE_TRUNC.search(r["source_text"]) or RE_MIDW.search(r["source_text"]))
        for r in refs
    ),
    "negatives_containing_silent_pause": sum(
        r["source_FPAUSE_present"] == 0 and bool(RE_PAUSE.search(r["source_text"]))
        for r in refs
    ),
}
# per-actor positives for the 'unique-speaker' caveat
pa = collections.Counter(r["actor_id"] for r in refs if r["source_FPAUSE_present"] == 1)
res["codex_reference_set"]["positives_per_actor"] = dict(pa)

# EPIC-UdS
zu = zipfile.ZipFile(PILOT / "source_data" / "EPIC_UdS" / "EPIC-UdS.zip")
vrt = [n for n in zu.namelist() if n.endswith("epic_uds_es_v2_1-6.vrt")][0]
txt = zu.read(vrt).decode("utf-8", errors="replace")
n_text = txt.count("<text ")
n_s = txt.count("<s>")
tokens = [l for l in txt.splitlines() if l and not l.startswith("<")]
cnt_eh = sum(
    1
    for l in tokens
    if l.split("\t")[0].lower() in {"eh", "ehm", "em", "mm", "uh", "ah"}
)
trunc_like = sum(1 for l in tokens if re.match(r"^\S+[-=]$", l.split("\t")[0]))
res["epic_uds_es"] = {
    "texts": n_text,
    "sentences": n_s,
    "token_lines": len(tokens),
    "tokens_eh_ehm_like": cnt_eh,
    "tokens_ending_dash_or_equals": trunc_like,
    "has_audio_video": False,
    "note": "cleaned text + automatic morphosyntax; no pause/filler/truncation marks => no disfluency reference",
}
save("a4_epic", res)
