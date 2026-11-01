"""Independent counts for AMERESCO (ELAN/Val.Es.Co. transcripts + audio, Spanish)."""

import collections
import csv
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from audit_common import PILOT, REPO, save, sha256

root = REPO / "12_AMERESCO"
recs = sorted(p.name for p in root.iterdir() if p.is_dir())
per = {}
tot_iv = tot_pos = 0
SPEECH_TIERS_EXCLUDE = {"Observaciones", "Anonimización", "Anonimizacion"}
RE_ALARG = re.compile(r"<alargamiento\s*/>")
RE_RESTART = re.compile(
    r"\S+-(?=\s|$)"
)  # token ending with '-' (Val.Es.Co.: restart / self-interruption)
for r in recs:
    eaf = root / r / f"{r}.eaf"
    tree = ET.parse(eaf)
    ts = {
        t.get("TIME_SLOT_ID"): int(t.get("TIME_VALUE"))
        for t in tree.iter("TIME_SLOT")
        if t.get("TIME_VALUE")
    }
    tiers = {}
    n_iv = n_pos = n_restart_iv = 0
    dur_total = 0.0
    dur_pos = []
    for tier in tree.iter("TIER"):
        tid = tier.get("TIER_ID")
        if tid in SPEECH_TIERS_EXCLUDE:
            continue
        ni = npos = 0
        for a in tier.iter("ALIGNABLE_ANNOTATION"):
            v = a.findtext("ANNOTATION_VALUE") or ""
            if not v.strip():
                continue
            ni += 1
            s, e = ts[a.get("TIME_SLOT_REF1")], ts[a.get("TIME_SLOT_REF2")]
            dur_total += (e - s) / 1000
            has = bool(RE_ALARG.search(v))
            npos += has
            if has:
                dur_pos.append((e - s) / 1000)
            if RE_RESTART.search(v):
                n_restart_iv += 1
        tiers[tid] = {"intervals": ni, "alargamiento_intervals": npos}
        n_iv += ni
        n_pos += npos
    per[r] = {
        "eaf": str(eaf),
        "eaf_sha256": sha256(eaf),
        "speech_tiers": tiers,
        "speech_intervals": n_iv,
        "alargamiento_positive_intervals": n_pos,
        "intervals_with_restart_dash": n_restart_iv,
        "speech_seconds": round(dur_total, 1),
        "has_mp3": (root / r / f"{r}.mp3").exists(),
        "has_textgrid": (root / r / f"{r}.TextGrid").exists(),
    }
    tot_iv += n_iv
    tot_pos += n_pos

# Cross-check against the pilot's derived index
idx = PILOT / "source_data" / "AMERESCO" / "source_annotation_index.csv"
rows = list(csv.DictReader(open(idx, encoding="utf-8-sig")))
cand = [r for r in rows if not r["automatic_exclusion_reasons"]]
pos_cand = [r for r in cand if r["source_alargamiento_present"] == "1"]
dur3_8 = [r for r in cand if 3000 <= int(r["end_ms"]) - int(r["start_ms"]) <= 8000]
dur3_8_pos = [r for r in dur3_8 if r["source_alargamiento_present"] == "1"]
excl = collections.Counter()
for r in rows:
    for x in filter(None, r["automatic_exclusion_reasons"].split("|")):
        excl[x] += 1

# the pilot's 32 selected samples
ref = list(csv.DictReader(open(PILOT / "references.csv", encoding="utf-8-sig")))
man = None
import json

man = json.loads((PILOT / "input_manifest.json").read_text(encoding="utf-8"))
items = man if isinstance(man, list) else man.get("items") or man.get("samples") or man
rec_of = collections.Counter()
if isinstance(items, list):
    for it in items:
        rec_of[it.get("recording_id") or it.get("source_recording") or "?"] += 1

out = {
    "root": str(root),
    "recordings": per,
    "total_speech_intervals": tot_iv,
    "total_alargamiento_positive_intervals": tot_pos,
    "total_audio_minutes_mp3_not_measured": None,
    "index_csv": str(idx),
    "index_rows": len(rows),
    "index_positive": sum(r["source_alargamiento_present"] == "1" for r in rows),
    "candidates_after_automatic_filters": len(cand),
    "candidate_positive": len(pos_cand),
    "candidates_3_to_8s": len(dur3_8),
    "candidates_3_to_8s_positive": len(dur3_8_pos),
    "exclusion_reason_counts": dict(excl),
    "pilot_refs_n": len(ref),
    "pilot_refs_positive": sum(r["source_present"] == "1" for r in ref),
    "pilot_manifest_recording_counts": dict(rec_of),
    "pilot_distinct_speaker_tiers": len({(it["recording_id"], it["source_tier"]) for it in items}),
    "pilot_tier_pos_neg": {
        "positives_by_recording": dict(collections.Counter(
            it["recording_id"] for it, rr in zip(items, ref) if rr["source_present"] == "1")),
    },
    "protocol_notes": {
        "alargamiento": "<alargamiento/> marks vocalic AND consonantic lengthening, any function (protocol sec. 4.2)",
        "pauses": "Val.Es.Co. pauses are NOT marked (protocol sec. 4.4: alignment with ELAN makes them unnecessary)",
        "restarts": "dash '-' marks restarts/self-interruptions; repeated words are transcribed as many times as pronounced",
        "times": "interval times are human-aligned ELAN speech turns (segment level), not event boundaries",
    },
    "licence": "CC BY-NC-SA 4.0 (declared by project site; see pilot LICENSE_NOTICES.md)",
}
save("a3_ameresco", out)
