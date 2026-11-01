"""Independent counts for disfluentES_v1 (text, children reading, Spanish)."""

import collections
import json
import re
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from audit_common import PILOT, REPO, save, sha256

p_repo = REPO / "10_disfluentES" / "disfluentES_v1.json"
p_pilot = PILOT / "source_data" / "disfluentES_v1.json"
path = p_repo if p_repo.exists() else p_pilot
data = json.loads(path.read_text(encoding="utf-8"))

tok = collections.Counter()
fam = collections.Counter()
sent_with = collections.defaultdict(set)  # family -> set of (record idx, sent idx)
n_sent = 0
n_sent_with_label_any_nonC = 0
spk_with = collections.defaultdict(set)
labeled_speakers = set()
ambiguous_pos_sent = 0
FAMS = ["REP", "FILL", "CUT", "PRE", "INS", "S", "D", "PHO", "FSC", "FSP", "RC", "POS"]


def family(t: str) -> str:
    if t.startswith("POS"):
        return "POS*"
    m = re.match(r"([A-Z]+?)(\d*)$", t)
    return m.group(1) if m else t


for ri, r in enumerate(data):
    ds = r["disfluencies"]
    if not ds:
        continue
    labeled_speakers.add(r["speaker"])
    for si, s in enumerate(ds):
        n_sent += 1
        toks = [x.strip() for x in s["lab"].split("|") if x.strip()]
        fs = {family(t) for t in toks}
        tok.update(toks)
        fam.update(family(t) for t in toks)
        if fs - {"C"}:
            n_sent_with_label_any_nonC += 1
        if "POS*" in fs:
            ambiguous_pos_sent += 1
        for f in fs:
            sent_with[f].add((ri, si))
            spk_with[f].add(r["speaker"])

# REP positive sentences exactly as the pilot defined (REP + optional digits, excluding POS_*)
rep_pat = re.compile(r"^REP\d*$")
rep_sent = set()
rep_spk = set()
fill_sent = set()
fill_spk = set()
for ri, r in enumerate(data):
    for si, s in enumerate(r["disfluencies"] or []):
        toks = [x.strip() for x in s["lab"].split("|") if x.strip()]
        if any(rep_pat.match(t) for t in toks):
            rep_sent.add((ri, si))
            rep_spk.add(r["speaker"])
        if any(t == "FILL" for t in toks):
            fill_sent.add((ri, si))
            fill_spk.add(r["speaker"])

out = {
    "path": str(path),
    "sha256": sha256(path),
    "pilot_copy_identical": (sha256(p_pilot) == sha256(path))
    if p_pilot.exists()
    else None,
    "records": len(data),
    "records_with_detailed_labels": len(labeled_speakers),
    "records_without_detailed_labels": len(data) - len(labeled_speakers),
    "unique_speaker_ids_all": len({r["speaker"] for r in data}),
    "unique_speaker_ids_labeled": len(labeled_speakers),
    "labeled_sentences_total": n_sent,
    "sentences_with_POS_tokens": ambiguous_pos_sent,
    "sentences_with_any_non_C_label": n_sent_with_label_any_nonC,
    "token_counts": dict(tok.most_common()),
    "family_counts": dict(fam.most_common()),
    "sentences_by_family": {k: len(v) for k, v in sorted(sent_with.items())},
    "speakers_by_family": {k: len(v) for k, v in sorted(spk_with.items())},
    "REP_strict_positive_sentences": len(rep_sent),
    "REP_strict_positive_speakers": len(rep_spk),
    "FILL_positive_sentences": len(fill_sent),
    "FILL_positive_speakers": len(fill_spk),
    "REP_strict_sentences_excluding_POS_sentences": len(
        {x for x in rep_sent if x in sent_with["REP"]} - sent_with["POS*"]
    ),
    "grades": dict(collections.Counter(r["nivel"] for r in data)),
    "grades_labeled": dict(
        collections.Counter(r["nivel"] for r in data if r["disfluencies"])
    ),
    "notes": [
        "unlabeled records (877) carry no detailed labels: absence of annotation, not a negative",
        "per-token alignment-style labels (C=correct token); label alphabet is reading-error oriented",
    ],
}
# negatives within labeled sentences for REP: sentences with labels but no REP family
out["REP_negative_sentences_within_labeled_records_noPOS"] = n_sent - len(
    sent_with["REP"] | sent_with["POS*"]
)
save("a1_disfluentes", out)
