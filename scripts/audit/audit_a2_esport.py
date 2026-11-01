"""Independent counts for ES-Port (text, call-center dialogues, Spanish)."""

import collections
import json
import re
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from audit_common import PILOT, REPO, save, sha256

zp = REPO / "09_ES_Port" / "esport.zip"
if not zp.exists():
    zp = PILOT / "source_data" / "esport.zip"
z = zipfile.ZipFile(zp)
files = [n for n in z.namelist() if n.endswith(".json")]

# token-level patterns over each raw utterance string
RE_FILL = re.compile(r"<%[^>\s]*>")  # <%eh> (filler/continuer)
RE_UNF = re.compile(r"<[^%>\s][^>]*-\s*>|<%[^>]*->")  # <serv-> / <%eh->  unfinished
RE_ANGLE = re.compile(r"<[^>]*>")
RE_LEN = re.compile(
    r"\S*=\S*"
)  # any whitespace token containing "=" (README: lengthening)

c = collections.Counter()
utts = 0
turns = 0
utt_with = collections.Counter()
dlg_with = collections.defaultdict(set)
fill_words = collections.Counter()
events = collections.Counter()
silence_turns = 0
silence_dlg = set()
for n in files:
    d = json.loads(z.read(n))
    did = d["file-id"] if "file-id" in d else n
    for t in d["turns"]:
        turns += 1
        has_sil = False
        for e in t.get("events", []):
            events[e["name"]] += 1
            if e["name"] == "silence":
                has_sil = True
        if has_sil:
            silence_turns += 1
            silence_dlg.add(did)
        for spk, us in t["speakers"].items():
            for u in us:
                s = u["utterance"]
                utts += 1
                fills = RE_FILL.findall(s)
                allang = RE_ANGLE.findall(s)
                unf = [a for a in allang if a.rstrip(">").rstrip().endswith("-")]
                fill_only = [a for a in fills if not a.endswith("->")]
                rep = [
                    a
                    for a in allang
                    if not a.startswith("<%")
                    and not a.rstrip(">").rstrip().endswith("-")
                ]
                ln = RE_LEN.findall(s)
                c["filled_or_continuer"] += len(fill_only)
                c["unfinished_or_nonword"] += len(unf)
                c["repetition_or_false_start"] += len(rep)
                c["lengthening_equals"] += len(ln)
                for k, v in (
                    ("filled_or_continuer", fill_only),
                    ("unfinished_or_nonword", unf),
                    ("repetition_or_false_start", rep),
                    ("lengthening_equals", ln),
                ):
                    if v:
                        utt_with[k] += 1
                        dlg_with[k].add(did)
                for a in fill_only:
                    fill_words[a.strip("<%>")] += 1

out = {
    "archive": str(zp),
    "archive_sha256": sha256(zp),
    "dialogue_json_files": len(files),
    "turns": turns,
    "utterances": utts,
    "mark_counts": dict(c),
    "utterances_with_mark": dict(utt_with),
    "dialogues_with_mark": {k: len(v) for k, v in dlg_with.items()},
    "filler_continuer_top": fill_words.most_common(25),
    "filler_continuer_all": dict(fill_words.most_common()),
    "filler_continuer_distinct": len(fill_words),
    "event_counts": dict(events.most_common()),
    "silence_events": events.get("silence", 0),
    "turns_with_silence_event": silence_turns,
    "dialogues_with_silence_event": len(silence_dlg),
    "speaker_identity_note": "speakers are per-dialogue labels (spk1/spk2); not linkable across dialogues -> partition unit = dialogue",
    "negative_coverage_note": "README does not state exhaustive marking; absence of a mark is not an accredited negative",
}
save("a2_esport", out)
