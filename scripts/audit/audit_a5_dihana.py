"""Independent counts for DIHANA (user turns: text marks + per-turn 8 kHz telephone audio, Spanish)."""

import collections
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from audit_common import REPO, save

root = REPO / "01_DIHANA" / "extraido" / "dihana"
xmls = sorted((root / "data").glob("*/*/*.xml"))
tag_counts = collections.Counter()  # "tag:tipo"
turn_with = collections.Counter()
spk_with = collections.defaultdict(set)
speakers = set()
turns = 0
turns_audio_missing = 0
turns_with_any_mark = 0
dialogues = 0
annotators = collections.Counter()
n_turn_audio = 0
per_speaker_turns = collections.Counter()
parse_err = 0
# mark taxonomy used for presence-by-turn
SINGLE = {
    "f": "filled_pause",
    "a": "lengthening",
    "p": "silent_pause",
    "b": "abandoned_segment",
}
GROUPS = {
    "filled_pause_f": {"f:a", "f:e", "f:m", "f:sin_identificar"},
    "lengthening_a": {"a"},
    "repetition_r": {"r:repetición"},
    "reformulation_r_subst_ins_del": {"r:sustitución", "r:inserción", "r:borrado"},
    "abandoned_segment_b": {"b"},
    "cut_word_l": {"l:cortada"},
    "silent_pause_p": {"p"},
    "discourse_filler_d_rellenar": {"d:rellenar"},
}
grp_turn = collections.Counter()
grp_spk = collections.defaultdict(set)
grp_dlg = collections.defaultdict(set)
spk_grp_turns = collections.Counter()
spk_turns = collections.Counter()
for x in xmls:
    try:
        tree = ET.parse(x)  # honours the ISO-8859-1 declaration
    except ET.ParseError:
        parse_err += 1
        continue
    dialogues += 1
    rt = tree.getroot()
    annotators[rt.get("anotador")] += 1
    spk = None
    for h in rt.iter("hablante"):
        if h.get("id") == "usuario":
            spk = h.get("nombre")
    speakers.add(spk)
    for tn in rt.iter("turno"):
        if tn.get("hablante") != "usuario":
            continue
        turns += 1
        per_speaker_turns[spk] += 1
        wav = x.parent / (tn.get("fichero_audio_ss") or "")
        if tn.get("fichero_audio_ss") and wav.exists():
            n_turn_audio += 1
        else:
            turns_audio_missing += 1
        t = tn.find("t")
        found = set()
        for el in t.iter():
            if el is t:
                continue
            tg = el.tag
            ty = el.get("tipo")
            key = f"{tg}:{ty}" if ty else tg
            tag_counts[key] += 1
            found.add(key)
            if tg == "r":
                found.add("r:ANY")
        if found:
            turns_with_any_mark += 1
        for k in found:
            turn_with[k] += 1
            spk_with[k].add(spk)
        for g, members in GROUPS.items():
            if found & members:
                grp_turn[g] += 1
                grp_spk[g].add(spk)
                grp_dlg[g].add(x.name)
                spk_grp_turns[(spk, g)] += 1

guides = {}
for g in sorted((root / "guides").glob("*.gui")):
    guides[g.name] = [l.strip() for l in g.read_text().splitlines() if l.strip()]
allg = [s for v in guides.values() for s in v]
out = {
    "root": str(root),
    "xml_dialogues_parsed": dialogues,
    "xml_parse_errors": parse_err,
    "annotator_attr_counts": dict(annotators),
    "distinct_user_speakers": len(speakers - {None}),
    "user_turns": turns,
    "user_turns_with_audio_file": n_turn_audio,
    "user_turns_without_audio_file": turns_audio_missing,
    "user_turns_with_any_mark": turns_with_any_mark,
    "tag_counts": dict(sorted(tag_counts.items())),
    "turns_with_tag": dict(sorted(turn_with.items())),
    "speakers_with_tag": {k: len(v) for k, v in sorted(spk_with.items())},
    "group_turns": dict(grp_turn),
    "group_speakers": {k: len(v) for k, v in grp_spk.items()},
    "group_dialogues": {k: len(v) for k, v in grp_dlg.items()},
    "fold_support_turns_by_group": {
        gn: {g: sum(spk_grp_turns[(s, g)] for s in sp) for g in GROUPS} for gn, sp in guides.items()
    },
    "fold_turns_total": {gn: sum(per_speaker_turns[s] for s in sp) for gn, sp in guides.items()},
    "guide_partitions": {k: len(v) for k, v in guides.items()},
    "guide_speakers_distinct": len(set(allg)),
    "guide_speakers_match_xml_speakers": set(allg) == (speakers - {None}),
    "turn_audio_note": "per-turn 8 kHz telephone .ul; XML has NO event times (no start/end attributes) -> text/turn level only",
    "licence": "CC BY 4.0 (Zenodo record 5572470, 2021)",
    "time_attributes_found": sorted(
        {
            a
            for x in xmls[:50]
            for el in ET.parse(x).iter()
            for a in el.attrib
            if re.search("ini|fin|start|end|t0|t1|time", a, re.I)
        }
    ),
}
save("a5_dihana", out)
