"""Independent inventory for NOCANDO (Spanish narrations; MP3 + transcripts + TSA + 14 TextGrids)."""

import collections
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from audit_common import REPO, save

root = REPO / "02_NOCANDO_ES"


def read_tg(p: Path) -> str:
    b = p.read_bytes()
    if b[:2] in (b"\xfe\xff", b"\xff\xfe"):
        return b.decode("utf-16", errors="replace")
    return b.decode("latin-1")


audio = sorted((root / "audio").glob("*.mp3"))
ts = sorted((root / "transcripciones").glob("*_ts.txt"))
tsa = sorted((root / "anotaciones").glob("*_tsa.txt"))
tgs = sorted((root / "alineamientos").glob("*.TextGrid"))
speakers = {re.sub(r"_es_\d$", "", p.stem) for p in audio}
spk_ts = {re.sub(r"_es_\d_ts$", "", p.stem) for p in ts}

tg_info = {}
for p in tgs:
    t = read_tg(p).replace("\r", "")
    tiers = re.findall(r'name = "([^"]*)"', t)
    ivs = re.findall(
        r'xmin = ([\d.eE+-]+)\s*\n\s*xmax = ([\d.eE+-]+)\s*\n\s*text = "([^"]*)"', t
    )
    tg_info[p.name] = {
        "tiers": tiers,
        "intervals_total": len(ivs),
        "intervals_nonempty": sum(1 for i in ivs if i[2].strip()),
    }

SEG = re.compile(r"<segment\b[^>]*>(.*?)</segment>", re.S)
marks = collections.Counter()
nseg = 0
nseg_with = collections.Counter()
for p in ts:
    t = p.read_text(encoding="utf-8", errors="replace")
    for s in SEG.findall(t):
        nseg += 1
        s = s.strip()
        flags = {
            "underscore_suffix": bool(re.search(r"\w_(?=\s|$|[,.?;])", s)),
            "hm_token": bool(re.search(r"\b(hm+|eh+|em+|mm+)\b", s, re.I)),
            "hash": "#" in s,
            "square_bracket_word": bool(re.search(r"\[[^\]]+\]", s)),
        }
        for k, v in flags.items():
            if v:
                nseg_with[k] += 1
        marks["underscore_suffix_tokens"] += len(re.findall(r"\w_(?=\s|$|[,.?;])", s))
        marks["hm_eh_tokens"] += len(re.findall(r"\b(?:hm+|eh+|em+|mm+)\b", s, re.I))

nocans = collections.Counter()
for p in tsa:
    t = p.read_text(encoding="utf-8", errors="replace")
    for v in re.findall(r'nocans="([^"]*)"', t):
        nocans[v] += 1

out = {
    "root": str(root),
    "mp3_files": len(audio),
    "mp3_bytes": sum(p.stat().st_size for p in audio),
    "transcripts_ts": len(ts),
    "annotations_tsa": len(tsa),
    "textgrids": len(tgs),
    "audio_distinct_speakers": len(speakers),
    "transcript_distinct_speakers": len(spk_ts),
    "speaker_ids_audio": sorted(speakers),
    "textgrid_files": tg_info,
    "textgrid_total_nonempty_intervals": sum(
        v["intervals_nonempty"] for v in tg_info.values()
    ),
    "transcript_segments": nseg,
    "segments_with": dict(nseg_with),
    "token_marks": dict(marks),
    "tsa_nocans_label_counts_top": nocans.most_common(15),
    "licence": "CC BY-NC-SA 3.0 (declared on NOCANDO site)",
    "convention_documented_locally": False,
    "note": "'_' / 'hm' / '#' are transcription traits; their convention is NOT documented in the downloaded files -> not accepted as labels",
}
save("a6_nocando", out)
