"""DIHANA repetition/revision label audit, DEV ONLY (folds 1-3 = dev_units_frozen.json 'dev_units').

Read-only on corpus + frozen ASR cache. Writes NEW files in results/auditoria_etiquetas_20261030/.
Fold 5: only descriptive support counts (turns/speakers with each mark), nothing else is computed or looked at.
Fold 4 (tune) is not used either.

Nothing here is a label: 'survival in ASR' and 'unmarked repeat' are descriptive text statistics used to
measure how much of a HUMAN mark is recoverable from the frozen ASR text, never as a reference.
"""

import os
import collections
import csv
import difflib
import json
import re
import statistics
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(r"C:\zArnes-tesis\mrcd")
CORP = Path(
    (os.environ.get("MRCD_CORPORA_ROOT", r"<CORPORA_ROOT>") + r"\01_DIHANA\extraido\dihana")
)
OUT = REPO / "results" / "auditoria_etiquetas_20261030"
CACHE = REPO / "results" / "mejoras_20261030" / "asr_cache" / "dihana"
OUT.mkdir(parents=True, exist_ok=True)

frozen = json.loads(
    (REPO / "results/mejoras_20261030/dev_units_frozen.json").read_text("utf8")
)
DEV = set(frozen["dev_units"])
turns_csv = {
    r["turn_id"]: r
    for r in csv.DictReader(
        (REPO / "results/baselines_20261030/dihana/turns.csv").open(
            encoding="utf-8-sig"
        )
    )
}
fold_of = {u: int(r["fold"]) for u, r in turns_csv.items()}
assert {fold_of[u] for u in DEV} == {1, 2, 3}, "dev must be folds 1-3"


def norm(s):
    s = unicodedata.normalize("NFD", s.lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9ñ ]+", " ", s)


def toks(s):
    return norm(s).split()


def raw_toks(s):
    return [t for t in re.split(r"\s+", s.strip()) if t]


def txt(el):
    return " ".join("".join(el.itertext()).split())


def adj_repeat(tk, lo=0, hi=None):
    """adjacent duplicate n-gram (n<=4) or fragment+word inside tk[lo:hi]; returns list of (i,n,kind)."""
    hi = len(tk) if hi is None else hi
    out = []
    for n in (1, 2, 3, 4):
        for i in range(lo, hi - 2 * n + 1):
            if tk[i : i + n] == tk[i + n : i + 2 * n]:
                out.append((i, n, "exact"))
    for i in range(lo, hi - 1):
        a, b = tk[i], tk[i + 1]
        if a != b and 1 <= len(a) < len(b) and b.startswith(a) and len(a) <= 3:
            out.append((i, 1, "fragment"))
    return out


def seq_cov(asr, ref):
    """fraction of ref tokens recovered in-order within asr (matching blocks)."""
    if not ref:
        return 0.0, None
    sm = difflib.SequenceMatcher(None, asr, ref, autojunk=False)
    blocks = [b for b in sm.get_matching_blocks() if b.size]
    cov = sum(b.size for b in blocks) / len(ref)
    region = (blocks[0].a, blocks[-1].a + blocks[-1].size) if blocks else None
    return cov, region


def contiguous(asr, ref):
    n = len(ref)
    return n > 0 and any(asr[i : i + n] == ref for i in range(len(asr) - n + 1))


def load_asr(u):
    p = CACHE / f"{u}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text("utf8"))
    return [w["norm"] for w in d["words"] if w.get("norm")], " ".join(
        w["text"] for w in d["words"]
    )


REPTYPE = {
    "repetición": "repetition",
    "sustitución": "substitution",
    "inserción": "insertion",
    "borrado": "deletion",
}

marks = []  # one row per <r> element (dev)
turn_rows = []  # one row per dev turn
spk_turn = collections.Counter()
spk_rep = collections.Counter()
spk_rev = collections.Counter()
spk_b = collections.Counter()
spk_any = collections.Counter()
date_turn = collections.Counter()
date_rep = collections.Counter()
date_rev = collections.Counter()
unmarked = []  # human-transcript adjacent repeats NOT inside any <r>/<b>
f5 = {
    "turns": 0,
    "speakers": set(),
    "turns_rep": 0,
    "spk_rep": set(),
    "turns_rev": 0,
    "spk_rev": set(),
    "turns_b": 0,
    "spk_b": set(),
}
total_all = collections.Counter()

for x in sorted((CORP / "data").glob("*/*/*.xml")):
    root = ET.parse(x).getroot()
    fecha = (root.get("fecha") or "")[:20]
    month = " ".join(fecha.split()[1:3]) if fecha else "NA"
    spk = None
    for h in root.iter("hablante"):
        if h.get("id") == "usuario":
            spk = h.get("nombre")
    dlg = x.stem
    for tn in root.iter("turno"):
        if tn.get("hablante") != "usuario":
            continue
        u = f"{dlg}_{tn.get('id')}"
        t = tn.find("t")
        fold = fold_of.get(u)
        rs = [e for e in t.iter("r")]
        bs = [e for e in t.iter("b")]
        rep_types = [REPTYPE.get(e.get("tipo"), e.get("tipo")) for e in rs]
        has_rep = "repetition" in rep_types
        has_rev = any(k in ("substitution", "insertion", "deletion") for k in rep_types)
        if fold == 5:  # descriptive support counts ONLY
            f5["turns"] += 1
            f5["speakers"].add(spk)
            if has_rep:
                f5["turns_rep"] += 1
                f5["spk_rep"].add(spk)
            if has_rev:
                f5["turns_rev"] += 1
                f5["spk_rev"].add(spk)
            if bs:
                f5["turns_b"] += 1
                f5["spk_b"].add(spk)
            continue
        if u not in DEV:  # fold 4 (tune) and anything else: not touched
            continue
        spk_turn[spk] += 1
        date_turn[month] += 1
        if rs or bs:
            spk_any[spk] += 1
        if has_rep:
            spk_rep[spk] += 1
            date_rep[month] += 1
        if has_rev:
            spk_rev[spk] += 1
            date_rev[month] += 1
        if bs:
            spk_b[spk] += 1
        asr = load_asr(u)
        asr_tk, asr_txt = asr if asr else ([], "")
        child_tags = collections.Counter(
            e.tag + (":" + e.get("tipo") if e.get("tipo") else "")
            for e in t.iter()
            if e is not t
        )
        turn_text_marked = ET.tostring(t, encoding="unicode")
        turn_text_marked = re.sub(r"\s+", " ", turn_text_marked).strip()
        turn_rows.append(
            dict(
                unit=u,
                fold=fold,
                speaker=spk,
                month=month,
                n_r=len(rs),
                n_b=len(bs),
                has_rep=int(has_rep),
                has_rev=int(has_rev),
                asr_text=asr_txt,
            )
        )
        # parent map for nesting depth / containment
        parent = {c: p for p in t.iter() for c in p}
        for e in rs:
            ty = REPTYPE.get(e.get("tipo"), e.get("tipo"))
            m = e.find("m")
            s = e.find("s")
            c = e.find("c")
            span_txt = txt(e)
            m_txt = txt(m) if m is not None else ""
            s_txt = txt(s) if s is not None else ""
            c_txt = txt(c) if c is not None else ""
            depth = 0
            p = parent.get(e)
            while p is not None and p is not t:
                if p.tag == "r":
                    depth += 1
                p = parent.get(p)
            inner = collections.Counter(
                d.tag + (":" + d.get("tipo") if d.get("tipo") else "")
                for d in e.iter()
                if d is not e and d.tag not in ("m", "s", "c")
            )
            span_tk = toks(span_txt)
            mt, ct = toks(m_txt), toks(c_txt)
            # relation between reparandum and correction (human text)
            if ty == "repetition":
                if mt == ct:
                    rel = "identical"
                elif mt and ct and ct[: len(mt)] == mt:
                    rel = "m_prefix_of_c(extended)"
                elif (
                    mt
                    and ct
                    and len(ct) >= len(mt)
                    and ct[: len(mt) - 1] == mt[:-1]
                    and ct[len(mt) - 1].startswith(mt[-1])
                ):
                    rel = "fragment_then_word"
                elif mt and ct and set(mt) <= set(ct):
                    rel = "m_subset_of_c"
                else:
                    rel = "other(not literal repeat)"
            else:
                rel = ("signal:" + s_txt) if s_txt else "no_signal"
            raw_ident = raw_toks(m_txt) == raw_toks(c_txt)
            orth_only = (not raw_ident) and mt == ct and bool(mt)
            # ASR survival
            ex = contiguous(asr_tk, span_tk) if asr_tk else False
            cov_span, region = seq_cov(asr_tk, span_tk) if asr_tk else (0.0, None)
            cov_m, _ = seq_cov(asr_tk, mt)
            cov_c, _ = seq_cov(asr_tk, ct)
            reps_in_region = adj_repeat(asr_tk, *region) if region else []
            asr_any_repeat = bool(adj_repeat(asr_tk))
            # for repetitions: ASR has the duplicated unit inside the aligned region
            if ty == "repetition":
                survives = cov_span >= 0.8 and bool(reps_in_region)
            else:
                survives = cov_m >= 0.8 and cov_c >= 0.8 and cov_span >= 0.8
            # surface-only: exact fails on raw tokens but normalised tokens match
            asr_raw = asr_txt.split()
            surface_only = (not contiguous(asr_raw, raw_toks(span_txt))) and ex
            digits_in_asr = any(ch.isdigit() for ch in asr_txt)
            marks.append(
                dict(
                    unit=u,
                    fold=fold,
                    speaker=spk,
                    dialogue=dlg,
                    type=ty,
                    depth=depth,
                    m=m_txt,
                    s=s_txt,
                    has_s=int(s is not None),
                    s_pause=int(s is not None and s.find("p") is not None),
                    c=c_txt,
                    relation=rel,
                    m_ntok=len(mt),
                    c_ntok=len(ct),
                    orth_only_diff=int(orth_only),
                    asr_exact=int(ex),
                    asr_cov_span=round(cov_span, 3),
                    asr_cov_m=round(cov_m, 3),
                    asr_cov_c=round(cov_c, 3),
                    asr_repeat_in_region=int(bool(reps_in_region)),
                    asr_any_repeat_in_turn=int(asr_any_repeat),
                    survives=int(survives),
                    surface_only_mismatch=int(surface_only),
                    inner_marks=json.dumps(dict(inner), ensure_ascii=False),
                    turn_other_marks=json.dumps(dict(child_tags), ensure_ascii=False),
                    asr_text=asr_txt,
                    human_marked=turn_text_marked,
                )
            )
        # unmarked adjacent repeats in the human transcript (marks stripped), outside any <r>/<b> span.
        # Serialise: <a>x</a> -> x (no space, so 'lleg<a>a</a>' stays one word); r/b -> sentinels; p/f -> pause tokens;
        # punctuation kept as separate tokens so 'No , no' and 'precio . Precio' are NOT counted as repeats.
        ser = ET.tostring(t, encoding="unicode")
        ser = re.sub(r"</?a>", "", ser)
        ser = re.sub(r"<r[^>]*>", " SENTOPEN ", ser); ser = ser.replace("</r>", " SENTCLOSE ")
        ser = re.sub(r"<b>", " SENTOPEN ", ser); ser = ser.replace("</b>", " SENTCLOSE ")
        ser = re.sub(r"<p\s*/>", " SENTPAUSE ", ser); ser = re.sub(r"<f[^>]*/>", " SENTFILLED ", ser)
        ser = re.sub(r"<[^>]+>", " ", ser)
        depth_c = 0; words_only = []
        PUNCT = {",", ".", "?", "¿", "!", ";", ":"}
        for tkn in ser.split():
            if tkn == "SENTOPEN": depth_c += 1; continue
            if tkn == "SENTCLOSE": depth_c -= 1; continue
            if tkn in ("SENTPAUSE", "SENTFILLED") or tkn in PUNCT:
                words_only.append((tkn, None)); continue
            for w in toks(tkn): words_only.append((w, depth_c > 0))
        for k in range(len(words_only) - 1):
            (w1, c1), (w2, c2) = words_only[k], words_only[k + 1]
            if c1 is not None and c2 is not None and w1 == w2 and not (c1 and c2):
                unmarked.append(dict(unit=u, fold=fold, speaker=spk, token=w1, between="",
                                     marked_text=turn_text_marked, asr_text=asr_txt))
        # same word either side of a pause / filled pause, neither inside a mark (manual 2.6.1.1: pause between = not a repetition)
        for k in range(len(words_only) - 2):
            (w1, c1), (sep, _), (w2, c2) = words_only[k], words_only[k + 1], words_only[k + 2]
            if c1 is not None and c2 is not None and sep in ("SENTPAUSE", "SENTFILLED") and w1 == w2 and not (c1 and c2):
                unmarked.append(dict(unit=u, fold=fold, speaker=spk, token=w1, between="<p>" if sep == "SENTPAUSE" else "<f>",
                                     marked_text=turn_text_marked, asr_text=asr_txt))

# ------------------------------------------------------------------ aggregate
def pct(a, b):
    return round(100 * a / b, 1) if b else None


rep = [m for m in marks if m["type"] == "repetition"]
rev = [m for m in marks if m["type"] in ("substitution", "insertion", "deletion")]
n_dev = len(turn_rows)
res = {
    "dev_turns": n_dev,
    "dev_speakers": len(spk_turn),
    "fold_of_dev_units": dict(collections.Counter(r["fold"] for r in turn_rows)),
    "asr_cache_missing": sum(1 for r in turn_rows if not r["asr_text"]),
}
res["marks_by_type"] = dict(collections.Counter(m["type"] for m in marks))
res["turns_with_rep"] = sum(r["has_rep"] for r in turn_rows)
res["turns_with_rev"] = sum(r["has_rev"] for r in turn_rows)
res["turns_with_b"] = sum(1 for r in turn_rows if r["n_b"])
res["speakers_with_rep"] = len(spk_rep)
res["speakers_with_rev"] = len(spk_rev)
res["speakers_with_b"] = len(spk_b)
res["speakers_with_any_r_or_b"] = len(spk_any)
res["speakers_with_zero_r_or_b"] = len([s for s in spk_turn if spk_any[s] == 0])
res["speakers_with_zero_rep"] = len([s for s in spk_turn if spk_rep[s] == 0])
res["speakers_with_zero_rev"] = len([s for s in spk_turn if spk_rev[s] == 0])


def block(ms, label):
    n = len(ms)
    d = {"n_marks": n}
    d["relation"] = dict(
        collections.Counter(
            re.sub(r"^signal:.*", "signal:<lexical>", m["relation"]) for m in ms
        )
    )
    d["depth>0 (nested)"] = sum(1 for m in ms if m["depth"] > 0)
    d["contains_nested_r"] = sum(
        1 for m in ms if "r:" in m["inner_marks"] or '"r"' in m["inner_marks"]
    )
    d["orth_only_diff (m vs c differ only by case/accent/punct)"] = sum(
        m["orth_only_diff"] for m in ms
    )
    d["asr_exact_span"] = sum(m["asr_exact"] for m in ms)
    d["asr_cov_span>=0.8"] = sum(1 for m in ms if m["asr_cov_span"] >= 0.8)
    d["asr_cov_span==0"] = sum(1 for m in ms if m["asr_cov_span"] == 0)
    d["phenomenon_survives_in_asr"] = sum(m["survives"] for m in ms)
    d["phenomenon_survives_pct"] = pct(d["phenomenon_survives_in_asr"], n)
    d["surface_only_mismatch(raw fails, normalised matches)"] = sum(
        m["surface_only_mismatch"] for m in ms
    )
    d["asr_any_adjacent_repeat_in_turn"] = sum(m["asr_any_repeat_in_turn"] for m in ms)
    # compound / overlap with other classes: inner marks and turn-level co-marks
    inner = collections.Counter()
    co = collections.Counter()
    for m in ms:
        for k in json.loads(m["inner_marks"]):
            inner[k] += 1
        for k in json.loads(m["turn_other_marks"]):
            if not k.startswith("r:") and k not in ("r", "m", "s", "c"):
                co[k] += 1
    d["inner_marks_inside_span (marks)"] = dict(inner)
    d["turn_co_marks (marks whose turn also has tag)"] = dict(co)
    # turn level
    return d


res["repetition"] = block(rep, "rep")
res["revision"] = block(rev, "rev")
for ty in ("substitution", "insertion", "deletion"):
    res["revision_" + ty] = block([m for m in rev if m["type"] == ty], ty)

# turn-level survival (turn survives if any of its marks of that class survives)
for name, ms, flag in (("repetition", rep, "has_rep"), ("revision", rev, "has_rev")):
    byturn = collections.defaultdict(list)
    for m in ms:
        byturn[m["unit"]].append(m)
    res[f"turn_level_{name}"] = {
        "turns": len(byturn),
        "any_mark_survives": sum(
            1 for v in byturn.values() if any(m["survives"] for m in v)
        ),
        "no_mark_survives": sum(
            1 for v in byturn.values() if not any(m["survives"] for m in v)
        ),
        "asr_any_adjacent_repeat_in_turn": sum(
            1 for v in byturn.values() if any(m["asr_any_repeat_in_turn"] for m in v)
        ),
    }


# consistency across speakers/months
def rates(cnt, den=spk_turn):
    return {s: cnt[s] / den[s] for s in den if den[s] > 0}


for name, cnt in (
    ("rep", spk_rep),
    ("rev", spk_rev),
    ("b", spk_b),
    ("any_r_or_b", spk_any),
):
    r = rates(cnt)
    vals = sorted(r.values())
    tot = sum(cnt.values())
    N = sum(spk_turn.values())
    p = tot / N
    # overdispersion: observed variance of per-speaker counts vs binomial expectation
    obs_var = statistics.pvariance([cnt[s] / spk_turn[s] for s in spk_turn])
    exp_var = statistics.mean([p * (1 - p) / spk_turn[s] for s in spk_turn])
    top = sorted(((cnt[s], spk_turn[s], s) for s in spk_turn), reverse=True)[:5]
    res[f"speaker_rate_{name}"] = dict(
        pooled_turn_rate=round(p, 4),
        speakers=len(spk_turn),
        zero_speakers=sum(1 for s in spk_turn if cnt[s] == 0),
        median_speaker_rate=round(statistics.median(vals), 4),
        p90=round(vals[int(0.9 * len(vals))], 4),
        max=round(vals[-1], 4),
        overdispersion_obs_over_binomial_var=round(obs_var / exp_var, 2),
        top5_speakers_turns_with_mark=[(s, c, n) for c, n, s in top],
        share_of_marked_turns_in_top5_speakers=pct(sum(c for c, _, _ in top), tot),
    )
    res[f"month_rate_{name}"] = (
        {
            mth: [cnt_m, date_turn[mth]]
            for mth, cnt_m in sorted(
                {"rep": date_rep, "rev": date_rev}.get(
                    name, collections.Counter()
                ).items()
            )
        }
        if name in ("rep", "rev")
        else None
    )
res["turns_per_month"] = dict(date_turn)

# unmarked adjacent repeats in human transcript
um = collections.Counter()
um_rows = []
for r in unmarked:
    kind = (
        "with_<p>_between"
        if "<p>" in r["between"]
        else ("with_<f>_between" if "<f>" in r["between"] else "contiguous")
    )
    um[kind] += 1
    um_rows.append({**r, "kind": kind})
res["unmarked_adjacent_word_repeat_in_human_transcript"] = dict(um)
res["unmarked_repeat_turns"] = len({r["unit"] for r in unmarked})
res["unmarked_contiguous_turns"] = len(
    {r["unit"] for r in um_rows if r["kind"] == "contiguous"}
)
res["unmarked_contiguous_top_tokens"] = collections.Counter(
    r["token"] for r in um_rows if r["kind"] == "contiguous"
).most_common(15)
# of the contiguous unmarked, how many are in turns with NO rep mark at all
rep_turns = {m["unit"] for m in rep}
res["unmarked_contiguous_in_turns_without_any_rep_mark"] = len(
    {
        r["unit"]
        for r in um_rows
        if r["kind"] == "contiguous" and r["unit"] not in rep_turns
    }
)
res["fold5_descriptive_counts_only"] = {
    k: (len(v) if isinstance(v, set) else v) for k, v in f5.items()
}

(OUT / "dihana_rep_rev_dev.json").write_text(
    json.dumps(res, ensure_ascii=False, indent=1, default=str), "utf8"
)


def wcsv(name, rows):
    if not rows:
        return
    with (OUT / name).open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


wcsv("dihana_rep_rev_marks_dev.csv", marks)
wcsv("dihana_unmarked_repeats_dev.csv", um_rows)
print(
    json.dumps(
        {
            k: res[k]
            for k in (
                "dev_turns",
                "dev_speakers",
                "marks_by_type",
                "turns_with_rep",
                "turns_with_rev",
                "turns_with_b",
                "asr_cache_missing",
            )
        },
        indent=1,
    )
)
