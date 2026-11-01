"""Repetition / revision / false-start marks in disfluentES, ES-Port, AMERESCO and EPIC (source files, read-only).
DIHANA is handled in b1/b2 (dev folds 1-3 only). Counts here do not involve DIHANA fold 5.
Heuristic structure statistics (is the marked span a literal copy of the next words?) describe HUMAN marks; they create no labels."""

import os
import collections
import glob
import hashlib
import json
import re
import unicodedata
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(
    os.environ.get("MRCD_CORPORA_ROOT", r"<CORPORA_ROOT>")
)
OUT = Path(r"C:\zArnes-tesis\mrcd\results\auditoria_etiquetas_20261030")
MEJ = Path(r"C:\zArnes-tesis\mrcd\results\mejoras_20261030")
BASE = Path(r"C:\zArnes-tesis\mrcd\results\baselines_20261030")


def nrm(s):
    s = unicodedata.normalize("NFD", s.lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9ñ ]+", " ", s).split()


def h(s):
    return hashlib.sha1(s.encode()).hexdigest()


res = {}
# ------------------------------------------------------------------ disfluentES
d = json.loads((REPO / "10_disfluentES" / "disfluentES_v1.json").read_text("utf8"))
fam = collections.Counter()
sent_fam = collections.defaultdict(set)
spk_fam = collections.defaultdict(set)
pre_kind = collections.Counter()
rep_kind = collections.Counter()
ex_pre = []
ex_rep = []
both = set()
n_sent = 0
for ri, r in enumerate(d):
    for si, s in enumerate(r["disfluencies"] or []):
        n_sent += 1
        lab = [x.strip() for x in s["lab"].split("|")]
        man = [x.strip() for x in s["man"].split("|")]
        ref = [x.strip() for x in s["ref"].split("|")]
        pos = any(l.startswith("POS") for l in lab)
        fs = set()
        for i, l in enumerate(lab):
            m = re.match(r"(REP|PRE|CUT|FILL)\d*$", l)
            if not m or pos:
                continue
            f = m.group(1)
            fs.add(f)
            fam[f] += 1
            w = nrm(man[i])[0] if i < len(man) and nrm(man[i]) else ""
            if f == "PRE":
                nxt = next(
                    (nrm(man[j])[0] for j in range(i + 1, len(man)) if nrm(man[j])), ""
                )
                pre_kind[
                    "fragment_of_next_word (PRECUT)"
                    if w and nxt.startswith(w) and w != nxt
                    else "different_word (PRESUB-like)"
                ] += 1
            if f == "REP":
                near = [nrm(man[j])[0] for j in range(max(0, i - 4), min(len(man), i + 5)) if j != i and nrm(man[j])]
                rep_kind[
                    "word_also_present_within_4_slots(literal copy nearby)"
                    if w in near
                    else "no_copy_within_4_slots"
                ] += 1
        for f in fs:
            sent_fam[f].add((ri, si))
            spk_fam[f].add(r["speaker"])
        if {"REP", "PRE"} <= fs:
            both.add((ri, si))
        row = dict(
            speaker=r["speaker"],
            man=" ".join(man),
            ref=" ".join(ref),
            lab=" ".join(lab),
        )
        if "PRE" in fs and len(ex_pre) < 600:
            ex_pre.append(row)
        if "REP" in fs and len(ex_rep) < 600:
            ex_rep.append(row)
res["disfluentES"] = dict(
    labeled_sentences=n_sent,
    token_counts_excl_POS_sentences=dict(fam),
    sentences={k: len(v) for k, v in sent_fam.items()},
    speakers={k: len(v) for k, v in spk_fam.items()},
    sentences_with_both_REP_and_PRE=len(both),
    PRE_kind=dict(pre_kind),
    REP_kind=dict(rep_kind),
    examples_REP=sorted(ex_rep, key=lambda r: h(r["man"]))[:3],
    examples_PRE=sorted(ex_pre, key=lambda r: h(r["man"]))[:3],
    note="labels sit on aligned word slots (ref|man|lab); REP = verbatim re-reading of a prompt word, PRE = abandoned attempt corrected, CUT = truncated and left uncorrected. No audio: text-only, children reading.",
)

# ------------------------------------------------------------------ ES-Port
cnt = collections.Counter()
ex = collections.defaultdict(list)
RE = re.compile(r"<(?!%)([^<>]*)>")
for f in sorted(glob.glob(str(REPO / "09_ES_Port/extracted/esport/*.json"))):
    dd = json.loads(Path(f).read_text("utf8"))
    for t in dd["turns"]:
        for spk, us in t["speakers"].items():
            for u in us:
                tx = u["utterance"]
                for m in RE.finditer(tx):
                    inner = m.group(1).strip()
                    if inner.endswith("-"):
                        cnt["unfinished<x->"] += 1
                        continue
                    cnt["rep_or_false_start<...>"] += 1
                    toks = nrm(inner)
                    after = tx[m.end() :]
                    after = re.sub(
                        r"<%[^>]*>|<[^<>]*->|@\[[^\]]*\]|\[[^\]]*\]", " ", after
                    )  # skip fillers/fragments/events
                    nxt = nrm(re.sub(r"<([^<>]*)>", r"\1", after))[: max(len(toks), 1)]
                    if not toks:
                        k = "empty"
                    elif nxt == toks:
                        k = "literal_repeat(next words identical)"
                    elif (
                        nxt[: len(toks) - 1] == toks[:-1]
                        and nxt[len(toks) - 1 : len(toks)]
                        and nxt[len(toks) - 1].startswith(toks[-1])
                    ):
                        k = "fragment_or_lengthened_variant"
                    elif nxt[:1] == toks[:1]:
                        k = "partial_overlap_restart"
                    else:
                        k = "different_words(false_start_or_revision)"
                    cnt[k] += 1
                    if "=" in inner:
                        cnt["mark_contains_lengthening_="] += 1
                    if "-" in re.sub(r"\s", "", inner):
                        cnt["mark_contains_hyphen"] += 1
                    ex[k].append(tx)
res["ES-Port"] = dict(
    counts=dict(cnt),
    examples={k: sorted(set(v), key=h)[:3] for k, v in ex.items()},
    note="'< >' wraps the FIRST copy (reparandum); no separate repetition vs false-start label. Text only.",
)

# ------------------------------------------------------------------ AMERESCO
root = REPO / "12_AMERESCO"
asr_dir = MEJ / "asr_cache" / "ameresco"
mani = {
    m["unit_id"]: m
    for m in json.loads((BASE / "ameresco/manifest.json").read_text("utf8"))
}
cnt = collections.Counter()
exs = []
RE_DASH = re.compile(r"\S+-(?=\s|$)")
ann_text = {}
for conv in sorted(p.name for p in root.iterdir() if p.is_dir()):
    tree = ET.parse(root / conv / f"{conv}.eaf")
    for tier in tree.iter("TIER"):
        if tier.get("TIER_ID") in ("Observaciones", "Anonimización", "Anonimizacion"):
            continue
        for a in tier.iter("ALIGNABLE_ANNOTATION"):
            v = (a.findtext("ANNOTATION_VALUE") or "").strip()
            if not v:
                continue
            ann_text[(conv, a.get("ANNOTATION_ID"))] = (v, tier.get("TIER_ID"))
            cnt["intervals"] += 1
            ds = RE_DASH.findall(v)
            if not ds:
                continue
            cnt["intervals_with_dash"] += 1
            tk = v.split()
            if any(
                tk[i] == tk[i + 1] for i in range(len(tk) - 1) if tk[i].endswith("-")
            ):
                cnt["dash_fragment_repeated(la- la-)"] += 1
            elif any(
                i + 1 < len(tk)
                and tk[i].endswith("-")
                and nrm(tk[i + 1])[:1]
                and nrm(tk[i])[:1]
                and nrm(tk[i + 1])[0].startswith(nrm(tk[i])[0])
                for i in range(len(tk))
            ):
                cnt["dash_fragment_then_completed_word"] += 1
            else:
                cnt["dash_other(restart_or_interruption_without_lexical_trace)"] += 1
            if "<alargamiento/>" in v:
                cnt["dash_and_alargamiento_same_interval"] += 1
            # adjacent plain repeats without dash
        tk = v.split() if v else []
        # literal adjacent word repeat without dash (unlabeled in protocol)
    # plain repeats
for (conv, aid), (v, tier) in ann_text.items():
    tk = [w for w in nrm(re.sub(r"<[^>]*>", " ", v))]
    if any(tk[i] == tk[i + 1] for i in range(len(tk) - 1)) and not RE_DASH.search(v):
        cnt[
            "intervals_with_plain_adjacent_repeat_no_dash(protocol: transcribe each repetition, no tag)"
        ] += 1
# the 98 evaluated segments
ev = collections.Counter()
for uid, m in mani.items():
    conv = m["group"]
    aid = uid[len(conv) + 1 :]
    v = ann_text.get((conv, aid))
    ap = asr_dir / f"{uid}.json"
    asr = (
        " ".join(w["text"] for w in json.loads(ap.read_text("utf8"))["words"])
        if ap.exists()
        else ""
    )
    if not v:
        ev["unit_without_text_match"] += 1
        continue
    if RE_DASH.search(v[0]):
        ev["eval_units_with_dash"] += 1
        tkh = nrm(v[0])
        tka = nrm(asr)
        run = 1; loop = False
        for _a, _b in zip(tka, tka[1:]):
            run = run + 1 if _a == _b else 1
            loop = loop or run >= 6
        has_rep = any(tka[i] == tka[i + 1] for i in range(len(tka) - 1))
        ev["eval_dash_units_where_ASR_has_adjacent_repeat"] += int(has_rep)
        ev["eval_dash_units_ASR_loop_hallucination(>=6 identical tokens)"] += int(loop)
        ev["eval_dash_units_ASR_repeat_excluding_loop_units"] += int(has_rep and not loop)
        exs.append(dict(unit=uid, human=v[0], asr=asr))
res["AMERESCO"] = dict(
    counts=dict(cnt),
    eval_pool_98=dict(ev),
    examples=sorted(exs, key=lambda r: h(r["unit"]))[:4],
    note="Protocol sec.4.3: '-' = reinicios y autointerrupciones sin pausa; repetitions transcribed verbatim, never tagged. No repetition/revision label exists.",
)

# ------------------------------------------------------------------ EPIC v2 (Spanish originals)
zal = zipfile.ZipFile(REPO / "16_EPIC_v2/08_alignments_v2.0.zip")
cnt = collections.Counter()
exs = []
for n in zal.namelist():
    if not n.endswith(".json") or "-org-es" not in n or "__MACOSX" in n or "/._" in n:
        continue
    dd = json.loads(zal.read(n))
    for s in dd["sentences"]:
        t = s.get("full_text") or s.get("tokens") or ""
        cnt["sentences"] += 1
        tk = t.split()
        for i, w in enumerate(tk):
            if re.fullmatch(r"\S+=", w):
                cnt["truncation_marks(x=)"] += 1
                stem = nrm(w)
                if (
                    i + 1 < len(tk)
                    and stem
                    and nrm(tk[i + 1])
                    and nrm(tk[i + 1])[0].startswith(stem[0])
                ):
                    cnt["truncation_followed_by_completed_word(restart)"] += 1
                    if len(exs) < 400:
                        exs.append(dict(file=n, text=t))
        clean = [nrm(w) for w in tk if not re.fullmatch(r"\(\S*\)|-\S+-|//", w)]
        flat = [c[0] for c in clean if c]
        if any(flat[i] == flat[i + 1] for i in range(len(flat) - 1)):
            cnt["sentences_with_adjacent_identical_words(unlabeled, heuristic)"] += 1
res["EPIC_v2_es_originals"] = dict(
    counts=dict(cnt),
    examples=sorted(exs, key=lambda r: h(r["text"]))[:3],
    note="README lines 120-121: only TRUNCATION (=), FILLED (-x-), SPAUSE, MIDWORDPAUSE, MISPRONUNCIATION/CORRECTION. No repetition/revision tag; counts above are heuristics over verbatim text.",
)
(OUT / "other_corpora_rep_rev.json").write_text(
    json.dumps(res, ensure_ascii=False, indent=1), "utf8"
)
print(
    json.dumps(
        {
            k: {
                kk: vv
                for kk, vv in v.items()
                if kk not in ("examples", "examples_REP", "examples_PRE")
            }
            for k, v in res.items()
        },
        ensure_ascii=False,
        indent=1,
    )[:5000]
)
