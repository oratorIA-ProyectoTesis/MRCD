"""Build the paired ASR evaluation units with LEGITIMATE HUMAN REFERENCES (Task 3.2).

AMERESCO: clip wavs of results/baselines_20261030/ameresco (+-5 s context around a 3-8 s annotated segment). The reference
is the human ELAN (Val.Es.Co.) transcript of EVERY tier whose annotation midpoint lies inside the clip, in time order.
Clips are chosen greedily so that no two clips of a recording share audio (the 98-clip pool overlaps itself).
DIHANA folds 1-4 (never fold 5): user turns >= 10 s, reference = mrcd.corpora.dihana.dihana_input_view (turn text,
no response marks), 15 turns per fold, deterministic hash order. Group = speaker (DIHANA) / recording (AMERESCO).
Reference cleaning (frozen, documented in the report): see ``clean_ameresco``.
"""
import os
import csv, hashlib, json, re, sys, unicodedata, xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import soundfile as sf  # noqa: E402
from mrcd.corpora.dihana import dihana_input_view  # noqa: E402

B = REPO / "results/baselines_20261030"
CORP = Path(os.environ.get("MRCD_CORPORA_ROOT", r"<CORPORA_ROOT>"))
OUT = REPO / "results/cierre_oe1_oe3_20261031/3b/asr"
SEED, PER_FOLD, MIN_DIHANA_S = 20261031, 15, 10.0
SKIP_TIERS = {"Observaciones", "Anonimización", "Anonimizacion"}


def clean_ameresco(v: str) -> tuple[str, dict]:
    """Val.Es.Co. markup -> plain words. Returns (text, counters). Rules (frozen):
    <ininteligible/>, <risas/>, <obs .../>, <alargamiento/> and other self-closing tags are removed (ininteligible counted);
    <tag ...>inner</tag> keeps ``inner`` (standard orthography for fsr/extranjero/cita/anonimo...); [ ] and (( )) brackets are dropped
    keeping their content (overlap / uncertain, counted); fragments ending in '-' (cut words) are dropped (counted)."""
    c = {"ininteligible": len(re.findall(r"<ininteligible\s*/>", v)), "uncertain": v.count("(("), "overlap": v.count("["), "cut_words": 0}
    v = re.sub(r"<[^>]*/>", " ", v)
    v = re.sub(r"</?[^>]+>", " ", v)
    v = v.replace("[", " ").replace("]", " ").replace("((", " ").replace("))", " ")
    toks = []
    for t in v.split():
        if t.endswith("-") and len(t) > 1:
            c["cut_words"] += 1
            continue
        toks.append(t)
    return " ".join(toks), c


def ameresco_units():
    man = json.loads((B / "ameresco/manifest.json").read_text("utf8"))
    chosen, last_end = [], {}
    for m in sorted(man, key=lambda m: (m["group"], m["source_start_ms"])):
        cs, ce = m["source_start_ms"] - m["target_start_s"] * 1000, None
        info = sf.info(m["wav"]); ce = cs + info.duration * 1000
        if cs < last_end.get(m["group"], -1):
            continue
        last_end[m["group"]] = ce
        chosen.append((m, cs, ce, info.duration))
    eaf_cache, units = {}, []
    for m, cs, ce, dur in chosen:
        rid = m["group"]
        if rid not in eaf_cache:
            t = ET.parse(CORP / "12_AMERESCO" / rid / f"{rid}.eaf")
            ts = {x.get("TIME_SLOT_ID"): int(x.get("TIME_VALUE")) for x in t.iter("TIME_SLOT") if x.get("TIME_VALUE")}
            ann = []
            for tier in t.iter("TIER"):
                if tier.get("TIER_ID") in SKIP_TIERS:
                    continue
                for a in tier.iter("ALIGNABLE_ANNOTATION"):
                    ann.append((ts[a.get("TIME_SLOT_REF1")], ts[a.get("TIME_SLOT_REF2")], tier.get("TIER_ID"), a.findtext("ANNOTATION_VALUE") or ""))
            eaf_cache[rid] = sorted(ann)
        parts, cnt, tiers = [], {"ininteligible": 0, "uncertain": 0, "overlap": 0, "cut_words": 0}, set()
        for s, e, tier, val in eaf_cache[rid]:
            if cs <= (s + e) / 2 <= ce and val.strip():
                txt, c = clean_ameresco(val)
                parts.append(txt); tiers.add(tier)
                for k in cnt: cnt[k] += c[k]
        ref = " ".join(p for p in parts if p)
        units.append({"unit_id": m["unit_id"], "corpus": "ameresco", "group": rid, "wav": m["wav"], "wav_sha256": m["wav_sha256"],
                      "duration_s": dur, "ref_raw": ref, "ref_markup_counts": cnt, "ref_tiers": sorted(tiers),
                      "subset": "clean" if cnt["ininteligible"] + cnt["uncertain"] + cnt["overlap"] == 0 and len(tiers) == 1 else "noisy_ref"})
    return units


def dihana_units():
    T = list(csv.DictReader((B / "dihana/turns.csv").open(encoding="utf-8-sig")))
    wavs = {m["unit_id"]: m for m in json.loads((B / "dihana/manifest.json").read_text("utf8"))}
    texts = {}
    for x in sorted((Path(CORP) / "01_DIHANA/extraido/dihana/data").glob("*/*/*.xml")):
        for tn in ET.parse(x).getroot().iter("turno"):
            if tn.get("hablante") == "usuario":
                texts[f"{x.stem}_{tn.get('id')}"] = dihana_input_view(tn.find("t"))
    units = []
    for fold in (1, 2, 3, 4):
        cand = []
        for t in T:
            if int(t["fold"]) != fold or t["turn_id"] not in wavs:
                continue
            w = wavs[t["turn_id"]]; d = sf.info(w["wav"]).duration
            if d >= MIN_DIHANA_S:
                cand.append((hashlib.sha1(f"{SEED}:{t['turn_id']}".encode()).hexdigest(), t, w, d))
        for _, t, w, d in sorted(cand, key=lambda x: x[0])[:PER_FOLD]:
            assert texts[t["turn_id"]] == t["ref_text_marks_stripped"], t["turn_id"]
            units.append({"unit_id": t["turn_id"], "corpus": "dihana", "fold": fold, "group": t["speaker"], "wav": w["wav"],
                          "duration_s": d, "ref_raw": texts[t["turn_id"]], "subset": f"fold{fold}"})
    return units


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    am, di = ameresco_units(), dihana_units()
    for u in am + di:
        h = hashlib.sha256(Path(u["wav"]).read_bytes()).hexdigest(); u["wav_sha256"] = h
    units = am + di
    (OUT / "units.json").write_text(json.dumps(units, indent=1, ensure_ascii=False), "utf8")
    print(len(am), "ameresco", len(di), "dihana", round(sum(u["duration_s"] for u in units) / 60, 1), "min audio")
    print("ameresco groups", sorted({u["group"] for u in am}), "clean", sum(u["subset"] == "clean" for u in am))
    print("dihana speakers", len({u["group"] for u in di}))


if __name__ == "__main__":
    main()
