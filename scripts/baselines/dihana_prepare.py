"""DIHANA: parse user turns -> results/baselines_20261030/dihana/turns.csv (original labels kept, folds from official guides).

Labels (original families, presence per user turn; negative coverage documented by the DIHANA manual):
  f  -> filler_present   (any f tipo, incl. sin_identificar)
  a  -> lengthening_present
  r:repeticion -> repetition_present
  r:sustitucion|insercion|borrado -> reformulation_present
Not mapped (kept only as raw tag counts): b (abandoned), l:cortada, d:*, n:*, p.
Partition (declared BEFORE any inference): folds 1-3 = development, fold 4 = tuning/selection, fold 5 = reserved test.
"""
import csv, json, sys, unicodedata, xml.etree.ElementTree as ET, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from bl_common import CORPORA, OUT, write_config, sha256_file

ROOT = CORPORA / "01_DIHANA" / "extraido" / "dihana"
SPLIT = {1: "dev", 2: "dev", 3: "dev", 4: "tune", 5: "test"}


def strip(s):
    return unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode().lower()


def main():
    fold_of = {}
    for g in sorted((ROOT / "guides").glob("*.gui")):
        k = int(g.stem[-1])
        for s in g.read_text().split():
            fold_of[s.strip()] = k
    rows = []
    for x in sorted((ROOT / "data").glob("*/*/*.xml")):
        rt = ET.parse(x).getroot()
        spk = [h.get("nombre") for h in rt.iter("hablante") if h.get("id") == "usuario"][0]
        for tn in rt.iter("turno"):
            if tn.get("hablante") != "usuario":
                continue
            t = tn.find("t")
            tags = collections.Counter()
            for el in t.iter():
                if el is t: continue
                ty = el.get("tipo")
                tags[f"{el.tag}:{strip(ty)}" if ty else el.tag] += 1
            text = " ".join("".join(t.itertext()).split())
            ul = x.parent / tn.get("fichero_audio_ss")
            rows.append({
                "turn_id": f"{x.stem}_{tn.get('id')}", "speaker": spk, "dialogue": x.stem, "fold": fold_of[spk],
                "split": SPLIT[fold_of[spk]], "ul_rel": str(ul.relative_to(ROOT)).replace("\\", "/"),
                "ul_bytes": ul.stat().st_size if ul.exists() else -1,
                "ref_text_marks_stripped": text,
                "filler_present": int(tags["f:a"] + tags["f:e"] + tags["f:m"] + tags["f:sin_identificar"] > 0),
                "lengthening_present": int(tags["a"] > 0),
                "repetition_present": int(tags["r:repeticion"] > 0),
                "reformulation_present": int(tags["r:sustitucion"] + tags["r:insercion"] + tags["r:borrado"] > 0),
                "n_f": sum(v for k, v in tags.items() if k.startswith("f")), "n_a": tags["a"],
                "n_r_rep": tags["r:repeticion"], "n_r_ref": tags["r:sustitucion"] + tags["r:insercion"] + tags["r:borrado"],
                "raw_tag_counts": json.dumps(dict(tags), ensure_ascii=False, sort_keys=True)})
    d = OUT / "dihana"; d.mkdir(parents=True, exist_ok=True)
    with (d / "turns.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summ = collections.defaultdict(lambda: collections.Counter())
    for r in rows:
        for k in ["filler_present", "lengthening_present", "repetition_present", "reformulation_present"]:
            summ[r["fold"]][k] += r[k]
        summ[r["fold"]]["turns"] += 1
    spk = {k: len({r["speaker"] for r in rows if r["fold"] == k}) for k in range(1, 6)}
    (d / "turns_summary.json").write_text(json.dumps({"per_fold": {k: dict(v) for k, v in sorted(summ.items())}, "speakers_per_fold": spk,
                                                      "split": SPLIT, "n_turns": len(rows)}, indent=1))
    write_config(d / "prepare_run", "dihana_prepare", __file__, {"split": SPLIT, "unit": "user turn"},
                 {"guides": {g.name: sha256_file(g) for g in (ROOT / "guides").glob("*.gui")}, "turns.csv": sha256_file(d / "turns.csv")})
    print(json.dumps({"n": len(rows), "per_fold": {k: dict(v) for k, v in sorted(summ.items())}, "spk": spk}))


main()
