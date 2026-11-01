"""Second pass over dihana_rep_rev_marks_dev.csv (DEV ONLY, folds 1-3): taxonomy of what a DIHANA 'r' mark is,
ASR failure modes, site/scenario consistency, deterministic example selection. No fold 4/5 data is read."""

import collections
import csv
import hashlib
import json
import re
from pathlib import Path

OUT = Path(r"C:\zArnes-tesis\mrcd\results\auditoria_etiquetas_20261030")
M = list(
    csv.DictReader((OUT / "dihana_rep_rev_marks_dev.csv").open(encoding="utf-8-sig"))
)
T = {}
for r in csv.DictReader(
    (Path(r"C:\zArnes-tesis\mrcd\results\baselines_20261030\dihana\turns.csv")).open(
        encoding="utf-8-sig"
    )
):
    T[r["turn_id"]] = r
dev = json.loads(
    (
        Path(r"C:\zArnes-tesis\mrcd\results\mejoras_20261030\dev_units_frozen.json")
    ).read_text("utf8")
)["dev_units"]
dev_rows = [T[u] for u in dev]
assert {int(r["fold"]) for r in dev_rows} == {1, 2, 3}


def _tk(x):
    import unicodedata
    x = "".join(c for c in unicodedata.normalize("NFD", x.lower()) if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9ñ ]+", " ", x).split()


def asr_loop(r):
    """ASR hallucination loop: >=6 identical consecutive normalised tokens."""
    tk = _tk(r["asr_text"])
    run = 1
    for a, b in zip(tk, tk[1:]):
        run = run + 1 if a == b else 1
        if run >= 6:
            return True
    return False


def inner(r):
    return json.loads(r["inner_marks"])


def tax_rep(r):
    """What the human 'repetición' mark actually is (from reparandum m / correction c structure + inner marks)."""
    inn = {k.split(":")[0] if not k.startswith("l:") else k for k in inner(r)}
    m_has_cut = any(k == "l:cortada" for k in inner(r))
    m_has_mis = any(k == "l:mal_pronunciada" for k in inner(r))
    if r["relation"] == "identical" and not m_has_cut and not m_has_mis:
        return "A_literal_repeat"
    if m_has_mis:
        return "C_mispronunciation_corrected"
    if m_has_cut or r["relation"] == "fragment_then_word":
        return "B_cut_word_restart"
    if r["relation"] == "identical":
        return "A_literal_repeat"
    if len(set(_tk(r["m"] + " " + r["c"]))) == 1:
        return "A_literal_repeat"  # nested / multiple literal repeats of one word (e.g. 'no no no')
    return "D_other_or_expanded"


def tax_rev(r):
    inn = inner(r)
    if inn.get("l:cortada"):
        return "R2_cut_word_then_restart"
    if r["s"]:
        return "R1_explicit_edit_signal"
    if inn.get("r:repetición"):
        return "R3_contains_nested_repetition"
    return "R4_silent_substitution_insertion_deletion"


def fail_mode(r):
    if r["survives"] == "1":
        return "survives"
    cov_c, cov_m, cov_s = float(r["asr_cov_c"]), float(r["asr_cov_m"]), float(r["asr_cov_span"])
    if r["type"] == "repetition":
        if cov_c >= 0.8 and r["asr_repeat_in_region"] == "0":
            return "ASR_collapsed_repeat(corrected form kept, duplicate removed)"
        if cov_c >= 0.8:
            return "ASR_repeat_present_but_span_partial"
        return "ASR_text_mismatch(corrected form <0.8 recovered)"
    if cov_c >= 0.8 and cov_m < 0.8:
        return "ASR_collapsed_reparandum(correction kept, reparandum dropped)"
    return "ASR_text_mismatch(<0.8 of reparandum/correction recovered)"


rep = [r for r in M if r["type"] == "repetition"]
rev = [r for r in M if r["type"] != "repetition"]
for r in rep:
    r["tax"] = tax_rep(r)
for r in rev:
    r["tax"] = tax_rev(r)
for r in M:
    r["fail"] = fail_mode(r)

res = {}
for name, ms in (("repetition", rep), ("revision", rev)):
    d = {"n": len(ms), "taxonomy": dict(collections.Counter(r["tax"] for r in ms))}
    d["taxonomy_x_survival"] = {
        t: [
            sum(1 for r in ms if r["tax"] == t and r["survives"] == "1"),
            sum(1 for r in ms if r["tax"] == t),
        ]
        for t in sorted({r["tax"] for r in ms})
    }
    d["failure_modes"] = dict(collections.Counter(r["fail"] for r in ms))
    d["type_split"] = dict(collections.Counter(r["type"] for r in ms))
    # short vs long reparandum
    d["reparandum_ntok"] = dict(
        collections.Counter(min(int(r["m_ntok"]), 4) for r in ms)
    )
    # case-only difference between m and c (raw), a pure normalisation artefact
    d["case_only_m_vs_c"] = sum(int(r["orth_only_diff"]) for r in ms)
    # does the same turn also carry both classes?
    res[name] = d
rep_turns = {r["unit"] for r in rep}
rev_turns = {r["unit"] for r in rev}
res["turns_with_both_rep_and_rev"] = len(rep_turns & rev_turns)
res["turns_with_rep_only"] = len(rep_turns - rev_turns)
res["turns_with_rev_only"] = len(rev_turns - rep_turns)


# turn-level survival at "turn" granularity, split by taxonomy (a turn is 'A_literal' if it has at least one A mark)
def turn_surv(ms, tax_subset=None):
    by = collections.defaultdict(list)
    for r in ms:
        if tax_subset is None or r["tax"] in tax_subset:
            by[r["unit"]].append(r)
    return len(by), sum(1 for v in by.values() if any(r["survives"] == "1" for r in v))


res["turn_survival"] = {
    "rep_all": turn_surv(rep),
    "rep_A_literal": turn_surv(rep, {"A_literal_repeat"}),
    "rep_B_cutword": turn_surv(rep, {"B_cut_word_restart"}),
    "rep_C_mispron": turn_surv(rep, {"C_mispronunciation_corrected"}),
    "rep_D_other": turn_surv(rep, {"D_other_or_expanded"}),
    "rev_all": turn_surv(rev),
}
# turns with a rep mark whose ONLY rep-class marks are non-literal (A absent)
by = collections.defaultdict(set)
for r in rep:
    by[r["unit"]].add(r["tax"])
res["rep_turns_by_composition"] = {
    "has_A": sum(1 for v in by.values() if "A_literal_repeat" in v),
    "no_A(only B/C/D)": sum(1 for v in by.values() if "A_literal_repeat" not in v),
    "turns": len(by),
}


# site / scenario consistency (dev turns only)
def site(u):
    return u[0]


cnt = collections.defaultdict(
    lambda: [0, 0, 0, 0]
)  # turns, rep_turns, rev_turns, b_turns
for r in dev_rows:
    for key in (
        "site:" + r["speaker"][0],
        "scenario:" + r["dialogue"].split("_")[1][:2],
    ):
        c = cnt[key]
        c[0] += 1
        c[1] += int(r["repetition_present"])
        c[2] += int(r["reformulation_present"])
res["site_scenario_rates"] = {
    k: dict(
        turns=v[0],
        rep=v[1],
        rev=v[2],
        rep_rate=round(v[1] / v[0], 3),
        rev_rate=round(v[2] / v[0], 3),
    )
    for k, v in sorted(cnt.items())
}
# per-dialogue: dialogues with >=1 user turns but zero marks of any r/b
dl = collections.defaultdict(lambda: [0, 0])
for r in dev_rows:
    dl[r["dialogue"]][0] += 1
    dl[r["dialogue"]][1] += int(r["repetition_present"]) + int(
        r["reformulation_present"]
    )
res["dialogues_dev"] = len(dl)
res["dialogues_with_zero_rep_rev"] = sum(1 for v in dl.values() if v[1] == 0)


# deterministic examples: per (name, tax, survives) take 2 lowest sha1(unit+text)
def pick(ms, k=2):
    return sorted(
        ms,
        key=lambda r: hashlib.sha1(
            (r["unit"] + r["human_marked"]).encode()
        ).hexdigest(),
    )[:k]


def strip_marked(s):
    s = re.sub(r"</?(?:t|m|c|s)\b[^>]*>", "", s)
    s = re.sub(r"<r tipo=\"([^\"]+)\">", lambda m: "[R:" + m.group(1)[:4] + " ", s)
    s = s.replace("</r>", "]")
    s = re.sub(r"<p />", "(p)", s)
    s = re.sub(r"<f tipo=\"([^\"]*)\" />", r"(f:\1)", s)
    s = re.sub(r"<a>(.*?)</a>", r"\1:", s)
    s = re.sub(r"<l tipo=\"([^\"]*)\"[^>]*>", r"(l:\1 ", s)
    s = s.replace("</l>", ")")
    s = re.sub(r"<n [^>]*/>", "(n)", s)
    return re.sub(r"\s+", " ", s).strip()


ex = []
for name, ms in (("repetition", rep), ("revision", rev)):
    for tax in sorted({r["tax"] for r in ms}):
        for sv in ("1", "0"):
            sub = [r for r in ms if r["tax"] == tax and r["survives"] == sv]
            for r in pick(sub, 2):
                ex.append(
                    dict(
                        cls=name,
                        tax=tax,
                        survives=sv,
                        unit=r["unit"],
                        fold=r["fold"],
                        human=strip_marked(r["human_marked"])[:300],
                        asr=r["asr_text"][:260],
                        fail=r["fail"],
                    )
                )
res["examples"] = ex
loops = [r for r in M if asr_loop(r)]
res["asr_loop_hallucination"] = {"marks_in_turns_with_loop": len(loops), "turns": len({r["unit"] for r in loops}),
                                  "of_which_counted_as_surviving": sum(1 for r in loops if r["survives"] == "1")}
for name, ms in (("repetition", rep), ("revision", rev)):
    ok = [r for r in ms if not asr_loop(r)]
    res[name]["survival_excluding_loop_turns"] = [sum(1 for r in ok if r["survives"] == "1"), len(ok)]
    res[name]["survival_all"] = [sum(1 for r in ms if r["survives"] == "1"), len(ms)]
res["revision"]["minimal_edit_1tok_to_1tok"] = sum(1 for r in rev if r["m_ntok"] == "1" and r["c_ntok"] == "1")
res["revision"]["with_pause_in_signal"] = sum(int(r["s_pause"]) for r in rev)
res["repetition"]["with_pause_between_copies(s contains <p/>)"] = sum(int(r["s_pause"]) for r in rep)
res["repetition"]["with_filled_pause_or_lengthening_inside"] = sum(1 for r in rep if any(k.startswith("f:") or k == "a" for k in inner(r)))
res["repetition"]["inner_lengthening_a"] = sum(1 for r in rep if "a" in inner(r))
(OUT / "dihana_rep_rev_breakdown_dev.json").write_text(
    json.dumps(res, ensure_ascii=False, indent=1), "utf8"
)
for k, v in res.items():
    if k != "examples":
        print(k, json.dumps(v, ensure_ascii=False)[:900])
with (OUT / "dihana_rep_rev_examples_dev.csv").open(
    "w", newline="", encoding="utf-8-sig"
) as f:
    w = csv.DictWriter(f, fieldnames=list(ex[0]))
    w.writeheader()
    w.writerows(ex)
