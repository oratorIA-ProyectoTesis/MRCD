"""WER vs HUMAN references, paired comparison, group bootstrap, disagreement and error decomposition (Task 3.2).

Frozen normalization (``norm_text``): NFKC, lowercase, '_' -> space, every character that is not a letter/digit/space removed
(punctuation, symbols; apostrophes join: "pa'" -> "pa"), whitespace collapsed. Accents are KEPT (primary); the accent-folded WER is a
sensitivity column. Disfluency words are NOT removed from the reference or the hypothesis, EXCEPT in the declared secondary
variant ``nofill`` where the hypothesis loses the pure vocalic fillers {eh, ehh, em, ehm, mm, mmm, hm, ah, uh, aj} because DIHANA
transcripts mark fillers as empty ``<f/>`` elements (the filler word is not in the reference text).
usage: python asr_analyze.py --pred DIR --units units.json --out DIR
"""

import argparse
import json
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

FILLERS = {"eh", "ehh", "em", "ehm", "mm", "mmm", "hm", "ah", "uh", "aj", "eeh", "mhm"}
SEED, NBOOT = 20261031, 2000


def norm_text(s: str, fold_accents: bool = False) -> list[str]:
    s = unicodedata.normalize("NFKC", s).lower().replace("_", " ")
    if fold_accents:
        s = "".join(
            c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)
        )
    s = "".join(
        c if (c.isalnum() or c.isspace()) else ("" if c in "'’`" else " ") for c in s
    )
    return s.split()


def align(ref, hyp):
    """Levenshtein alignment -> (S, D, I, ops) with ops = list of (op, i, j)."""
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), np.int32)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        ri = ref[i - 1]
        for j in range(1, m + 1):
            d[i, j] = min(
                d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (ri != hyp[j - 1])
            )
    i, j, ops = n, m, []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append(("C" if ref[i - 1] == hyp[j - 1] else "S", i - 1, j - 1))
            i -= 1
            j -= 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            ops.append(("D", i - 1, None))
            i -= 1
        else:
            ops.append(("I", None, j - 1))
            j -= 1
    ops.reverse()
    s = sum(o[0] == "S" for o in ops)
    dl = sum(o[0] == "D" for o in ops)
    ins = sum(o[0] == "I" for o in ops)
    return s, dl, ins, ops


def adj_dups(toks):
    return sum(1 for a, b in zip(toks, toks[1:]) if a == b)


def pooled(rows, key="err", nkey="n_ref"):
    n = sum(r[nkey] for r in rows)
    return sum(r[key] for r in rows) / n if n else None


def boot_ci(rows, fn, groups, seed=SEED, nboot=NBOOT):
    """Percentile 95% CI resampling GROUPS (speakers / recordings) with replacement."""
    by = defaultdict(list)
    for r in rows:
        by[r["group"]].append(r)
    keys = sorted(by)
    if len(keys) < 2:
        return None
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(nboot):
        pick = rng.integers(0, len(keys), len(keys))
        sample = [r for k in pick for r in by[keys[k]]]
        v = fn(sample)
        if v is not None:
            vals.append(v)
    return (
        [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
        if vals
        else None
    )


def load_pred(pdir, cond):
    d = Path(pdir) / cond
    return (
        {p.stem: json.loads(p.read_text("utf8")) for p in d.glob("*.json")}
        if d.exists()
        else {}
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--units", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conds", nargs="*", default=None)
    a = ap.parse_args()
    units = {u["unit_id"]: u for u in json.loads(Path(a.units).read_text("utf8"))}
    conds = a.conds or sorted(
        p.name
        for p in Path(a.pred).iterdir()
        if p.is_dir() and not p.name.startswith("_")
    )
    P = {c: load_pred(a.pred, c) for c in conds}
    # units evaluated per condition = intersection with the reference set; paired comparisons use common units
    refs = {uid: norm_text(u["ref_raw"]) for uid, u in units.items()}
    refs_f = {uid: norm_text(u["ref_raw"], True) for uid, u in units.items()}
    hyp = {
        c: {
            uid: norm_text(" ".join(w["text"] for w in r["words"]))
            for uid, r in P[c].items()
        }
        for c in conds
    }
    hyp_f = {
        c: {
            uid: norm_text(" ".join(w["text"] for w in r["words"]), True)
            for uid, r in P[c].items()
        }
        for c in conds
    }
    result = {
        "normalization": "NFKC, lowercase, '_'->space, drop non-alphanumeric, accents kept (accent-folded = sensitivity)",
        "conditions": {},
        "paired": {},
        "disagreement": {},
        "decomposition": {},
    }

    def rows_for(c, uids, variant="primary"):
        out = []
        for uid in uids:
            u = units[uid]
            ref = refs_f[uid] if variant == "accent_folded" else refs[uid]
            h = hyp_f[c][uid] if variant == "accent_folded" else hyp[c][uid]
            if variant == "nofill":
                h = [t for t in h if t not in FILLERS]
            if not ref:
                continue
            s, dl, ins, _ = align(ref, h)
            out.append(
                {
                    "unit": uid,
                    "group": u["group"],
                    "corpus": u["corpus"],
                    "subset": u["subset"],
                    "n_ref": len(ref),
                    "S": s,
                    "D": dl,
                    "I": ins,
                    "err": s + dl + ins,
                    "n_hyp": len(h),
                    "dups": adj_dups(h),
                    "dups_ref": adj_dups(ref),
                    "digits": sum(t.isdigit() for t in h),
                }
            )
        return out

    subsets = {
        "all": lambda u: True,
        "ameresco": lambda u: u["corpus"] == "ameresco",
        "ameresco_clean": lambda u: (
            u["corpus"] == "ameresco" and u["subset"] == "clean"
        ),
        "dihana_f1-4": lambda u: u["corpus"] == "dihana",
    }
    for f in (1, 2, 3, 4):
        subsets[f"dihana_fold{f}"] = (
            lambda f: lambda u: u["corpus"] == "dihana" and u.get("fold") == f
        )(f)
    for c in conds:
        result["conditions"][c] = {
            "n_units": len(P[c]),
            "timing": {
                "wall_s_total": sum(r["wall_s"] for r in P[c].values()),
                "audio_s_total": sum(r["duration_s"] for r in P[c].values()),
                "asr_calls_total": sum(r["asr_calls"] for r in P[c].values()),
            },
        }
        t = result["conditions"][c]["timing"]
        t["rtf"] = (
            t["wall_s_total"] / t["audio_s_total"] if t["audio_s_total"] else None
        )
        for variant in ("primary", "accent_folded", "nofill"):
            for sname, pred in subsets.items():
                uids = [uid for uid in P[c] if uid in units and pred(units[uid])]
                rows = rows_for(c, uids, variant)
                if not rows:
                    continue
                ng = len({r["group"] for r in rows})
                wer = pooled(rows)
                result["conditions"][c].setdefault(variant, {})[sname] = {
                    "n_units": len(rows),
                    "n_groups": ng,
                    "n_ref_words": sum(r["n_ref"] for r in rows),
                    "wer": wer,
                    "ci95_group_bootstrap": boot_ci(rows, pooled, ng),
                    "S": sum(r["S"] for r in rows),
                    "D": sum(r["D"] for r in rows),
                    "I": sum(r["I"] for r in rows),
                    "adjacent_dup_tokens_hyp": sum(r["dups"] for r in rows),
                    "adjacent_dup_tokens_ref": sum(r["dups_ref"] for r in rows),
                    "digit_tokens_hyp": sum(r["digits"] for r in rows),
                    "ci_note": "few groups: interval not informative"
                    if ng < 10
                    else "",
                }
    # paired differences (same units only)
    pairs = [
        ("str_p0_t2", "off_p0"),
        ("str_p1_t2", "off_p1"),
        ("off_p1", "off_p0"),
        ("str_p1_t2", "str_p0_t2"),
        ("str_p0_t0.5", "off_p0"),
        ("str_p1_t0.5", "off_p1"),
        ("str_p0_t0.5", "str_p0_t2"),
        ("str_p0_t2_ca", "str_p0_t2"),
        ("str_p0_t2_ca", "off_p0"),
        ("str_p1_t2_ca", "str_p1_t2"),
        ("str_p1_t2_ca", "off_p1"),
    ]
    for x, y in pairs:
        if x not in P or y not in P:
            continue
        common = [uid for uid in P[x] if uid in P[y] and uid in units]
        for sname, pred in subsets.items():
            uids = [uid for uid in common if pred(units[uid])]
            rx, ry = (
                {r["unit"]: r for r in rows_for(x, uids)},
                {r["unit"]: r for r in rows_for(y, uids)},
            )
            keep = [uid for uid in uids if uid in rx and uid in ry]
            if not keep:
                continue
            rows = [
                {
                    "group": units[uid]["group"],
                    "n_ref": rx[uid]["n_ref"],
                    "d": rx[uid]["err"] - ry[uid]["err"],
                }
                for uid in keep
            ]
            f = lambda rr: pooled(rr, "d")  # noqa: E731
            ng = len({r["group"] for r in rows})
            result["paired"].setdefault(f"{x}_minus_{y}", {})[sname] = {
                "n_units": len(rows),
                "n_groups": ng,
                "delta_wer": f(rows),
                "ci95_group_bootstrap": boot_ci(rows, f, ng),
                "units_worse": sum(r["d"] > 0 for r in rows),
                "units_better": sum(r["d"] < 0 for r in rows),
                "units_equal": sum(r["d"] == 0 for r in rows),
                "ci_note": "few groups: interval not informative" if ng < 10 else "",
            }
    # disagreement streaming vs offline (NOT accuracy)
    for x, y in (
        ("str_p0_t2", "off_p0"),
        ("str_p1_t2", "off_p1"),
        ("str_p0_t0.5", "off_p0"),
        ("str_p1_t0.5", "off_p1"),
        ("str_p0_t2_ca", "off_p0"),
    ):
        if x not in P or y not in P:
            continue
        rows = []
        for uid in P[x]:
            if uid in P[y] and hyp[y][uid]:
                s, dl, ins, _ = align(hyp[y][uid], hyp[x][uid])
                rows.append(
                    {
                        "group": units[uid]["group"],
                        "n_ref": len(hyp[y][uid]),
                        "err": s + dl + ins,
                    }
                )
        if rows:
            result["disagreement"][f"{x}_vs_{y}"] = {
                "label": "WORD DISAGREEMENT between two Whisper runs (one used as reference); NOT accuracy",
                "n_units": len(rows),
                "pooled_disagreement": pooled(rows),
                "ci95_group_bootstrap": boot_ci(
                    rows, pooled, len({r["group"] for r in rows})
                ),
                "per_unit_median": float(
                    np.median([r["err"] / r["n_ref"] for r in rows])
                ),
            }
    # decomposition: where streaming differs; timestamps; seam locality
    from mrcd.asr_profiles import detect_asr_anomalies, get_profile

    for x, y, prof in (
        ("str_p0_t2", "off_p0", "p0"),
        ("str_p1_t2", "off_p1", "p1_disfluent_prompt"),
        ("str_p0_t0.5", "off_p0", "p0"),
        ("str_p1_t0.5", "off_p1", "p1_disfluent_prompt"),
        ("str_p0_t2_ca", "off_p0", "p0"),
    ):
        if x not in P or y not in P:
            continue
        tick = float(re.search(r"_t([0-9.]+)", x).group(1))
        dt, near_left, near_total, total_dis, covered, timeline = [], 0, 0, 0, 0.0, 0.0
        n_anom_x = n_anom_y = 0
        wrongs = defaultdict(int)
        for uid in P[x]:
            if uid not in P[y] or uid not in units:
                continue
            wx, wy = P[x][uid]["words"], P[y][uid]["words"]
            nx, ny = (
                [norm_text(w["text"]) for w in wx],
                [norm_text(w["text"]) for w in wy],
            )
            fx = [(t, i) for i, ts in enumerate(nx) for t in ts]
            fy = [(t, i) for i, ts in enumerate(ny) for t in ts]
            s, dl, ins, ops = align([t for t, _ in fy], [t for t, _ in fx])
            dur = P[x][uid]["duration_s"]
            ticks = list(np.arange(6.0, dur, tick)) + [dur]
            zones = [
                (max(0.0, T - 13.0), max(0.0, T - 13.0) + 1.5)
                for T in ticks
                if T - 13.0 > 0
            ]
            for op, i, j in ops:
                if op == "C":
                    dt.append(wx[fx[j][1]]["start"] - wy[fy[i][1]]["start"])
                else:
                    total_dis += 1
                    tt = (
                        wx[fx[j][1]]["start"]
                        if j is not None
                        else wy[fy[i][1]]["start"]
                    )
                    if any(a0 <= tt <= a1 for a0, a1 in zones):
                        near_left += 1
                    wrongs[op] += 1
            if zones:
                lo = zones[0][0]
                timeline += max(dur - lo, 0)
                merged = []
                for a0, a1 in zones:
                    if merged and a0 <= merged[-1][1]:
                        merged[-1] = (merged[-1][0], max(merged[-1][1], a1))
                    else:
                        merged.append((a0, a1))
                covered += sum(min(a1, dur) - a0 for a0, a1 in merged)
            n_anom_x += bool(
                detect_asr_anomalies(
                    " ".join(w["text"] for w in wx), get_profile(prof).initial_prompt
                )
            )
            n_anom_y += bool(
                detect_asr_anomalies(
                    " ".join(w["text"] for w in wy), get_profile(prof).initial_prompt
                )
            )
        dt = np.array(dt) if dt else np.array([0.0])
        result["decomposition"][f"{x}_vs_{y}"] = {
            "matched_words": len(dt) if dt.any() else 0,
            "start_delta_s_median": float(np.median(dt)),
            "start_delta_abs_p90": float(np.percentile(np.abs(dt), 90)),
            "share_abs_delta_gt_0.12s": float(np.mean(np.abs(dt) > 0.12)),
            "disagreement_tokens": total_dis,
            "ops": dict(wrongs),
            "disagreement_in_left_edge_zones_1.5s": near_left,
            "share_disagreement_in_left_edge_zones": near_left / total_dis
            if total_dis
            else None,
            "share_of_timeline_in_left_edge_zones": covered / timeline
            if timeline
            else None,
            "units_with_asr_anomaly_stream": n_anom_x,
            "units_with_asr_anomaly_offline": n_anom_y,
        }
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "asr_analysis.json").write_text(
        json.dumps(result, indent=1, ensure_ascii=False), "utf8"
    )
    # brief console table
    for c in conds:
        pr = result["conditions"][c].get("primary", {}).get("all")
        if pr:
            print(
                c,
                "n",
                pr["n_units"],
                "WER",
                round(pr["wer"], 4),
                "CI",
                pr["ci95_group_bootstrap"],
                "rtf",
                round(result["conditions"][c]["timing"]["rtf"], 2),
            )


if __name__ == "__main__":
    main()
