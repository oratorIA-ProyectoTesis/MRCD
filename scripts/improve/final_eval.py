"""Final evaluation of all pre-declared versions on DIHANA fold 5 (test), EPIC-136 (filler), AMERESCO-98 (prolongation).
Everything frozen is READ from frozen_selection.json / frozen_head_<tag>.json; nothing is fitted here.
Writes results/mejoras_20261030/final/{FINAL_metrics.json,FINAL_TABLES.md,predictions_*.csv,config.json}."""

import csv
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import OUT, write_cfg, CLASSES3, CLASSES4  # noqa
from evalkit import *  # noqa
from mrcd import adaptation as A  # noqa

FIN = OUT / "final"
FIN.mkdir(parents=True, exist_ok=True)
CORP = {
    "dihana": (CLASSES4, {"test": "test"}),
    "epic": (["filler_word"], {"dev(all 136)": "all"}),
    "ameresco": (["prolongation"], {"all": "all"}),
}


def load_frozen():
    sel = json.loads((OUT / "frozen_selection.json").read_text())
    heads = {}
    for tag in ("zero", "adopted"):
        p = OUT / f"frozen_head_{tag}.json"
        if p.exists():
            heads[tag] = json.loads(p.read_text())
    return sel, heads


def head_fn(tag_info, rows_by_uid_scores, tau, fallback):
    """Head decision for f/a/r; revision falls back to V0 rule."""
    idx = {c: i for i, c in enumerate(CLASSES3)}
    thr = {c: tag_info["thresholds"][c]["threshold"] for c in CLASSES3}

    def f(row, cls):
        if tau is not None and row["dur_s"] < tau:
            return 0
        if cls not in idx:
            return fallback(row, cls)
        return int(rows_by_uid_scores[row["unit_id"]][:, idx[cls]].max() >= thr[cls])

    return f


def head_scores_for(rows, info):
    ck, _ = A.load_adapted(info["checkpoint"])
    return {
        r["unit_id"]: A.head_scores(
            ck["state"],
            r["rec"]["pooled"],
            len(ck["classes"]),
            ck["head_kind"],
            ck["hidden"],
        )
        for r in rows
    }


def build_versions(corpus, sel, heads):
    """name -> (feature_mode, fn_builder(rows)->fn). External corpora have no padded unit, so 'zero' features serve every Ea variant."""
    ad = sel["ea_adopted"]
    ext = corpus != "dihana"
    amode = (ad["mode"] if ad and not ext else "zero") if ad else None
    atau = ad["tau"] if ad else None
    V = {"V0": ("zero", lambda rows: pred_argmax)}
    if ad:
        V["Va"] = (amode, lambda rows: with_abstention(pred_argmax, atau))

    def rule_fn(key, tau):
        rules = sel["eb"][key]

        def f(row, cls):
            r = rules[cls]
            if tau is not None and row["dur_s"] < tau:
                return 0
            return (
                make_threshold_pred(r["agg"], r["threshold"])(row, cls)
                if r["adopted"]
                else pred_argmax(row, cls)
            )

        return f

    V["Vb"] = ("zero", lambda rows: rule_fn("zero", None))
    if ad and "adopted" in sel["eb"]:
        V["Vab"] = (amode, lambda rows: rule_fn("adopted", atau))
    if "zero" in heads:
        V["Vc"] = (
            "zero",
            lambda rows: head_fn(
                heads["zero"], head_scores_for(rows, heads["zero"]), None, pred_argmax
            ),
        )
    if ad and "adopted" in heads:
        V["Vac"] = (
            amode,
            lambda rows: head_fn(
                heads["adopted"],
                head_scores_for(rows, heads["adopted"]),
                atau,
                with_abstention(pred_argmax, atau),
            ),
        )
    return V


def best_untrained(sel):
    """Untrained comparator per class chosen on DIHANA DEV (not test)."""
    rows = [
        r
        for r in unit_table("dihana", "zero", {"dev"})
        if r["unit_id"]
        in set(json.loads((OUT / "dev_units_frozen.json").read_text())["dev_units"])
    ]
    ch = {}
    for c in CLASSES4:
        y = np.array([r["ref"][c] for r in rows])
        best = None
        for name in ("always_present", "rule_existing_autolabeler", "rule_asr_lexical"):
            p = preds(rows, COMPARATORS[name], c)
            if p is None:
                continue
            f = confusion(y, p)["f1"]
            if best is None or f > best[1]:
                best = (name, f)
        ch[c] = {"comparator": best[0], "dev_f1": best[1]}
    return ch


def residual(rows, fn, cls):
    y = np.array([r["ref"][cls] for r in rows])
    p = np.array([fn(r, cls) for r in rows])

    def grp(mask, label):
        sel = [r for r, m in zip(rows, mask) if m]
        if not sel:
            return {"n": 0}
        tags = lambda r: (
            json.loads(r["ref"].get("raw_tags", "{}") or "{}")
            if "raw_tags" in r["ref"]
            else {}
        )
        d = {
            "n": len(sel),
            "pct_padded_short_turn": float(np.mean([r["padded"] for r in sel])),
            "median_dur_s": float(np.median([r["dur_s"] for r in sel])),
        }
        if rows[0]["ref"].get("raw_tags") is not None:
            d["pct_with_d_rellenar_tag"] = float(
                np.mean([any(k.startswith("d:rellenar") for k in tags(r)) for r in sel])
            )
            d["pct_with_silent_pause_tag_p"] = float(
                np.mean([("p" in tags(r)) for r in sel])
            )
            d["pct_with_other_reference_class"] = float(
                np.mean([any(r["ref"][c] for c in CLASSES4 if c != cls) for r in sel])
            )
        d["pct_asr_has_rule_filler"] = float(
            np.mean([bool(r["asr"]["filler_word"]) for r in sel])
        )
        d["pct_asr_has_rule_repetition"] = float(
            np.mean([bool(r["asr"]["repetition"]) for r in sel])
        )
        return d

    return {
        "FP": grp((y == 0) & (p == 1), "FP"),
        "FN": grp((y == 1) & (p == 0), "FN"),
        "TP": grp((y == 1) & (p == 1), "TP"),
        "TN": grp((y == 0) & (p == 0), "TN"),
    }


def main():
    sel, heads = load_frozen()
    sel_sha = sha256_file(OUT / "frozen_selection.json")
    untrained = best_untrained(sel)
    out = {
        "frozen_selection_sha256": sel_sha,
        "heads": {
            k: {
                "checkpoint_sha256": v["checkpoint_sha256"],
                "thresholds": v["thresholds"],
                "selected": v["selected"],
            }
            for k, v in heads.items()
        },
        "untrained_comparator_chosen_on_dev": untrained,
        "tables": [],
        "paired": [],
        "matrices": {},
        "residual": {},
        "ea_candidates_on_test": [],
    }
    md = [
        "# Final evaluation (frozen selections; fold 5 = reserved test)\n",
        f"frozen_selection.json sha256 `{sel_sha}`",
        "Untrained comparator per class (chosen on DEV): "
        + ", ".join(
            f"{c}: {v['comparator']} (dev F1 {v['dev_f1']:.2f})"
            for c, v in untrained.items()
        ),
        "",
    ]
    adopted = sel["ea_adopted"]
    md.append(
        f"Ea adopted: **{adopted['name'] if adopted else 'none (Va = V0)'}**; Eb adopted classes (zero features): "
        + ", ".join(
            f"{c}:{'thr/' + r['agg'] + f'@{r['threshold']:.2f}' if r['adopted'] else 'V0 rule'}"
            for c, r in sel["eb"]["zero"].items()
        )
        + "\n"
    )
    for corpus, (classes, parts) in CORP.items():
        if not load_features(corpus, "zero"):
            md.append(f"(no features for {corpus}: skipped)"); continue
        V = build_versions(corpus, sel, heads)
        tabs = {}
        for name, (mode, mk) in V.items():
            mode_eff = mode if corpus == "dihana" else "zero"
            rows = [
                r
                for r in unit_table(corpus, mode_eff)
                if (corpus != "dihana" or r["split"] == "test")
            ]
            tabs[name] = (rows, mk(rows))
        base_rows, _ = tabs["V0"]
        assert all(
            [r["unit_id"] for r in tabs[n][0]] == [r["unit_id"] for r in base_rows]
            for n in tabs
        ), "unit sets differ between versions"
        for c in classes:
            md += [
                f"\n#### {corpus} / {c}  (units: {len(base_rows)}, test partition only)\n",
                "| version | partition | n | pos | TP/FP/FN/TN | P | R | F1 |",
                "|---|---|---|---|---|---|---|---|",
            ]
            y = np.array([r["ref"][c] for r in base_rows])
            g = np.array([r["group"] for r in base_rows])
            cmp_name = (
                untrained[c]["comparator"]
                if corpus == "dihana"
                else untrained[c]["comparator"]
            )
            P = {}
            for name, (rows, fn) in tabs.items():
                P[name] = preds(rows, fn, c)
            for name in [
                "always_present",
                "rule_existing_autolabeler",
                "rule_asr_lexical",
            ]:
                p = preds(base_rows, COMPARATORS[name], c)
                if p is not None:
                    P[name] = p
            for name, p in P.items():
                m = confusion(y, p)
                m.update(
                    {
                        "corpus": corpus,
                        "class": c,
                        "version": name,
                        "partition": "test",
                        "n_groups": int(len(set(g))),
                    }
                )
                if m["support_pos"] > 0 and len(set(g)) > 4:
                    m["f1_ci95"] = boot_ci(y, p, g)
                m["comparator"] = name
                m["partition"] = "test"
                out["tables"].append(m)
                md.append(fmt_row(m))
            md += [
                "",
                "| version | dF1 vs V0 [95% CI by speaker] | verdict vs V0 | dF1 vs "
                + cmp_name
                + " [95% CI] | verdict vs untrained | McNemar vs V0 (v right/V0 wrong : V0 right/v wrong, p) |",
                "|---|---|---|---|---|---|",
            ]
            for name in tabs:
                if name == "V0":
                    continue
                row = {"corpus": corpus, "class": c, "version": name}
                if corpus == "ameresco" or len(set(g)) < 5:
                    mc = mcnemar(y, P[name], P["V0"])
                    mc2 = mcnemar(y, P[name], P[cmp_name]) if cmp_name in P else None
                    row.update(
                        {
                            "mcnemar_vs_V0": mc,
                            "mcnemar_vs_untrained": mc2,
                            "dF1_point_vs_V0": confusion(y, P[name])["f1"]
                            - confusion(y, P["V0"])["f1"],
                        }
                    )
                    md.append(
                        f"| {name} | point {row['dF1_point_vs_V0']:+.3f} (no CI: 4 conversations) | McNemar only | "
                        + (
                            f"point {confusion(y, P[name])['f1'] - confusion(y, P[cmp_name])['f1']:+.3f}"
                            if cmp_name in P
                            else "n/a"
                        )
                        + f" | McNemar | {mc['v_right_ref_wrong']}:{mc['ref_right_v_wrong']} p={mc['p_exact']:.3f} |"
                    )
                else:
                    d0 = paired_delta(y, P[name], P["V0"], g)
                    d1 = (
                        paired_delta(y, P[name], P[cmp_name], g)
                        if cmp_name in P
                        else None
                    )
                    mc = mcnemar(y, P[name], P["V0"])
                    row.update(
                        {
                            "vs_V0": d0,
                            "vs_untrained": d1,
                            "untrained": cmp_name,
                            "mcnemar_vs_V0": mc,
                        }
                    )
                    md.append(
                        f"| {name} | {d0['delta']:+.3f} [{d0['ci'][0]:+.3f}, {d0['ci'][1]:+.3f}] | {verdict(d0['ci'])} | "
                        + (
                            f"{d1['delta']:+.3f} [{d1['ci'][0]:+.3f}, {d1['ci'][1]:+.3f}] | {verdict(d1['ci'])}"
                            if d1
                            else "n/a | n/a"
                        )
                        + f" | {mc['v_right_ref_wrong']}:{mc['ref_right_v_wrong']} p={mc['p_exact']:.3f} |"
                    )
                out["paired"].append(row)
        if corpus == "dihana":  # cross-class matrices and residual analysis on fold 5
            md.append(
                "\n#### DIHANA fold 5: reference class present (rows) x predicted class (columns), counts of UNITS (multi-label: a unit can sit in several rows/cols)\n"
            )
            for name, (rows, fn) in tabs.items():
                md += [
                    f"\n`{name}`\n",
                    "| reference \\ predicted | filler_word | prolongation | repetition | revision |",
                    "|---|---|---|---|---|",
                ]
                mat = {}
                for refn, rc in [
                    ("f", "filler_word"),
                    ("a", "prolongation"),
                    ("r", "repetition"),
                    ("rev", "revision"),
                    ("none", None),
                ]:
                    sub = [
                        r
                        for r in rows
                        if (
                            r["ref"][rc]
                            if rc
                            else not any(r["ref"][k] for k in CLASSES4)
                        )
                    ]
                    mat[refn] = {
                        "n": len(sub),
                        **{pc: int(sum(fn(r, pc) for r in sub)) for pc in CLASSES4},
                    }
                    md.append(
                        f"| {refn} (n={len(sub)}) | "
                        + " | ".join(str(mat[refn][pc]) for pc in CLASSES4)
                        + " |"
                    )
                out["matrices"][name] = mat
            for name in [n for n in ("V0", "Vb", "Vc") if n in tabs]:
                out["residual"][name] = {
                    c: residual(tabs[name][0], tabs[name][1], c) for c in CLASSES3
                }
            with (FIN / "predictions_dihana_fold5.csv").open(
                "w", encoding="utf-8-sig", newline=""
            ) as f:
                w = csv.writer(f)
                names = list(tabs)
                w.writerow(
                    ["unit_id", "speaker", "dur_s", "padded"]
                    + [f"y_{c}" for c in CLASSES4]
                    + [f"{n}_{c}" for n in names for c in CLASSES4]
                )
                for i, r in enumerate(tabs["V0"][0]):
                    w.writerow(
                        [
                            r["unit_id"],
                            r["group"],
                            round(r["dur_s"], 3),
                            int(r["padded"]),
                        ]
                        + [r["ref"][c] for c in CLASSES4]
                        + [
                            tabs[n][1](tabs[n][0][i], c)
                            for n in names
                            for c in CLASSES4
                        ]
                    )
            # every declared Ea candidate on test (to report worsenings)
            zr = tabs["V0"][0]
            nr = [r for r in unit_table("dihana", "noise") if r["split"] == "test"]
            for cand in sel["ea_candidates"]:
                rows = zr if cand["mode"] == "zero" else nr
                fn = with_abstention(pred_argmax, cand["tau"])
                gg = np.array([r["group"] for r in rows])
                item = {"name": cand["name"]}
                for c in CLASSES3:
                    yy = np.array([r["ref"][c] for r in rows])
                    pp = preds(rows, fn, c)
                    p0 = preds(zr, pred_argmax, c)
                    d = paired_delta(yy, pp, p0, gg)
                    item[c] = {
                        "f1": confusion(yy, pp)["f1"],
                        "dF1_vs_V0": d["delta"],
                        "ci": d["ci"],
                        "verdict": verdict(d["ci"]),
                    }
                out["ea_candidates_on_test"].append(item)
            md += [
                "\n#### All declared Ea candidates on fold 5 (V0 decision rule), paired vs V0\n",
                "| candidate | f1 filler (dF1 [CI]) | f1 prolongation | f1 repetition |",
                "|---|---|---|---|",
            ]
            for it in out["ea_candidates_on_test"]:
                md.append(
                    f"| {it['name']} | "
                    + " | ".join(
                        f"{it[c]['f1']:.3f} ({it[c]['dF1_vs_V0']:+.3f} [{it[c]['ci'][0]:+.3f},{it[c]['ci'][1]:+.3f}])"
                        for c in CLASSES3
                    )
                    + " |"
                )
    (FIN / "FINAL_metrics.json").write_text(json.dumps(out, indent=1, default=float))
    (FIN / "FINAL_TABLES.md").write_text("\n".join(md), encoding="utf8")
    write_cfg(
        FIN,
        "final_eval",
        __file__,
        {
            "frozen_selection_sha256": sel_sha,
            "heads": {k: v["checkpoint_sha256"] for k, v in heads.items()},
            "n_boot": 2000,
        },
    )
    print("\n".join(md))


if __name__ == "__main__":
    main()
