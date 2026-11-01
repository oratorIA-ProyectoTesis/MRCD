"""Selection of Ea (short-turn handling) and Eb (threshold rule) using ONLY DIHANA dev (folds 1-3) and tune (fold 4).
This script never loads fold 5, EPIC or AMERESCO labels. Criteria: PREREGISTRATION.md.
Writes: dev_units_frozen.json, frozen_selection.json, selection_dev_tune.md, config.json (in results/mejoras_20261030/selection/)."""

import json
import hashlib
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import OUT, REPO, write_cfg  # noqa
from evalkit import *  # noqa
from mrcd import decision as D  # noqa

SEL = OUT / "selection"
SEL.mkdir(parents=True, exist_ok=True)
CL3, CL4 = (
    ["filler_word", "prolongation", "repetition"],
    ["filler_word", "prolongation", "repetition", "revision"],
)
TAUS = [None, 0.5, 1.0, 1.5]


def table_for(mode, ids):
    return [
        r
        for r in unit_table("dihana", mode, splits={"dev", "tune"})
        if r["unit_id"] in ids
    ]


def arrays(rows, fn, classes):
    return {c: preds(rows, fn, c) for c in classes}, {
        c: np.array([r["ref"][c] for r in rows]) for c in classes
    }


def main():
    z_all, n_all = (
        unit_table("dihana", "zero", {"dev", "tune"}),
        unit_table("dihana", "noise", {"dev", "tune"}),
    )
    ids = {r["unit_id"] for r in z_all} & {r["unit_id"] for r in n_all}
    dev_ids = sorted(
        u for u in ids if int(references("dihana")[u]["fold"]) in (1, 2, 3)
    )
    tune_ids = sorted(u for u in ids if int(references("dihana")[u]["fold"]) == 4)
    assert len(tune_ids) == 1240, len(tune_ids)
    frozen = {"dev_units": dev_ids, "tune_units": tune_ids}
    sha = hashlib.sha256(json.dumps(frozen).encode()).hexdigest()
    (OUT / "dev_units_frozen.json").write_text(
        json.dumps({**frozen, "sha256_of_json_dumps": sha})
    )
    R = {
        "zero": {
            s: [
                r
                for r in table_for("zero", set(ids))
                if r["unit_id"] in (set(dev_ids) if s == "dev" else set(tune_ids))
            ]
            for s in ("dev", "tune")
        },
        "noise": {
            s: [
                r
                for r in table_for("noise", set(ids))
                if r["unit_id"] in (set(dev_ids) if s == "dev" else set(tune_ids))
            ]
            for s in ("dev", "tune")
        },
    }
    for s in ("dev", "tune"):
        assert [r["unit_id"] for r in R["zero"][s]] == [
            r["unit_id"] for r in R["noise"][s]
        ]
    lines = [
        f"# Selection on DIHANA dev ({len(dev_ids)} turns, folds 1-3) and tune ({len(tune_ids)} turns, fold 4). Fold 5 / EPIC / AMERESCO not loaded.\n"
    ]

    # ------------------------------------------------ Ea
    base = {s: arrays(R["zero"][s], pred_argmax, CL3) for s in ("dev", "tune")}
    cands = []
    for mode in ("zero", "noise"):
        for tau in TAUS:
            if mode == "zero" and tau is None:
                continue
            out = {"mode": mode, "tau": tau, "name": f"A_{mode}_tau{tau}"}
            for s in ("dev", "tune"):
                P, Y = arrays(R[mode][s], with_abstention(pred_argmax, tau), CL3)
                g = [r["group"] for r in R[mode][s]]
                pd = paired_macro_delta(Y, P, base[s][0], g, CL3)
                out[s] = {
                    "macro_f1": float(
                        np.mean([confusion(Y[c], P[c])["f1"] for c in CL3])
                    ),
                    "delta_vs_V0": pd["delta"],
                    "ci": pd["ci"],
                    "per_class_f1": {c: confusion(Y[c], P[c])["f1"] for c in CL3},
                }
            cands.append(out)
    v0 = {
        s: {
            "macro_f1": float(
                np.mean([confusion(base[s][1][c], base[s][0][c])["f1"] for c in CL3])
            ),
            "per_class_f1": {
                c: confusion(base[s][1][c], base[s][0][c])["f1"] for c in CL3
            },
        }
        for s in ("dev", "tune")
    }
    ok = [c for c in cands if c["dev"]["ci"][0] > 0 and c["tune"]["delta_vs_V0"] >= 0]
    adopted = max(ok, key=lambda c: c["dev"]["macro_f1"]) if ok else None
    lines += [
        "## Ea short-turn handling (V0 decision rule held fixed)\n",
        f"V0 macro-F1(f,a,r): dev {v0['dev']['macro_f1']:.3f}, tune {v0['tune']['macro_f1']:.3f}\n",
        "| candidate | dev macroF1 | dev dMacroF1 [95% CI] | tune macroF1 | tune dMacroF1 | passes criterion |",
        "|---|---|---|---|---|---|",
    ]
    for c in cands:
        lines.append(
            f"| {c['name']} | {c['dev']['macro_f1']:.3f} | {c['dev']['delta_vs_V0']:+.3f} [{c['dev']['ci'][0]:+.3f}, {c['dev']['ci'][1]:+.3f}] | {c['tune']['macro_f1']:.3f} | {c['tune']['delta_vs_V0']:+.3f} | {c in ok} |"
        )
    lines.append(f"\nAdopted: **{adopted['name'] if adopted else 'none (Va = V0)'}**\n")

    # ------------------------------------------------ Eb
    def fit_rule(mode, tau, tag):
        rules = {}
        rows_dev, rows_tune = R[mode]["dev"], R[mode]["tune"]
        keep_dev = [r for r in rows_dev if not (tau is not None and r["dur_s"] < tau)]
        gdev = [r["group"] for r in keep_dev]
        for c in CL4:
            y_dev = np.array([r["ref"][c] for r in keep_dev])
            argmax_fn = with_abstention(pred_argmax, tau)
            base_dev = confusion(
                [r["ref"][c] for r in rows_dev], preds(rows_dev, argmax_fn, c)
            )["f1"]
            base_tune = confusion(
                [r["ref"][c] for r in rows_tune], preds(rows_tune, argmax_fn, c)
            )["f1"]
            best = None
            per_agg = {}
            for agg in D.AGGREGATIONS:
                sc = np.array(
                    [D.unit_score(r["rec"]["probs"], LABELS, c, agg) for r in keep_dev]
                )
                cv = D.cv_threshold(sc, y_dev, gdev)
                # OOF F1 over ALL dev units (abstained units count as predicted absent)
                per_agg[agg] = {
                    "threshold": cv["threshold"],
                    "oof_f1_nonabstained": cv["oof_f1"],
                    "in_sample_f1": cv["dev_f1_in_sample"],
                    "fold_thresholds": cv["fold_thresholds"],
                }
                if best is None or cv["oof_f1"] > per_agg[best]["oof_f1_nonabstained"]:
                    best = agg
            thr = per_agg[best]["threshold"]
            fn = with_abstention(make_threshold_pred(best, thr), tau)
            tune_f1 = confusion(
                [r["ref"][c] for r in rows_tune], preds(rows_tune, fn, c)
            )["f1"]
            dev_oof_f1_all = None
            # honest dev comparison on the same (non-abstained) units: V0 rule vs OOF thresholds
            y_k = y_dev
            p_v0 = np.array([pred_argmax(r, c) for r in keep_dev])
            base_dev_kept = confusion(y_k, p_v0)["f1"]
            adopt = (per_agg[best]["oof_f1_nonabstained"] > base_dev_kept) and (
                tune_f1 >= base_tune
            )
            rules[c] = {
                "adopted": bool(adopt),
                "agg": best,
                "threshold": thr,
                "per_agg": per_agg,
                "dev_rule_f1_nonabstained_units": base_dev_kept,
                "dev_oof_f1_threshold_rule": per_agg[best]["oof_f1_nonabstained"],
                "tune_f1_threshold_rule": tune_f1,
                "tune_f1_argmax_rule": base_tune,
                "dev_support_pos": int(y_dev.sum()),
            }
        return rules

    eb = {"zero": fit_rule("zero", None, "zero")}
    if adopted:
        eb["adopted"] = fit_rule(adopted["mode"], adopted["tau"], "adopted")
    for key, rules in eb.items():
        lines += [
            f"\n## Eb threshold rule, features = {'V0 zero-pad' if key == 'zero' else adopted['name']}\n",
            "| class | dev pos | agg | thr | dev F1 V0 rule | dev OOF F1 thr rule | tune F1 V0 rule | tune F1 thr rule | adopted |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for c, r in rules.items():
            lines.append(
                f"| {c} | {r['dev_support_pos']} | {r['agg']} | {r['threshold']:.2f} | {r['dev_rule_f1_nonabstained_units']:.3f} | {r['dev_oof_f1_threshold_rule']:.3f} | {r['tune_f1_argmax_rule']:.3f} | {r['tune_f1_threshold_rule']:.3f} | {r['adopted']} |"
            )
    frozen_sel = {
        "dev_units_sha256": sha,
        "n_dev": len(dev_ids),
        "n_tune": len(tune_ids),
        "ea_candidates": cands,
        "ea_adopted": adopted,
        "v0": v0,
        "eb": eb,
    }
    p = OUT / "frozen_selection.json"
    p.write_text(json.dumps(frozen_sel, indent=1, default=float))
    (OUT / "selection_dev_tune.md").write_text("\n".join(lines), encoding="utf8")
    write_cfg(
        SEL,
        "select_dev",
        __file__,
        {
            "frozen_selection_sha256": sha256_file(p),
            "dev_units_sha256": sha,
            "uses_folds": [1, 2, 3, 4],
        },
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
