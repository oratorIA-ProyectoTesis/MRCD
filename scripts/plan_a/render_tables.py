"""Render the numeric tables of RESULTS_A.md from results_a.json (and g4/g4_summary.json if present), so that no
figure is transcribed by hand. Prints Markdown to stdout.

    python scripts/plan_a/render_tables.py > results/plan_a_20261108/_tables.md
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "results" / "plan_a_20261108"


def f(x, d=3):
    return "n/a" if x is None else f"{x:.{d}f}"


def ci(m):
    lo, hi = m["ci95"]
    return f"[{f(lo)}, {f(hi)}]"


def sys_table(systems: dict) -> str:
    rows = [
        "| system | TP | FP | FN | precision [95 % CI] | recall [95 % CI] | F1 [95 % CI] |",
        "|---|---|---|---|---|---|---|",
    ]
    for k, v in systems.items():
        rows.append(
            f"| {k} | {v['tp']} | {v['fp']} | {v['fn']} | {f(v['precision']['value'])} {ci(v['precision'])} | "
            f"{f(v['recall']['value'])} {ci(v['recall'])} | **{f(v['f1']['value'])}** {ci(v['f1'])} |"
        )
    return "\n".join(rows)


def delta_row(name, d, extra=""):
    return (
        f"| {name} | {f(d['f1']['delta'], 4)} {ci(d['f1'])} | {f(d['f1']['p_boot'], 4)} | "
        f"{f(d['precision']['delta'], 4)} {ci(d['precision'])} | {f(d['recall']['delta'], 4)} {ci(d['recall'])} |{extra}"
    )


def fold_table(rows, key, systems) -> str:
    head = f"| {key} | n | pos | " + " | ".join(f"{s} F1" for s in systems) + " |"
    out = [head, "|" + "---|" * (3 + len(systems))]
    for r in rows:
        out.append(
            f"| {r[key]} | {r['n']} | {r['positives']} | "
            + " | ".join(f(r[s]["f1"]) for s in systems)
            + " |"
        )
    return "\n".join(out)


def main() -> None:
    R = json.loads((OUT / "results_a.json").read_text("utf8"))
    rep = R["repetition"]
    print(
        f"protocol sha256 `{R['protocol_sha256']}`, git `{R['git_head'][:10]}`, {R['label']}\n"
    )
    print(f"### Repetition ({rep['n_turns']} turns, {rep['n_speakers']} speakers)\n")
    print(sys_table(rep["systems"]))
    print(
        "\n| contrast | ΔF1 [95 % CI] | p_boot | Δprecision [95 % CI] | Δrecall [95 % CI] | p_Holm | verdict |"
    )
    print("|---|---|---|---|---|---|---|")
    for k, d in rep["primary_family_holm"].items():
        print(delta_row(k, d, f" {f(d['f1']['p_holm'], 4)} | {d['f1']['verdict']} |"))
    for k, d in rep["descriptive_deltas"].items():
        print(delta_row(k + " (descriptive)", d, " - | - |"))
    dm = rep["demotion"]
    print(
        f"\nDemotion: {dm['n_candidates']} candidates; states {dm['by_state']}; share demoted {f(dm['share_demoted'])}; "
        f"by rule {dm['demoted_by_rule']} (shares {dm['share_demoted_by_rule']}); matched (any order) {dm['matched_by_rule']}."
    )
    print(
        f"Turn flips 1 -> 0: {dm['turn_flips_1_to_0']}; rules present on flipped turns: {dm['turn_flips_by_rule_present']}."
    )
    print(
        f"Raises precision without losing recall: **{dm['raises_precision_without_losing_recall']}** ({dm['rule']}).\n"
    )
    print(
        fold_table(
            rep["per_outer_fold"], "outer", ["V0", "p0_rule", "p1_rule", "o2026_12"]
        )
    )
    print()
    print(
        fold_table(
            rep["per_dihana_fold"],
            "dihana_fold",
            ["V0", "p0_rule", "p1_rule", "o2026_12"],
        )
    )
    for cls, b in R["filler_word_filled_pause_and_prolongation"].items():
        print(f"\n### {cls} (reference: {b['reference']})\n")
        print(sys_table(b["systems"]))
        print(
            "\n| contrast | ΔF1 [95 % CI] | p_boot | Δprecision [95 % CI] | Δrecall [95 % CI] | gate |"
        )
        print("|---|---|---|---|---|---|")
        g = b["gate_vs_o2026_11"]
        print(
            delta_row(
                "o2026_12 - o2026_11 (gate)",
                g,
                f" **{'pass' if g['pass'] else 'FAIL'}** |",
            )
        )
        print(
            delta_row(
                "o2026_12 - p0_restricted (ASR effect)",
                b["p1_minus_p0_restricted"],
                " - |",
            )
        )
        print(
            delta_row(
                "p0_restricted - o2026_11 (search restriction)",
                b["p0_restricted_minus_o2026_11"],
                " - |",
            )
        )
        sysn = list(b["systems"])
        print()
        print(fold_table(b["per_outer_fold"], "outer", sysn))
        sel = b["cv_per_fold_selection"]["p1"]
        print(
            "\nP1 per-fold selection (C, threshold, calibration): "
            + "; ".join(
                f"fold {s['fold']}: C={s['config']['hp']['C']}, thr={f(s['thr'], 2)}, {s['calibration']}"
                for s in sel
            )
        )
    lx = R["lexical_filler"]
    print(f"\n### Lexical filler ({lx['reference']})\n")
    print(sys_table(lx["systems"]))
    print(
        f"\nCandidate states (P1): {lx['candidate_states']}. By item: {lx['candidates_by_lexicon_item']}.\n"
    )
    print("Union against the f-mark reference (descriptive):\n")
    print(
        sys_table(
            {
                k: v
                for k, v in lx["union_filler_word_vs_filler_present"].items()
                if k != "note"
            }
        )
    )
    g4p = OUT / "g4" / "g4_summary.json"
    if g4p.exists():
        G = json.loads(g4p.read_text("utf8"))
        print("\n### Latency (G4 procedure)\n")
        print(
            "| profile / mode | median s | measured rounds s | equal to offline | background CPU % |"
        )
        print("|---|---|---|---|---|")
        for k, v in G.items():
            if "|" in k:
                print(
                    f"| {k} | {v['latency_median_s']} | {v['latency_measured_s']} | {v['all_equal_to_offline']} | {v['background_cpu_pct_before']} |"
                )
        print(f"\nG4: {G['G4']}")


if __name__ == "__main__":
    main()
