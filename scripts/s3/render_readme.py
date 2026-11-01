"""Render README_S3.md from the result JSONs (numbers are never typed by hand)."""

import json
from pathlib import Path

S3 = Path(__file__).resolve().parents[2] / "results" / "mejora_objetivos_20261101" / "S3"
L = lambda p: json.loads((S3 / p).read_text("utf8"))  # noqa: E731
e1, e2, e3, e4 = (
    L("e1/results.json"),
    L("e2/results.json"),
    L("e3/results.json"),
    L("e4/results.json"),
)
proto = L("PROTOCOL.sha256.json")


def ci(c):
    return f"[{c[0]:.3f}, {c[1]:.3f}]"


def row_w(name, m, spk, posspk):
    return (
        f"| {name} | {m['support_pos']}/{m['n']} | {m['tp']}/{m['fp']}/{m['fn']} | {m['w_precision']:.3f} | "
        f"{m['w_recall']:.3f} | {m['w_f1']:.3f} {ci(m['f1w_ci95'])} | {spk}/{posspk} |"
    )


def row_u(name, m):
    return (
        f"| {name} | {m['tp'] + m['fn']}/{m['n']} | {m['tp']}/{m['fp']}/{m['fn']} | {m['precision']:.3f} | "
        f"{m['recall']:.3f} | {m['f1']:.3f} {ci(m['f1_ci95'])} | {m['n_groups']}/{m['n_pos_groups']} |"
    )


H = "| model | support/n | TP/FP/FN | P | R | F1 | groups (speakers)/with positives |\n|---|---|---|---|---|---|---|"
out = []
w = out.append
w(f"""# S3 - OE4: classification improvements with what already exists (2026-11-01)

**Self-review by the executing agent; no independent review.** EXPLORATORY. Protocol frozen 2026-11-01 before measuring (`PROTOCOL.md`, sha256 `{proto["sha256"][:16]}...`, annex `PROTOCOL_ANNEX_20261101_1.md`). DIHANA folds 1-4 only (telephone, peninsular Spanish), speaker-grouped nested CV, speaker bootstrap, Holm per family. Fold 5 never loaded. Nothing is declared confirmed; no score is called a calibrated probability. Defaults (V0, P0, `live_cfg_v2`) unchanged: every S3 route is opt-in and experimental.

## 1. E1 - repetition literal on P1 text (same P1/P0 CPU decodes for every text model)
Frame of {e1["n_units"]} label-defined turns (weights = inverse sampling fraction for the sampled negatives), {e1["pos"]} positives from {e1["pos_speakers"]} speakers. Metrics F1 weighted; TP/FP/FN raw.

{H}""")
spk = e1["speakers"]
for k, m in e1["models"].items():
    w(row_w(k, m, spk, e1["pos_speakers"]))
w(
    "\nFamily E1 (Holm 4), paired weighted-F1 difference, speaker bootstrap:\n\n| test | delta [95% CI] | p | p Holm | verdict |\n|---|---|---|---|---|"
)
for d in e1["family_E1"]:
    w(
        f"| {d['test']} | {d['delta']:+.3f} {ci(d['ci'])} | {d['p_boot']:.4f} | {d['p_holm']:.4f} | {d['verdict']} |"
    )
v = e1["vc_fold4_only_descriptive"]
w(
    f"\nVc (fold 4 only, {v['n']} units, {v['pos']} positives; Vc-favourable, descriptive): weighted F1 Vc {v['Vc']['w_f1']:.3f}, A1 {v['A1_ctxLR_P1']['w_f1']:.3f}, rule(P1) {v['rule_P1']['w_f1']:.3f}, V0 {v['V0']['w_f1']:.3f}; A1 minus Vc {v['paired_A1_minus_Vc']['delta']:+.3f} {ci(v['paired_A1_minus_Vc']['ci'])}."
)
dp = e1["descriptive_paired"]
w(
    "Descriptive: "
    + "; ".join(f"{k} {d['delta']:+.3f} {ci(d['ci'])}" for k, d in dp.items())
    + "."
)
w("""
**Reading.** Retraining `ctx_LR` on P1 text (A1) beats the P0 text model and V0, but the gain over the *unretrained* P0 model applied to P1 text and over the one-line rule on P1 text is not demonstrated: the improvement comes from the P1 text itself (more repetitions survive), not from the retraining. The `asr_recovery` control prompt C1 did not reproduce the P1 gain, so what in P1 matters is still unknown. A1 is integrated as the opt-in route `rep_ctx_p1` (experimental) because it was the pre-declared route; it is not shown to beat the rule.""")

w(f"""
## 2. E2 - rule + model fusion
Repetition (P1 frame, table above): `F_OR_rule_A1` and `F_STACK_rule_A1` do not improve on A1 or the rule (learned OR thresholds were finite in 4 of 5 folds, yet the point F1 does not exceed the rule). Filler (5056 turns folds 1-4, P0 ASR, unweighted):

{H}""")
f = e2["filler"]
for k, m in f["models"].items():
    w(row_u(k, m))
w(
    "\nFamily E2 (Holm 4):\n\n| test | delta [95% CI] | p | p Holm | verdict |\n|---|---|---|---|---|"
)
for d in e2["family_E2"]:
    w(
        f"| {d['test']} | {d['delta']:+.3f} {ci(d['ci'])} | {d['p_boot']:.4f} | {d['p_holm']:.4f} | {d['verdict']} |"
    )
w(
    "\nDescriptive filler: "
    + "; ".join(
        f"{k} {d['delta']:+.3f} {ci(d['ci'])}"
        for k, d in f["paired_descriptive"].items()
    )
    + "."
)
g4 = e2["filler_vc_fold4"]
w(
    f"\nVc fusion for filler, fold 4 only ({g4['n']} units, {g4['pos']} positives, nested CV inside fold 4; Vc-favourable):\n\n{H}"
)
for k, m in g4["models"].items():
    w(row_u(k, m))
w(
    "\n"
    + "; ".join(
        f"{k} {d['delta']:+.3f} {ci(d['ci'])}"
        for k, d in g4["paired_descriptive"].items()
    )
    + "."
)
w("""
**Reading.** For `filler_word` the V0 score adds nothing to the lexical rule: the learned OR threshold was "never" in all 5 folds, so the OR equals the rule (identical predictions), and the stacked fusion is statistically tied with it. Fusion beats only the thresholded V0 score (a weak comparator). With Vc (fold 4) every fusion is *worse* than the frozen Vc decision (Vc recall 0.73 is what the rule lacks, but precision drops). Negative result: no fusion is integrated.""")

w(
    "\n## 3. E3 - cross-fitted calibration, abstention band, coverage/risk\nScores: raw score of the source; Brier/ECE with equal-mass (quantile) bins, 10 bins. Band chosen in each outer train on inner cross-fitted calibrated scores (maximise coverage subject to accepted error <= 0.75 x baseline error). All metrics pooled over the outer test folds (all cases).\n"
)
w(
    "| source (n, pos) | method | Brier raw->cal | ECE q10 raw->cal | ECE diff CI | coverage | error among accepted | accepted precision | positives left uncertain | F1 all cases (uncertain=absent) | no-abstention F1 | folds with feasible band |\n|---|---|---|---|---|---|---|---|---|---|---|---|"
)
for name, s in e3["sources"].items():
    ref = s["no_abstention_reference_platt"]
    for m, x in s["methods"].items():
        sel = x["selective"]
        w(
            f"| {name} ({s['n']}, {s['pos']}) | {m} | {x['brier_raw']:.4f}->{x['brier_cal']:.4f} | {x['ece_q10_raw']:.3f}->{x['ece_q10_cal']:.3f} | {ci(x['paired_ece_cal_minus_raw']['ci'])} | {sel['coverage']:.3f} | {sel['error_accepted']:.3f} | {sel['precision_present']:.3f} | {sel['pos_share_uncertain']:.3f} | {sel['f1_all_cases_uncertain_as_absent']:.3f} | {ref['f1_all_cases_uncertain_as_absent']:.3f} | {x['n_bands_feasible']}/5 |"
        )
w(
    "\nFamily E3 (Holm 10), Brier(calibrated) minus Brier(raw):\n\n| test | delta | p Holm | verdict |\n|---|---|---|---|"
)
for d in e3["family_E3"]:
    w(f"| {d['test']} | {d['delta']:+.4f} | {d['p_holm']:.4f} | {d['verdict']} |")
w("""
**Reading.** The raw V0 softmax scores are visibly miscalibrated inside this pool (ECE falls from 0.03-0.09 to 0.01-0.02 after cross-fitted calibration, CI below 0), so the wording "calibration improvement shown within this DIHANA pool" holds for the three V0 sources; it does NOT license calling the service output a calibrated probability on other corpora. The learned text model and the filler stack were already close to calibrated (no demonstrated change). The abstention band lowers error among accepted cases (e.g. filler 0.129 -> about 0.095) at coverage 0.92-0.97, but every accepted-case improvement is paid in recall: positives left uncertain and the all-cases F1 (uncertain counted as absent) is below the no-abstention F1, strongly so for repetition. The coverage/risk indicator is now measurable; it does not reach any OE4 threshold. The band is a trade-off knob, not a quality gain.""")

w(f"""
## 4. E4 - prolongation (observable lengthening; the label is a lengthening mark, not a disfluency claim)
DIHANA folds 1-4: {e4["n"]} turns, {e4["pos"]} positives, {e4["pos_speakers"]} positive speakers; {e4["units_with_pause_mark"]} turns carry a silent-pause mark `p`. Pause-associated FP = FP turn with a `p` mark.

{H.replace("groups (speakers)/with positives", "groups/with positives")}""")
for k, m in e4["folds_1_4"].items():
    w(row_u(k, m) + "")
w("\n| model (folds 1-4) | FP with pause mark | FP without |\n|---|---|---|")
for k, m in e4["folds_1_4"].items():
    w(f"| {k} | {m['fp_with_pause_mark']} | {m['fp_without_pause_mark']} |")
f4 = e4["fold4_vc"]
w(
    f"\nFold 4 only ({f4['n']} units, {f4['pos']} positives; nested CV inside fold 4; Vc-favourable):\n\n{H}"
)
for k, m in f4["models"].items():
    w(row_u(k, m))
w("\n| model (fold 4) | FP with pause mark | FP without |\n|---|---|---|")
for k, m in f4["models"].items():
    w(f"| {k} | {m['fp_with_pause_mark']} | {m['fp_without_pause_mark']} |")
w(
    "\nFamily E4 (Holm 4):\n\n| test | delta [95% CI] | p Holm | verdict |\n|---|---|---|---|"
)
for d in e4["family_E4"]:
    w(
        f"| {d['test']} | {d['delta']:+.3f} {ci(d['ci'])} | {d['p_holm']:.4f} | {d['verdict']} |"
    )
w("""
**Reading.** Fusing Vc (or V0) with observable-lengthening and pause features does not reduce pause-associated false positives and does not change F1 in a demonstrable way. Roughly half of the FP of every variant sit on turns with a `p` mark, whichever model is used. Negative result. Offline only: the service cannot compute these features without a second ASR pass, so prolongation events are only labelled "observable lengthening; disfluent function not demonstrated".

## 5. Service and app (task 5, engineering)
- `mrcd.analysis/v1` revision 1.2 (additive): default scope = filler_word, prolongation, repetition; `block`, `rhetorical_pause`, `neutral_pause`, `revision` only with `--include-out-of-scope` / `include_out_of_scope=true`, marked `out_of_scope_unvalidated`, `in_scope: false`, never in `summary.n_events` / `counts_by_class` / `events_per_minute`; `summary.out_of_scope` reports the detected count (hidden or included). Optional `decision` (`--decision`) and optional `text_route` (`--text-route rep_ctx_p1`, needs the engine's new opt-in `return_words`). Revision 1.1 documents still validate. Schema, fixtures (2 real documents from the CLI on a 36 s clip: default scope 9 in-scope events vs 22 before; all options), docs and tests updated. The text route emitted no repetition event on that clip (the clip has none detected); it is covered by unit tests with synthetic words.
- App patch updated (not applied to the real app): hides out-of-scope classes by default (also for older documents), excludes them from totals/rates, copy "Score (softmax, no calibrado)" instead of "probabilidad de clase". Verified in a temporary clone: `git apply --check` OK, tsc 0, eslint 0, vitest 656 passed.

## 6. What improved / worsened, and what is exploratory
- Improved vs a prior comparator (demonstrated after Holm, exploratory): repetition A1 over V0 and over ctx_LR on P0 text (driven by P1 text); filler fusion over thresholded V0.
- Not demonstrated / no change: A1 vs rule(P1) and vs the P0 model on P1 text; every filler fusion vs the lexical rule; prolongation fusion vs V0thr / Vc; pause-FP reduction.
- Worse: Vc-based filler fusion vs frozen Vc (fold 4); abstention lowers all-cases recall.
- Everything is exploratory: one corpus, units are turns, variety = peninsular telephone Spanish, unknown inter-annotator agreement, fold 4 already used to tune Vc.
- `revision`: only exploratory screening, not evaluated in S3. SOPHIAS discarded. No class is declared confirmed.
""")
(S3 / "README_S3.md").write_text("\n".join(out), "utf8")
print("README_S3.md written")
