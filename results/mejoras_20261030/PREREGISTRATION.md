# Pre-registration of the improvement experiments (written 2026-10-30, BEFORE any fold-5 / EPIC / AMERESCO number of a candidate was computed)

New files only. Scripts: `scripts/improve/`, modules: `mrcd/reproducibility.py`, `mrcd/windowing.py`, `mrcd/decision.py`, `mrcd/adaptation.py`.
The only fold-5 / EPIC / AMERESCO numbers that exist at the time of writing are the V0 baseline numbers of `results/baselines_20261030` (previous study).

## Data and partitions (fixed)

- DIHANA unit = user turn. Official 5 folds by speaker. **dev** = folds 1-3 (turns whose frozen ASR cache exists when the list is frozen; the order of the manifest is a seeded random order, so any prefix is a random sample; list stored in `dev_units_frozen.json`), **tune** = fold 4, **test** = fold 5 (1223 turns, 45 speakers), reserved.
- EPIC-136 (filler) and AMERESCO-98 (prolongation) are external, zero-shot, never used to select anything.
- Classes evaluated with F1: filler_word (DIHANA `f`), prolongation (`a`), repetition (`r:repeticion`), revision (`r:sustitucion|insercion|borrado`, partial, low support). block / rhetorical_pause / neutral_pause: no F1 (no reference).
- Absence of annotation is not a verified negative outside DIHANA/EPIC-filler; DIHANA `<b>`, EPIC CORRECTION are NOT mapped. No new annotation, no automatic judge.
- Same units, references, partitions and comparators for every version. ASR is read ONLY from the frozen caches (OMP_NUM_THREADS=4).

## Versions and experiments

- **V0** = base checkpoint, zero-pad short turns, rule "any window argmax == class". Comparators: always_present, always_absent, rule_existing_autolabeler, rule_asr_lexical (same fixed rules as the baseline study).
- **Ea (short turns)**. Candidates: A_noise (pad with noise at the unit's noise floor instead of digital zeros); A_abs(tau) abstain on units shorter than tau, tau in {0.5, 1.0, 1.5} s (abstention = predicted absent, and counted); A_noise+abs(tau). Decision rule held at V0's to isolate the effect.
  Selection criterion (dev only): highest dev macro-F1 over {filler_word, prolongation, repetition}, adopted only if the paired by-speaker bootstrap 95% CI of dMacroF1 vs V0 on dev has lower bound > 0 AND dMacroF1 on tune >= 0. Otherwise Va = V0 ("not adopted").
- **Eb (decision rule)**. Per class: score = aggregate over windows of the class probability, agg in {max, mean, top2}; threshold = argmax F1 on a 0.02 grid, fitted on dev only, with 5-fold speaker-grouped CV (out-of-fold F1 = honest dev estimate; final threshold fitted on all dev). agg chosen per class by highest out-of-fold dev F1. A class adopts the threshold rule only if dev OOF F1 > V0 dev F1 AND tune F1 >= V0 tune F1 (same units); otherwise that class keeps V0's rule. Frozen in `frozen_selection.json` (sha256 in later configs) before any test evaluation. Probabilities are NOT calibrated: these are scores.
  Vab = adopted Ea features + Eb thresholds re-fitted on those features (same protocol).
- **Ec (supervised adaptation)**. Head on the FROZEN pooled representation of the base network (multiple-instance: turn label = human presence of f / a / r, turn score = max over windows, BCE with pos_weight). Trained on dev (folds 1-3). Grid: head in {linear, mlp32}, weight decay in {1e-4, 1e-3, 1e-2}, lr 1e-2, up to 200 epochs, snapshot every 10. Selection of (head, wd, epoch) by mean average precision over the 3 classes on fold 4 (tune); per-class threshold = argmax F1 on fold 4. New checkpoint with sha256 + config; base checkpoint untouched. Zero-shot on EPIC (filler) / AMERESCO (prolongation) with the same frozen threshold. Revision is not trained (V0 decision kept in the table).
  Vc = features of V0 (zero pad). Vac = features of the adopted Ea variant (if any).
- Leakage guards: speakers verified disjoint across DIHANA folds; the head never sees fold 4/5, EPIC, AMERESCO labels except fold 4 for selection as declared.

## Claim rule (final evaluation, fold 5 / EPIC / AMERESCO)

Per class and version: support, TP/FP/FN/TN, P, R, F1 with 95% bootstrap CI by speaker (2000 resamples, seed 13). Paired comparison on the SAME units against V0 and against the best untrained comparator chosen on DEV (filler: rule_asr_lexical, repetition: rule_asr_lexical, prolongation: best of always_present / rule_existing on dev): paired by-speaker bootstrap of dF1 (version - comparator).

- "Improvement demonstrated" only if the 95% CI of dF1 has lower bound > 0 (DIHANA, EPIC: speakers/actors). "Worsening" if the upper bound < 0. Otherwise "not demonstrated".
- AMERESCO has 4 conversations: no CI by group is meaningful; same-unit exact McNemar test on correct/incorrect pairs (two-sided, alpha 0.05) plus discordant counts; reported as unit-level evidence only, flagged as non-independent.

## Classes without reference (no F1)

Determinism (repeat, torch threads), robustness to additive noise (SNR 30/20 dB) and gain (x0.5, x2), invariance to extra trailing padding, 3-state coherence, emission rate on DIHANA and its Spearman correlation with pause duration measured from the VAD track, descriptive co-occurrence with DIHANA silent-pause mark `p` (NOT a validation), what reference is missing.
