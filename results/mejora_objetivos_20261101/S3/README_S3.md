# S3 - OE4: classification improvements with what already exists (2026-11-01)

**Self-review by the executing agent; no independent review.** EXPLORATORY. Protocol frozen 2026-11-01 before measuring (`PROTOCOL.md`, sha256 `053d435cb7609c73...`, annex `PROTOCOL_ANNEX_20261101_1.md`). DIHANA folds 1-4 only (telephone, peninsular Spanish), speaker-grouped nested CV, speaker bootstrap, Holm per family. Fold 5 never loaded. Nothing is declared confirmed; no score is called a calibrated probability. Defaults (V0, P0, `live_cfg_v2`) unchanged: every S3 route is opt-in and experimental.

## 1. E1 - repetition literal on P1 text (same P1/P0 CPU decodes for every text model)
Frame of 1019 label-defined turns (weights = inverse sampling fraction for the sampled negatives), 214 positives from 84 speakers. Metrics F1 weighted; TP/FP/FN raw.

| model | support/n | TP/FP/FN | P | R | F1 | groups (speakers)/with positives |
|---|---|---|---|---|---|---|
| rule_P1 | 214/1019 | 147/34/67 | 0.677 | 0.687 | 0.682 [0.609, 0.745] | 122/84 |
| rule_P0 | 214/1019 | 108/23/106 | 0.674 | 0.505 | 0.577 [0.511, 0.638] | 122/84 |
| V0 | 214/1019 | 93/24/121 | 0.634 | 0.435 | 0.516 [0.446, 0.584] | 122/84 |
| ctx_LR_P0_text | 214/1019 | 109/24/105 | 0.675 | 0.509 | 0.581 [0.511, 0.644] | 122/84 |
| ctx_LR_P0model_on_P1 | 214/1019 | 142/27/72 | 0.720 | 0.664 | 0.690 [0.619, 0.756] | 122/84 |
| A1_ctxLR_P1 | 214/1019 | 144/26/70 | 0.727 | 0.673 | 0.699 [0.620, 0.768] | 122/84 |
| A2_ctxLR_P0P1 | 214/1019 | 142/23/72 | 0.765 | 0.664 | 0.711 [0.641, 0.770] | 122/84 |
| F_OR_rule_A1 | 214/1019 | 148/35/66 | 0.665 | 0.692 | 0.678 [0.605, 0.743] | 122/84 |
| F_STACK_rule_A1 | 214/1019 | 141/27/73 | 0.719 | 0.659 | 0.688 [0.613, 0.755] | 122/84 |

Family E1 (Holm 4), paired weighted-F1 difference, speaker bootstrap:

| test | delta [95% CI] | p | p Holm | verdict |
|---|---|---|---|---|
| E1: A1 minus rule_P1 | +0.017 [-0.018, 0.052] | 0.3387 | 0.6775 | not demonstrated |
| E1: A1 minus V0 | +0.183 [0.098, 0.266] | 0.0004 | 0.0016 | supported after Holm |
| E1: A1 minus ctx_LR_P0_text | +0.118 [0.068, 0.169] | 0.0004 | 0.0016 | supported after Holm |
| E1: A1 minus ctx_LR_P0model_on_P1 | +0.009 [-0.021, 0.035] | 0.5831 | 0.6775 | not demonstrated |

Vc (fold 4 only, 243 units, 43 positives; Vc-favourable, descriptive): weighted F1 Vc 0.563, A1 0.643, rule(P1) 0.625, V0 0.605; A1 minus Vc +0.079 [-0.045, 0.243].
Descriptive: A2_ctxLR_P0P1_minus_A1_ctxLR_P1 +0.012 [-0.015, 0.042]; A1_ctxLR_P1_minus_rule_P0 +0.122 [0.066, 0.177]; F_STACK_rule_A1_minus_A1_ctxLR_P1 -0.011 [-0.036, 0.015].

**Reading.** Retraining `ctx_LR` on P1 text (A1) beats the P0 text model and V0, but the gain over the *unretrained* P0 model applied to P1 text and over the one-line rule on P1 text is not demonstrated: the improvement comes from the P1 text itself (more repetitions survive), not from the retraining. The `asr_recovery` control prompt C1 did not reproduce the P1 gain, so what in P1 matters is still unknown. A1 is integrated as the opt-in route `rep_ctx_p1` (experimental) because it was the pre-declared route; it is not shown to beat the rule.

## 2. E2 - rule + model fusion
Repetition (P1 frame, table above): `F_OR_rule_A1` and `F_STACK_rule_A1` do not improve on A1 or the rule (learned OR thresholds were finite in 4 of 5 folds, yet the point F1 does not exceed the rule). Filler (5056 turns folds 1-4, P0 ASR, unweighted):

| model | support/n | TP/FP/FN | P | R | F1 | groups (speakers)/with positives |
|---|---|---|---|---|---|---|
| rule_asr_lexical | 774/5056 | 399/35/375 | 0.919 | 0.516 | 0.661 [0.619, 0.699] | 180/154 |
| V0_argmax | 774/5056 | 386/202/388 | 0.656 | 0.499 | 0.567 [0.525, 0.609] | 180/154 |
| V0thr | 774/5056 | 424/310/350 | 0.578 | 0.548 | 0.562 [0.522, 0.601] | 180/154 |
| F_OR | 774/5056 | 399/35/375 | 0.919 | 0.516 | 0.661 [0.619, 0.699] | 180/154 |
| F_STACK | 774/5056 | 408/60/366 | 0.872 | 0.527 | 0.657 [0.616, 0.695] | 180/154 |

Family E2 (Holm 4):

| test | delta [95% CI] | p | p Holm | verdict |
|---|---|---|---|---|
| E2-rep: F_OR minus rule_P1 | -0.004 [-0.020, 0.009] | 0.8150 | 1.0000 | not demonstrated |
| E2-rep: F_OR minus A1_ctxLR_P1 | -0.021 [-0.053, 0.010] | 0.1828 | 0.5483 | not demonstrated |
| E2-filler: F_OR minus rule_asr_lexical | +0.000 [0.000, 0.000] | 1.0000 | 1.0000 | not demonstrated |
| E2-filler: F_OR minus V0thr | +0.098 [0.062, 0.135] | 0.0004 | 0.0016 | supported after Holm |

Descriptive filler: F_STACK_minus_V0thr +0.095 [0.061, 0.130]; F_STACK_minus_rule_asr_lexical -0.004 [-0.011, 0.004]; V0thr_minus_V0_argmax -0.004 [-0.020, 0.012].

Vc fusion for filler, fold 4 only (1240 units, 168 positives, nested CV inside fold 4; Vc-favourable):

| model | support/n | TP/FP/FN | P | R | F1 | groups (speakers)/with positives |
|---|---|---|---|---|---|---|
| rule | 168/1240 | 78/6/90 | 0.929 | 0.464 | 0.619 [0.536, 0.701] | 45/37 |
| Vc_frozen | 168/1240 | 123/87/45 | 0.586 | 0.732 | 0.651 [0.582, 0.709] | 45/37 |
| V0_argmax | 168/1240 | 84/54/84 | 0.609 | 0.500 | 0.549 [0.462, 0.629] | 45/37 |
| Vcthr | 168/1240 | 123/87/45 | 0.586 | 0.732 | 0.651 [0.582, 0.709] | 45/37 |
| F_OR_Vc | 168/1240 | 106/76/62 | 0.582 | 0.631 | 0.606 [0.528, 0.673] | 45/37 |
| F_STACK_Vc | 168/1240 | 91/48/77 | 0.655 | 0.542 | 0.593 [0.511, 0.671] | 45/37 |

F_OR_Vc_minus_Vc_frozen -0.045 [-0.087, -0.008]; F_OR_Vc_minus_rule -0.013 [-0.074, 0.046]; F_STACK_Vc_minus_Vc_frozen -0.058 [-0.115, -0.003].

**Reading.** For `filler_word` the V0 score adds nothing to the lexical rule: the learned OR threshold was "never" in all 5 folds, so the OR equals the rule (identical predictions), and the stacked fusion is statistically tied with it. Fusion beats only the thresholded V0 score (a weak comparator). With Vc (fold 4) every fusion is *worse* than the frozen Vc decision (Vc recall 0.73 is what the rule lacks, but precision drops). Negative result: no fusion is integrated.

## 3. E3 - cross-fitted calibration, abstention band, coverage/risk
Scores: raw score of the source; Brier/ECE with equal-mass (quantile) bins, 10 bins. Band chosen in each outer train on inner cross-fitted calibrated scores (maximise coverage subject to accepted error <= 0.75 x baseline error). All metrics pooled over the outer test folds (all cases).

| source (n, pos) | method | Brier raw->cal | ECE q10 raw->cal | ECE diff CI | coverage | error among accepted | accepted precision | positives left uncertain | F1 all cases (uncertain=absent) | no-abstention F1 | folds with feasible band |
|---|---|---|---|---|---|---|---|---|---|---|---|
| V0_filler_word (5056, 774) | platt | 0.1033->0.0900 | 0.084->0.017 | [-0.075, -0.052] | 0.932 | 0.097 | 0.722 | 0.191 | 0.480 | 0.566 | 5/5 |
| V0_filler_word (5056, 774) | isotonic | 0.1033->0.0904 | 0.084->0.009 | [-0.080, -0.057] | 0.919 | 0.095 | 0.716 | 0.211 | 0.484 | 0.566 | 5/5 |
| V0_prolongation (5056, 967) | platt | 0.1254->0.1108 | 0.089->0.015 | [-0.084, -0.054] | 0.900 | 0.145 | 0.574 | 0.176 | 0.565 | 0.587 | 5/5 |
| V0_prolongation (5056, 967) | isotonic | 0.1254->0.1112 | 0.089->0.012 | [-0.087, -0.056] | 0.886 | 0.135 | 0.599 | 0.206 | 0.550 | 0.587 | 5/5 |
| V0_repetition (4989, 214) | platt | 0.0351->0.0308 | 0.028->0.008 | [-0.024, -0.012] | 0.962 | 0.029 | 0.617 | 0.308 | 0.222 | 0.496 | 5/5 |
| V0_repetition (4989, 214) | isotonic | 0.0351->0.0312 | 0.028->0.009 | [-0.025, -0.013] | 0.971 | 0.028 | 0.654 | 0.336 | 0.142 | 0.496 | 5/5 |
| A1_text_P1_repetition (1019, 214) | platt | 0.0296->0.0299 | 0.008->0.012 | [-0.005, 0.007] | 0.963 | 0.023 | 0.928 | 0.313 | 0.519 | 0.678 | 5/5 |
| A1_text_P1_repetition (1019, 214) | isotonic | 0.0296->0.0302 | 0.008->0.012 | [-0.007, 0.007] | 0.953 | 0.021 | 0.946 | 0.449 | 0.393 | 0.678 | 5/5 |
| STACK_filler_word (5056, 774) | platt | 0.0703->0.0705 | 0.021->0.021 | [-0.004, 0.005] | 0.889 | 0.061 | 0.930 | 0.181 | 0.650 | 0.658 | 5/5 |
| STACK_filler_word (5056, 774) | isotonic | 0.0703->0.0695 | 0.021->0.008 | [-0.019, -0.002] | 0.889 | 0.063 | 0.924 | 0.189 | 0.639 | 0.658 | 5/5 |

Family E3 (Holm 10), Brier(calibrated) minus Brier(raw):

| test | delta | p Holm | verdict |
|---|---|---|---|
| E3: Brier(platt) minus Brier(raw), V0_filler_word | -0.0133 | 0.0100 | supported after Holm |
| E3: Brier(isotonic) minus Brier(raw), V0_filler_word | -0.0129 | 0.0100 | supported after Holm |
| E3: Brier(platt) minus Brier(raw), V0_prolongation | -0.0146 | 0.0100 | supported after Holm |
| E3: Brier(isotonic) minus Brier(raw), V0_prolongation | -0.0142 | 0.0100 | supported after Holm |
| E3: Brier(platt) minus Brier(raw), V0_repetition | -0.0043 | 0.0100 | supported after Holm |
| E3: Brier(isotonic) minus Brier(raw), V0_repetition | -0.0039 | 0.0100 | supported after Holm |
| E3: Brier(platt) minus Brier(raw), A1_text_P1_repetition | +0.0003 | 0.5757 | not demonstrated |
| E3: Brier(isotonic) minus Brier(raw), A1_text_P1_repetition | +0.0006 | 0.5757 | not demonstrated |
| E3: Brier(platt) minus Brier(raw), STACK_filler_word | +0.0002 | 0.3268 | not demonstrated |
| E3: Brier(isotonic) minus Brier(raw), STACK_filler_word | -0.0008 | 0.2759 | not demonstrated |

**Reading.** The raw V0 softmax scores are visibly miscalibrated inside this pool (ECE falls from 0.03-0.09 to 0.01-0.02 after cross-fitted calibration, CI below 0), so the wording "calibration improvement shown within this DIHANA pool" holds for the three V0 sources; it does NOT license calling the service output a calibrated probability on other corpora. The learned text model and the filler stack were already close to calibrated (no demonstrated change). The abstention band lowers error among accepted cases (e.g. filler 0.129 -> about 0.095) at coverage 0.92-0.97, but every accepted-case improvement is paid in recall: positives left uncertain and the all-cases F1 (uncertain counted as absent) is below the no-abstention F1, strongly so for repetition. Exploratory result on DIHANA (folds 1-4, 5,056 user turns, 180 speakers, speaker-grouped nested cross-validation; labels are corpus marks, not the double human annotation the original indicator requires). Coverage = share of turns on which the engine emits present/absent; risk = error rate (false positives and false negatives) among those emitted. For `filler_word`, coverage 0.932 (95% speaker-bootstrap CI 0.923-0.940) and risk 0.097 (0.083-0.110): the original thresholds (coverage >= 0.90, risk <= 0.10) are met at the point estimate, but the upper bound of the risk interval exceeds 0.10, so compliance is not demonstrated. For `repetition`, coverage 0.962 (0.954-0.970) and risk 0.029 (0.022-0.036), but the risk criterion is not informative there (always answering 'absent' gives risk 0.043). The F1 over all cases falls (filler 0.566 to 0.480; repetition 0.496 to 0.222). Not a confirmation of the OE4 indicator. The band is a trade-off knob, not a quality gain.

## 4. E4 - prolongation (observable lengthening; the label is a lengthening mark, not a disfluency claim)
DIHANA folds 1-4: 5056 turns, 967 positives, 166 positive speakers; 1101 turns carry a silent-pause mark `p`. Pause-associated FP = FP turn with a `p` mark.

| model | support/n | TP/FP/FN | P | R | F1 | groups/with positives |
|---|---|---|---|---|---|---|
| V0_argmax | 967/5056 | 341/180/626 | 0.655 | 0.353 | 0.458 [0.420, 0.495] | 180/166 |
| V0thr | 967/5056 | 699/736/268 | 0.487 | 0.723 | 0.582 [0.546, 0.614] | 180/166 |
| P_a_stack_V0_signals | 967/5056 | 692/679/275 | 0.505 | 0.716 | 0.592 [0.555, 0.625] | 180/166 |

| model (folds 1-4) | FP with pause mark | FP without |
|---|---|---|
| V0_argmax | 122 | 58 |
| V0thr | 377 | 359 |
| P_a_stack_V0_signals | 383 | 296 |

Fold 4 only (1240 units, 249 positives; nested CV inside fold 4; Vc-favourable):

| model | support/n | TP/FP/FN | P | R | F1 | groups (speakers)/with positives |
|---|---|---|---|---|---|---|
| Vc_frozen | 249/1240 | 181/129/68 | 0.584 | 0.727 | 0.648 [0.578, 0.710] | 45/43 |
| Vcthr | 249/1240 | 171/122/78 | 0.584 | 0.687 | 0.631 [0.565, 0.690] | 45/43 |
| V0_argmax | 249/1240 | 95/39/154 | 0.709 | 0.382 | 0.496 [0.425, 0.563] | 45/43 |
| P_b_stack_Vc_signals | 249/1240 | 179/121/70 | 0.597 | 0.719 | 0.652 [0.586, 0.711] | 45/43 |

| model (fold 4) | FP with pause mark | FP without |
|---|---|---|
| Vc_frozen | 64 | 65 |
| Vcthr | 62 | 60 |
| V0_argmax | 28 | 11 |
| P_b_stack_Vc_signals | 62 | 59 |

Family E4 (Holm 4):

| test | delta [95% CI] | p Holm | verdict |
|---|---|---|---|
| E4: P-a minus V0thr, F1 | +0.010 [-0.006, 0.026] | 0.9166 | not demonstrated |
| E4: P-a minus V0thr, pause-associated FP count | +6.000 [-14.000, 27.000] | 1.0000 | not demonstrated |
| E4: P-b minus Vc, F1 | +0.005 [-0.025, 0.033] | 1.0000 | not demonstrated |
| E4: P-b minus Vc, pause-associated FP count | -2.000 [-11.000, 7.000] | 1.0000 | not demonstrated |

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

## Addendum 2026-11-01 (independent review S; nothing above was edited)
* **STACK filler calibration.** The sentence "the filler stack was already close to calibrated (no demonstrated change)" is imprecise. The table in section 3 shows STACK_filler_word isotonic ECE 0.021 -> 0.008, CI [-0.019, -0.002] (below 0): a calibration improvement *within this DIHANA pool*, as for the V0 sources. What is not demonstrated is a change of Brier (Holm-corrected family: platt p=0.33, isotonic p=0.28). ECE CIs are outside the Holm family (only Brier is corrected), so the ECE improvement is descriptive.
* **Unit of the text route (not evaluated).** `rep_ctx_p1` was trained and evaluated on whole DIHANA turns; the service applies it to ASR **segments** split at pauses >= 0.6 s or 30 words. **That segment-level behaviour was NOT evaluated.** Until this addendum the declaration lived only in the post-dated protocol annex (written after the results); it is now also in `docs/contracts/APP_INTEGRATION_CONTRACT.md`.
* **Annex timing.** `PROTOCOL_ANNEX_20261101_1.md` is dated after the results; its statement that no rule changed after seeing results is not independently verifiable, and the choice of Platt vs isotonic for the service used E3 results (post-hoc, as declared).
* `decision.estimate` with `lo`/`hi` is a development-fitted mapping with `calibration_claim: "none"`; it is not a probability of correctness and `lo`/`hi` are abstention bounds, not a confidence interval.
