# Results A (+E): single-pass P1 decision profile `o2026_12` (EXPLORATORY, DIHANA folds 1-4)

Every figure in this file is **exploratory (DIHANA folds 1-4)**: one corpus (telephone, peninsular Spanish,
train-information domain), turn-level scoring, transcript-mark references, speaker-cluster bootstrap (5 000 resamples,
seed 13, percentile 95 % CI). Pipeline figures are outer-fold cross-validated predictions. Nothing here is confirmatory.
Protocol: `PROTOCOL_EXPLORATORY_A.md` (sha256 `e58b912f...`, committed in `03710cb` before any decode, fit or metric);
aggregates: `results_a.json`, `g4/g4_summary.json`. Per-turn files stay in the git-ignored `cache/`.
Deviations from the protocol, recorded after the results: `ANNEX_20261108_DEVIATIONS.md` (sha256 in `ANNEX_20261108_DEVIATIONS.sha256.json`).

## Verdict

| item                                                                   | result                                                                                          |
| ---------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| HA1 repetition F1 `o2026_12` - V0                                      | +0.066 [-0.005, 0.134], p_Holm 0.135: **not supported**                                         |
| HA2 repetition F1 `o2026_12` - p0_rule                                 | +0.009 [-0.061, 0.078], p_Holm 0.830: **not supported**                                         |
| HA3 repetition F1 `o2026_12` - p1_rule (demotion helps)                | **-0.089** [-0.138, -0.039], p_Holm 0.0012: **not supported; the layer is significantly worse** |
| demotion raises precision without losing recall                        | **no**: precision +0.085 [0.031, 0.143], recall -0.224 [-0.275, -0.175]                         |
| GF filled pause, F1 `o2026_12` - `o2026_11` >= -0.02 and CI upper >= 0 | -0.0215 [-0.048, 0.003]: **fails** (by 0.0015 on the point estimate)                            |
| GP prolongation, same rule                                             | +0.0045 [-0.011, 0.020]: **passes**                                                             |
| GL latency, median wait after close <= 10 s | 9.885 s (rounds 9.88, 9.89): **met on one machine, n=2 measured rounds, margin 0.115 s, load-sensitive, not robust** |

`o2026_12` stays opt-in. Nothing is adopted; the default remains `o2026_11`.

## Reading

1. **The single P1 pass helps repetition; the demotion layer as frozen does not.** With the generator alone, P1 text lifts
   the confirmatory rule from 0.545 (P0) to 0.642 (+0.097 [0.055, 0.141], descriptive), mostly by recall
   (+0.173). The frozen contextual layer then removes that gain: `o2026_12` (all four rules) is 0.553.
2. **R4 (polarity -> functional) is the cause, as the protocol anticipated but in a larger amount than expected.** Among
   the 100 turns the layer flips from positive to negative, 48 are reference positives and 52 negatives; R4 is present on 39
   of the positive flips. DIHANA marks emphatic "si, si" as literal repetition, and the reference has no function label,
   so this is disagreement with a function-blind reference, not evidence that the functional reading is wrong. It is
   still a recall loss on the only available reference, and it is reported as such.
3. **The uncertain-only rules (R1-R3, `o2026_12_u`, declared secondary) are neutral:** 0.645 vs 0.642,
   +0.003 [-0.021, 0.028]; precision +0.042 [0.015, 0.075] for recall -0.042 [-0.072, -0.016]. R3 (seam) is present on 6
   reference-positive and 7 negative flipped turns, so on this corpus it removes about as many hits as errors.
4. **Candidate-level demotion share is dominated by ASR hallucination loops.** 2 096 of the 2 372 P1 candidates were
   demoted by R1 (loop), but they come from only 15 turns (up to 432 candidates in one turn; these two figures are from the untracked cache, not reproducible from tracked files). At turn level R1 is present on 2
   positive and 12 negative flipped turns. The candidate-level share (0.930) is therefore not a measure of how often the layer
   intervenes; the turn-flip counts are.
5. **P1 hurts the filled-pause pipeline's precision.** Re-fitting the same restricted pipeline on P1 text instead of
   P0 text costs 0.034 F1 [-0.058, -0.012] (precision -0.078, recall +0.007): P1 transcribes more non-lexical tokens that
   DIHANA does not mark (its prompt contains "Eh..." and "Mmm"). The restricted search itself is not the cause
   (`p0_restricted` - `o2026_11` = +0.013 [-0.001, 0.026]). This makes GF fail by 0.0015.
6. **Prolongation is unaffected** (+0.0045; GP passes). The final fit chose C = 0.3 from the frozen grid (o2026_11: 0.03).
7. **Lexical fillers have a usable DIHANA reference** (`d tipo="rellenar"`, 218 positive turns): the auto-labeler evidence
   rule on P1 words gives precision 0.644 [0.556, 0.721], recall 0.757 [0.675, 0.831], F1 0.696 [0.621, 0.758]; the
   lenient reading (disfluent + uncertain) has recall 0.807 at precision 0.408. On P0 words the same rule gives 0.683.
   Caveat: the negative coverage of DIHANA `d` marks was not verified, and the precision exceeded the range written
   before the run. Adding lexical fillers to `filler_word` lowers agreement with the f-mark reference (union F1 0.639 vs
   0.675), as expected, because that reference marks only non-lexical filled pauses: the two subclasses must be reported
   separately, never pooled into one filler F1.
8. **Descriptive:** V0 computed on P1 engine features reaches repetition F1 0.591 vs 0.487 on P0 features, so P1 text also
   helps the engine's own window model.

## Latency (G4 procedure, measured 2026-11-08T06:51Z)

Real service process on test port 8791, `--incremental --device cpu --preload p0,p1_disfluent_prompt --warmup`,
`OMP_NUM_THREADS=4`, the same 120 s recording as G4, real-time pace, 1 discarded cold round + 2 measured rounds.

| profile | wait after close, measured rounds | median | cold round (discarded) | offline job, measured rounds | offline RTF (median) |
|---|---|---|---|---|---|
| `o2026_12` (one P1 pass) | 9.88 s, 9.89 s | **9.885 s** | 10.40 s | 63.85 s, 50.55 s | 0.477 |
| `o2026_11` (same process, reference) | 9.56 s, 9.49 s | 9.525 s | 9.93 s | 41.14 s, 40.88 s | 0.342 |

All documents `completed` and identical to the offline document of the same round; every `o2026_12` document reports
`asr_profile: p1_disfluent_prompt` (one pass). Reading: the single pass removes the `o2026_11b` failure (13.5 s with two
passes), and the wait after close is about 0.36 s above `o2026_11`. The 10 s limit is met by 0.115 s only: the cold round
was above it, and the offline rounds of `o2026_12` vary from 50.6 s to 63.9 s with 5-21 % background CPU reported before
them (shared machine; Whisper int8 is load-sensitive). This is a single-machine measurement, not a guarantee; under load
the limit can be missed. The exclusion of the cold round was pre-declared in protocol section 5.

## Expectations written before the run vs observed

| quantity                                  | expected                      | observed                            | within                   |
| ----------------------------------------- | ----------------------------- | ----------------------------------- | ------------------------ |
| repetition F1 V0                          | [0.40, 0.60]                  | 0.487                               | yes                      |
| repetition F1 p0_rule                     | [0.50, 0.65]                  | 0.545                               | yes                      |
| repetition F1 p1_rule                     | [0.60, 0.75]                  | 0.642                               | yes                      |
| share of candidates demoted               | [0.03, 0.20], R4-dominated    | 0.930, R1-dominated (15 loop turns) | **no**                   |
| F1 o2026_12 - p1_rule                     | [-0.06, +0.02]                | -0.089                              | **no** (worse)           |
| F1 o2026_12_u - p1_rule                   | [-0.01, +0.02]                | +0.003                              | yes                      |
| HA2 supported, HA1 likely                 |                               | neither supported                   | **no**                   |
| filled pause F1 / delta vs o2026_11       | [0.66, 0.73] / [-0.03, +0.03] | 0.675 / -0.0215                     | yes / yes (but GF fails) |
| prolongation F1 / delta                   | [0.59, 0.64] / [-0.02, +0.01] | 0.623 / +0.0045                     | yes / yes                |
| lexical filler precision / lenient recall | [0.10, 0.40] / [0.50, 0.85]   | 0.644 / 0.807                       | **no** (higher) / yes    |
| G4 wait after close / offline RTF | [8, 14] s / [0.35, 0.50] | 9.885 s / 0.477 | yes (one machine, n=2, margin 0.115 s, load-sensitive, not robust) / yes |

## What was NOT measured, and why

- Accuracy of the `functional` state: DIHANA has no function labels (`not_available`).
- Lexical fillers outside DIHANA, and whether unmarked `pues` / `bueno` are true negatives (`d` coverage unverified).
- Event timing: scoring is turn-level; intervals are not compared with human boundaries.
- The service unit (pause-split ASR segments) differs from the DIHANA turn used here; that domain shift is unmeasured,
  as for `o2026_11`.
- Any other corpus: the NOCANDO cell is spent, and AMERESCO, EPIC and CHM150 were not read; fold-5 rows are filtered out after parsing `turns.csv` and never used (`evaluate.py:80-84`).

## Process notes

- P1 words: 1 025 frozen ASR-recovery decodes plus 4 031 new ones (same code path; a re-decode of 8 frozen turns gave
  identical text 8/8; counts in the annex, evidence in the untracked cache). P0 and P1 engine features were recomputed from audio with `DisfluencyEngine.unit_features`
  because the stored campaign feature shards also contain fold-5 rows and were not loaded.
- The host killed two decode shards and the feature job once each (allocation errors under memory pressure); the
  resumable loop restarted them from the per-turn cache with no lost or duplicated work.
- `V0_p1_features` and the lexical-filler P0 row are descriptive additions computed from the same cache; they are not
  in the hypothesis family.

## Numbers (generated by `scripts/plan_a/render_tables.py` from `results_a.json` and `g4/g4_summary.json`)
protocol sha256 `e58b912faaefde491a2a36e99194157d229015c446bcdacedcabd1d6d830b17e`, git `666f31dfb3`, exploratory (DIHANA folds 1-4)

### Repetition (4989 turns, 180 speakers)

| system | TP | FP | FN | precision [95 % CI] | recall [95 % CI] | F1 [95 % CI] |
|---|---|---|---|---|---|---|
| V0 | 93 | 75 | 121 | 0.554 [0.463, 0.640] | 0.435 [0.370, 0.500] | **0.487** [0.421, 0.546] |
| p0_rule | 110 | 80 | 104 | 0.579 [0.486, 0.662] | 0.514 [0.446, 0.581] | **0.545** [0.478, 0.601] |
| p1_rule | 147 | 97 | 67 | 0.603 [0.520, 0.677] | 0.687 [0.628, 0.744] | **0.642** [0.580, 0.695] |
| o2026_12 | 99 | 45 | 115 | 0.688 [0.601, 0.762] | 0.463 [0.395, 0.527] | **0.553** [0.487, 0.612] |
| o2026_12_u | 138 | 76 | 76 | 0.645 [0.559, 0.722] | 0.645 [0.585, 0.704] | **0.645** [0.582, 0.697] |
| V0_p1_features | 127 | 89 | 87 | 0.588 [0.505, 0.667] | 0.594 [0.525, 0.660] | **0.591** [0.525, 0.650] |

| contrast | ΔF1 [95 % CI] | p_boot | Δprecision [95 % CI] | Δrecall [95 % CI] | p_Holm | verdict |
|---|---|---|---|---|---|---|
| HA1_o2026_12_minus_V0 | 0.0662 [-0.005, 0.134] | 0.0676 | 0.1339 [0.052, 0.213] | 0.0280 [-0.052, 0.104] | 0.1352 | not supported |
| HA2_o2026_12_minus_p0_rule | 0.0085 [-0.061, 0.078] | 0.8302 | 0.1086 [0.032, 0.182] | -0.0514 [-0.132, 0.030] | 0.8302 | not supported |
| HA3_o2026_12_minus_p1_rule | -0.0888 [-0.138, -0.039] | 0.0004 | 0.0850 [0.031, 0.143] | -0.2243 [-0.275, -0.175] | 0.0012 | not supported |
| o2026_12_u_minus_p1_rule (descriptive) | 0.0029 [-0.021, 0.028] | 0.7998 | 0.0424 [0.015, 0.075] | -0.0421 [-0.072, -0.016] | - | - |
| p1_rule_minus_p0_rule (descriptive) | 0.0974 [0.055, 0.141] | 0.0004 | 0.0235 [-0.020, 0.065] | 0.1729 [0.120, 0.227] | - | - |
| o2026_12_minus_o2026_12_u (descriptive) | -0.0918 [-0.139, -0.045] | 0.0004 | 0.0426 [-0.009, 0.098] | -0.1822 [-0.234, -0.133] | - | - |

Demotion: 2372 candidates; states {'disfluent': 167, 'functional': 90, 'uncertain': 2115}; share demoted 0.930; by rule {'loop': 2096, 'gap': 4, 'seam': 15, 'polarity': 90} (shares {'loop': 0.8836, 'gap': 0.0017, 'seam': 0.0063, 'polarity': 0.0379}); matched (any order) {'loop': 2096, 'gap': 12, 'seam': 23, 'polarity': 600}.
Turn flips 1 -> 0: {'to_negative_on_reference_positive': 48, 'to_negative_on_reference_negative': 52}; rules present on flipped turns: {'positive': {'loop': 2, 'gap': 2, 'seam': 6, 'polarity': 39}, 'negative': {'loop': 12, 'gap': 2, 'seam': 7, 'polarity': 31}}.
Raises precision without losing recall: **False** (true iff delta precision (o2026_12 - p1_rule) CI lower > 0 and delta recall >= -0.02 (protocol section 4)).

| outer | n | pos | V0 F1 | p0_rule F1 | p1_rule F1 | o2026_12 F1 |
|---|---|---|---|---|---|---|
| 0 | 1027 | 72 | 0.487 | 0.619 | 0.719 | 0.569 |
| 1 | 999 | 23 | 0.393 | 0.393 | 0.412 | 0.368 |
| 2 | 963 | 35 | 0.426 | 0.438 | 0.564 | 0.433 |
| 3 | 948 | 42 | 0.559 | 0.567 | 0.744 | 0.636 |
| 4 | 1052 | 42 | 0.538 | 0.605 | 0.674 | 0.641 |

| dihana_fold | n | pos | V0 F1 | p0_rule F1 | p1_rule F1 | o2026_12 F1 |
|---|---|---|---|---|---|---|
| 1 | 1333 | 69 | 0.508 | 0.597 | 0.705 | 0.598 |
| 2 | 1271 | 58 | 0.505 | 0.621 | 0.672 | 0.581 |
| 3 | 1162 | 44 | 0.390 | 0.427 | 0.580 | 0.468 |
| 4 | 1223 | 43 | 0.529 | 0.500 | 0.580 | 0.535 |

### filler_word (reference: filler_present (all DIHANA f marks: non-lexical filled pauses))

| system | TP | FP | FN | precision [95 % CI] | recall [95 % CI] | F1 [95 % CI] |
|---|---|---|---|---|---|---|
| o2026_11 | 510 | 180 | 264 | 0.739 [0.691, 0.782] | 0.659 [0.616, 0.699] | **0.697** [0.658, 0.731] |
| p0_restricted | 524 | 179 | 250 | 0.745 [0.698, 0.786] | 0.677 [0.634, 0.719] | **0.710** [0.671, 0.745] |
| o2026_12_filled_pause | 529 | 264 | 245 | 0.667 [0.620, 0.709] | 0.683 [0.646, 0.720] | **0.675** [0.638, 0.708] |

| contrast | ΔF1 [95 % CI] | p_boot | Δprecision [95 % CI] | Δrecall [95 % CI] | gate |
|---|---|---|---|---|---|
| o2026_12 - o2026_11 (gate) | -0.0215 [-0.048, 0.003] | 0.0816 | -0.0720 [-0.098, -0.048] | 0.0245 [-0.011, 0.059] | **FAIL** |
| o2026_12 - p0_restricted (ASR effect) | -0.0344 [-0.058, -0.012] | 0.0048 | -0.0783 [-0.105, -0.054] | 0.0065 [-0.024, 0.036] | - |
| p0_restricted - o2026_11 (search restriction) | 0.0128 [-0.001, 0.026] | 0.0680 | 0.0062 [-0.013, 0.025] | 0.0181 [0.000, 0.036] | - |

| outer | n | pos | o2026_11 F1 | p0_restricted F1 | o2026_12_filled_pause F1 |
|---|---|---|---|---|---|
| 0 | 1040 | 213 | 0.735 | 0.776 | 0.729 |
| 1 | 1008 | 132 | 0.650 | 0.672 | 0.636 |
| 2 | 976 | 135 | 0.677 | 0.677 | 0.634 |
| 3 | 962 | 126 | 0.699 | 0.682 | 0.662 |
| 4 | 1070 | 168 | 0.701 | 0.704 | 0.680 |

P1 per-fold selection (C, threshold, calibration): fold 0: C=0.3, thr=0.28, platt; fold 1: C=3.0, thr=0.30, none; fold 2: C=0.03, thr=0.34, isotonic; fold 3: C=0.3, thr=0.32, none; fold 4: C=3.0, thr=0.32, none

### prolongation (reference: lengthening_present)

| system | TP | FP | FN | precision [95 % CI] | recall [95 % CI] | F1 [95 % CI] |
|---|---|---|---|---|---|---|
| o2026_11 | 696 | 587 | 271 | 0.542 [0.498, 0.587] | 0.720 [0.683, 0.753] | **0.619** [0.582, 0.652] |
| p0_restricted | 718 | 614 | 249 | 0.539 [0.495, 0.582] | 0.743 [0.708, 0.775] | **0.625** [0.589, 0.657] |
| o2026_12_prolongation | 697 | 573 | 270 | 0.549 [0.507, 0.589] | 0.721 [0.685, 0.756] | **0.623** [0.589, 0.653] |

| contrast | ΔF1 [95 % CI] | p_boot | Δprecision [95 % CI] | Δrecall [95 % CI] | gate |
|---|---|---|---|---|---|
| o2026_12 - o2026_11 (gate) | 0.0045 [-0.011, 0.020] | 0.5807 | 0.0063 [-0.009, 0.021] | 0.0010 [-0.022, 0.024] | **pass** |
| o2026_12 - p0_restricted (ASR effect) | -0.0015 [-0.015, 0.012] | 0.8278 | 0.0098 [-0.003, 0.023] | -0.0217 [-0.042, -0.002] | - |
| p0_restricted - o2026_11 (search restriction) | 0.0060 [-0.005, 0.017] | 0.2807 | -0.0034 [-0.016, 0.009] | 0.0228 [0.007, 0.040] | - |

| outer | n | pos | o2026_11 F1 | p0_restricted F1 | o2026_12_prolongation F1 |
|---|---|---|---|---|---|
| 0 | 1040 | 237 | 0.664 | 0.664 | 0.643 |
| 1 | 1008 | 173 | 0.633 | 0.612 | 0.624 |
| 2 | 976 | 173 | 0.591 | 0.607 | 0.623 |
| 3 | 962 | 179 | 0.630 | 0.668 | 0.651 |
| 4 | 1070 | 205 | 0.572 | 0.572 | 0.577 |

P1 per-fold selection (C, threshold, calibration): fold 0: C=0.03, thr=0.34, none; fold 1: C=0.03, thr=0.30, none; fold 2: C=0.03, thr=0.28, none; fold 3: C=0.03, thr=0.28, none; fold 4: C=0.03, thr=0.26, platt

### Lexical filler (turn has >= 1 DIHANA d tipo=rellenar mark (discourse marker with a filling function); negative coverage of d marks not verified)

| system | TP | FP | FN | precision [95 % CI] | recall [95 % CI] | F1 [95 % CI] |
|---|---|---|---|---|---|---|
| o2026_12_lexical_disfluent | 165 | 91 | 53 | 0.644 [0.556, 0.721] | 0.757 [0.675, 0.831] | **0.696** [0.621, 0.758] |
| o2026_12_lexical_lenient_disfluent_or_uncertain | 176 | 255 | 42 | 0.408 [0.324, 0.485] | 0.807 [0.732, 0.873] | **0.542** [0.456, 0.615] |
| p0_lexical_disfluent | 165 | 100 | 53 | 0.623 [0.526, 0.703] | 0.757 [0.670, 0.830] | **0.683** [0.598, 0.750] |

Candidate states (P1): {'n_candidates': 480, 'by_state': {'disfluent': 279, 'functional': 0, 'uncertain': 201}}. By item: {'bueno': {'disfluent': 60, 'uncertain': 0, 'functional': 0}, 'este': {'disfluent': 19, 'uncertain': 44, 'functional': 0}, 'a ver': {'disfluent': 26, 'uncertain': 10, 'functional': 0}, 'tipo': {'disfluent': 13, 'uncertain': 132, 'functional': 0}, 'pues': {'disfluent': 149, 'uncertain': 13, 'functional': 0}, 'o sea': {'disfluent': 12, 'uncertain': 2, 'functional': 0}}.

Union against the f-mark reference (descriptive):

| system | TP | FP | FN | precision [95 % CI] | recall [95 % CI] | F1 [95 % CI] |
|---|---|---|---|---|---|---|
| o2026_12_union | 548 | 394 | 226 | 0.582 [0.534, 0.628] | 0.708 [0.673, 0.742] | **0.639** [0.599, 0.675] |
| o2026_12_filled_pause_only | 529 | 264 | 245 | 0.667 [0.620, 0.709] | 0.683 [0.646, 0.720] | **0.675** [0.638, 0.708] |

### Latency (G4 procedure)

| profile / mode | median s | measured rounds s | equal to offline | background CPU % |
|---|---|---|---|---|
| o2026_12|offline | 57.2 | [63.85, 50.55] | True | [20.6, 5.0] |
| o2026_12|incremental | 9.885000000000002 | [9.88, 9.89] | True | [4.5, 3.0] |
| o2026_11|offline | 41.010000000000005 | [41.14, 40.88] | True | [0.9, 1.6] |
| o2026_11|incremental | 9.525 | [9.56, 9.49] | True | [1.6, 0.5] |

G4: {'profile': 'o2026_12', 'wait_after_close_median_s': 9.885000000000002, 'limit_s': 10.0, 'pass': True, 'rtf_offline_median': 0.4767, 'measured_utc': '2026-11-08T06:51:23Z', 'procedure': 'scripts/mejora_servicio/g4_benchmark.py (1 cold discarded + 2 measured rounds, 120 s, real time)'}
