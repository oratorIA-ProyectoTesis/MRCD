# Exploratory protocol A (+E): single-pass P1 decision profile `o2026_12` (2026-11-08)

Status: **exploratory, DIHANA folds 1-4 only**. Written and committed BEFORE any decode, feature extraction, fit or
metric of this plan. The sha256 of this file (CRLF normalised to LF) is recorded in every result JSON of
`results/plan_a_20261108/`. Any later change is an annex with its own hash; this file is not edited after the commit.

## 0. Why

Measured facts this plan starts from (all already published in the repository):

- Confirmatory NOCANDO cell (`results/confirmatoria_20261107/PRIMARY_CONCLUSIONS.md`): default (`o2026_11`) repetition
  F1 0.286 [0.194, 0.380] vs a one-line lexical rule on the same P0 words 0.525 [0.430, 0.632] (FP 97 vs 51); `default`
  and `v0` emit identical repetition events; filler recall 0.541 [0.379, 0.683]; the default flags 2.36 filler units/min
  vs 0.65/min of reference units. Secondary arm (`SECONDARY_CONCLUSIONS_CW2.md`): cw2 ASR + the same rule 0.781 / 0.784;
  most of the gain comes from the ASR with the rule held fixed. The confirmatory cell is spent and is not used here.
- ASR recovery (`results/asr_recovery_20261031/EXPERIMENTO_ASR.md`, `PERFIL_P1.md`, `C1_CONTROL.md`): on the 1 025-turn
  frame of DIHANA folds 1-4, P0 keeps 49.1 % of the human literal-repetition marks and P1 (`p1_disfluent_prompt`, only
  `initial_prompt` differs, `mrcd/asr_profiles.py`) keeps 67.3 %; ctx_LR +0.110 w-F1 [0.064, 0.159]; decode cost +7 %;
  a fluent control prompt does not reproduce the gain and the cause is unknown. P1 is licence-clean (same Whisper weights).
- `o2026_11` (`mrcd/service/decision_profile.py`, `mrcd/service/profiles/o2026_11/`) replaces only `filler_word`
  (LR C=3.0, blocks rule+v0+ctxfill+sig+pause, fusion or_rule) and `prolongation` (LR C=0.03, blocks
  v0+sig+lenx+pause+emb), both fitted on P0 text of the 5 056 turns of DIHANA folds 1-4. Repetition stays V0.
- `o2026_11b` adds P1 as a SECOND ASR pass for repetition and failed gate G4 (13.5 s wait after close vs the 10 s limit;
  offline RTF 0.715 vs 0.35; `results/mejora_servicio_20261102/README_RESULTADOS.md`). The cost is the second pass.
- The thesis architecture (`Documentos/ALCANCE-MOTOR.MD`, sections 2-3) is candidates -> contextual classification into
  disfluent / functional / uncertain. Today repetition bypasses it.

## 1. Candidate under evaluation: decision profile `o2026_12` (opt-in; the default stays `o2026_11`)

ONE ASR pass with `p1_disfluent_prompt` (faster-whisper small, int8, CPU, 8 s chunks / 2 s overlap, the existing profile,
unchanged) feeds ALL classes. Nothing else of the ASR changes.

### 1.1 Repetition = candidate generator + contextual demotion layer

**Generator (identical definition to the confirmatory `rule`, `scripts/confirmatoria/run_confirmatoria.py`
`rule_events`):** tokens = `norm_tokens` of every ASR word (NFD, accents removed, lower case, `[^a-z0-9ñ ]` -> space),
each token inheriting its word's times; non-lexical filled pauses (`NONLEXICAL_FILLER_RE`, conjunction-`e` exception) are
removed; every immediate exact repetition of 1-4 tokens in the remaining stream is a candidate `(pos, n)`; copy 1 = tokens
`[pos-n, pos)`, copy 2 = tokens `[pos, pos+n)`; the event interval is copy 2 (as in the confirmatory rule). The stream is the
whole recording (service) or the whole turn (DIHANA evaluation).

**Contextual layer (may only DEMOTE; it never creates, moves or extends a candidate).** Each candidate starts as
`disfluent`. Rules are evaluated in this order; the first that matches sets the state (every matching rule is logged for
the statistics). All thresholds are pre-existing engine constants; none is tuned here.

| id          | condition (from existing engine signals)                                                                                                                                                                                                                                                                                                                                                                           | state        |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------ |
| R1 loop     | the candidate's token range `[pos-n, pos+n)` intersects a hallucination loop found by `asr_profiles._scan_loops` (thresholds `LOOP_MIN_REPEATS`: 1-gram >= 5, 2-4-gram >= 4 back-to-back repeats) on the same filler-stripped stream                                                                                                                                                                               | `uncertain`  |
| R2 gap      | silence between the end of copy 1 and the start of copy 2 > `H["repetition_max_gap_s"]` = 1.5 s (the engine's repetition gap, `mrcd/constants.py`, `annotation/auto_labeler.py`)                                                                                                                                                                                                                                   | `uncertain`  |
| R3 seam     | the audio was chunked (duration > 8 s) and a chunk seam `s_k = k * (8 - 2) + 2` s (k >= 1; the point where `transcribe_chunked` hands over from chunk k-1 to chunk k) lies between the midpoint of the last token of copy 1 and the midpoint of the first token of copy 2, i.e. the two copies were decoded by different chunks (the documented seam-duplication risk, `DISAMBIGUATION_SPEC.md` section 9, row 17) | `uncertain`  |
| R4 polarity | `n == 1` and the token is in `textfeat.POLARITY` (si, no, ya, vale, claro, bueno, ok, bien, gracias, perfecto): emphatic reduplication of an answer/polarity item                                                                                                                                                                                                                                                  | `functional` |

Only `disfluent` candidates become repetition events of the profile (overlapping or touching disfluent token spans are
merged into one event). Demoted candidates are counted per rule in `provenance.decision_profile.classes.repetition`.

Declared limitation: DIHANA carries no function labels, so `functional` cannot be validated
(`DISAMBIGUATION_SPEC.md` section 10, item 4). DIHANA marks literal repetitions regardless of function, so R4 is EXPECTED to
cost recall against this reference (section 6). R4 stays in the profile because the architecture requires a functional
state; its cost is reported, not hidden, and no rule is added, removed or re-ordered after seeing results.

### 1.2 Filler (Plan E): `filled_pause` and `lexical_filler`

- `filled_pause` (non-lexical: eh, em, mm, hm, ah, ...) = the `filler_word` unit pipeline re-fitted on P1 (1.3). Its
  training target (DIHANA `f` marks: `f:e`, `f:m`, `f:a`, `f:sin_identificar`) contains only non-lexical filled pauses.
- `lexical_filler` (este, o sea, bueno, pues, tipo, a ver, digamos) = candidates are the lexical entries of the existing
  `constants.FILLER_LEXICON["es"]` (one- and two-word items; the lexicon is NOT extended); state by the existing engine
  rule for ambiguous lexical items (`annotation/auto_labeler.py`, `detect_fillers`): `disfluent` iff gap before >= 0.15 s or
  gap after >= 0.15 s or item duration >= 0.35 s or punctuation after in {",", "...", "…"} (a missing neighbour counts as a
  1.0 s gap, as in the auto-labeler); otherwise `uncertain` (polyfunctional item without hesitation evidence).
- In the service both are emitted as class `filler_word` with an additive event field `context.subclass`; only
  `disfluent` lexical candidates become events; uncertain ones are counted in provenance.

### 1.3 Unit pipelines re-fitted on P1 text, with no new freedom

`filler_word` and `prolongation` are re-fitted with the SAME feature blocks, fusion, model family and selection
machinery as the `o2026_11` final configurations: filler blocks rule+v0+ctxfill+sig+pause, fusion or_rule; prolongation
blocks v0+sig+lenx+pause+emb, fusion none; model = logistic regression with C chosen from the frozen grid
`space.MODELS` LR values {0.03, 0.3, 3.0} by the inner loop; threshold, OR-threshold, calibration and abstention band by
`search.outer_fold` / `choose_calibration` (inner K=3 speaker-grouped, outer = the frozen 5-fold speaker partition,
seed 13). Restricting to LR is required because only LR pipelines are exportable to plain JSON (`safe_pipeline`).
Features for P1: P1 words AND the engine features recomputed with P1 words by `DisfluencyEngine.unit_features` (the
service path), because the window probabilities depend on the linguistic branch. The shipped artifacts are the final fit
on all 5 056 rows (as for `o2026_11`), exported to JSON and sha256-pinned in source.

Same-procedure control: the identical restricted procedure run on P0 text and the cached P0 features (`p0_restricted`),
so that the P1 effect is separable from the restriction of the search.

## 2. Data (DIHANA folds 1-4 only)

- Pool: `mrcd.asr_recovery.common.load_pool()` (5 056 turns, 180 speakers, `ALLOWED_FOLDS == (1, 2, 3, 4)` asserted);
  outer partition `outer_folds(pool)`. Fold 5, AMERESCO, EPIC, CHM150 and the NOCANDO recordings are never read.
- P1 words: the existing 1 025 P1 decodes of the ASR-recovery frame are reused; the other 4 031 turns are decoded with the
  same code (`asr_recovery.decode.transcribe_chunked(..., "P1")`, CPU int8, `OMP_NUM_THREADS=4`, `ctranslate2` seed 13).
  P0 words: the existing cache (`results/mejoras_20261030/asr_cache/dihana`), P0 features: the existing shards.
- Cache (resumable, git-ignored): `results/plan_a_20261108/cache/` (one file per turn; finished turns are skipped).
- References: repetition = `pool[i]["ref"]["rep_literal"]` (human DIHANA `r:repeticion` subtype A; None = indeterminate,
  excluded; expected 214 positives / 4 775 negatives); filled pause = `filler_present` (all DIHANA `f` marks, 774
  positives); prolongation = `lengthening_present` (967); lexical filler = the turn has >= 1 DIHANA `d tipo="rellenar"`
  mark (discourse marker with a filling function; the negative coverage of `d` marks was not verified, declared).
- Inspected before writing (declared): counts of DIHANA `f` subtypes and the word lists of `d` marks in folds 1-4
  (d:rellenar is mostly "pues", then mira/mire/a ver/entonces/bueno/o sea), audio durations of the pool. Because the
  d:rellenar word list was seen, the lexical lexicon is frozen to the pre-existing `FILLER_LEXICON` (mira, mire, entonces
  are NOT added).

## 3. Systems and comparators (turn = unit; a turn is positive if the system emits >= 1 event of the class)

Repetition (all rule systems are fixed rules, no fitting): `V0` (engine argmax windows on cached P0 features, the current
default's repetition), `p0_rule` (generator on P0 words), `p1_rule` (generator on P1 words, no demotion),
`o2026_12` (generator on P1 + R1-R4, `disfluent` only), declared secondary `o2026_12_u` (R1-R3 only, no functional rule).
Filler: `o2026_11` (existing nested-CV predictions, `O_campaign/predictions_filler_word.csv`), `p0_restricted`,
`o2026_12` filled_pause (P1, cross-validated). Prolongation: the same three. Lexical filler: `o2026_12` disfluent state;
lenient reading disfluent + uncertain (= generator recall). All pipeline figures are outer-fold (cross-validated)
predictions; no figure is computed on rows a model was fitted on.

## 4. Hypotheses, metrics, statistics

Metrics: precision, recall, F1 per class, with TP/FP/FN; per outer fold table; confusion counts. Bootstrap: cluster =
speaker (dialogues are nested in speakers), 5 000 resamples, seed 13, percentile 95 % CI, paired deltas on the same
resamples, two-sided bootstrap p.

Primary family (Holm over 3):

- HA1: repetition F1 `o2026_12` - `V0` > 0 (CI lower > 0 and p_Holm < 0.05).
- HA2: repetition F1 `o2026_12` - `p0_rule` > 0.
- HA3: repetition F1 `o2026_12` - `p1_rule` > 0 (the demotion layer helps).

Demotion analysis (descriptive, with CIs): share of `p1_rule` candidates demoted, per rule; turns whose decision flips
1 -> 0 split into reference positives (recall lost) and negatives (FP removed); "demotion raises precision without losing
recall" is declared true iff Δprecision (`o2026_12` - `p1_rule`) has CI lower > 0 AND Δrecall >= -0.02 (point).

Gates (non-inferiority; not in the Holm family; same form as the frozen M2 gate of `mejora_servicio_20261102`):

- GF: filled_pause F1 `o2026_12` - `o2026_11` >= -0.02 and CI upper >= 0.
- GP: prolongation F1 `o2026_12` - `o2026_11` >= -0.02 and CI upper >= 0 ("must not degrade by more than 0.02").
- GL (latency, section 5): median wait after close <= 10 s.

Descriptive only: P1 vs `p0_restricted` deltas (ASR effect), `o2026_12_u`, lexical filler precision / recall / F1 against
d:rellenar and its lenient reading, the union filler_word (filled_pause OR lexical disfluent) against `filler_present`,
per-DIHANA-fold counts.

Absence is never 0: anything not measured is reported `not_available` with a reason (e.g. lexical-filler metrics outside
DIHANA, functional-state accuracy).

## 5. Latency (G4 procedure, unchanged)

`scripts/mejora_servicio/g4_benchmark.py` procedure: a real service process (`mrcd.service serve`, test port 8791,
`--incremental --device cpu --preload <ASR profiles> --warmup`, `OMP_NUM_THREADS=4`), the same 120 s recording of the S8
benchmark, real-time pace (2 s batches), 1 discarded cold round + 2 measured rounds per profile, median of the measured
rounds, offline RTF from the offline job. Profiles measured in the same process: `o2026_12` (preload p1 only) and
`o2026_11` as same-session reference. Background CPU load is logged before each round. Reported as measured; no claim
beyond this machine. Measured after all decoding has finished (no concurrent heavy job).

## 6. Expectations written before any run

| quantity                                      | expected range                                                                 |
| --------------------------------------------- | ------------------------------------------------------------------------------ |
| repetition F1 V0 (full pool, unweighted)      | [0.40, 0.60]                                                                   |
| repetition F1 p0_rule                         | [0.50, 0.65]                                                                   |
| repetition F1 p1_rule                         | [0.60, 0.75]                                                                   |
| share of p1_rule candidates demoted           | [0.03, 0.20], dominated by R4; R1-R3 together < 0.04                           |
| repetition F1 o2026_12 - p1_rule              | [-0.06, +0.02] (HA3 expected NOT supported: R4 removes DIHANA-marked "si, si") |
| repetition F1 o2026_12_u - p1_rule            | [-0.01, +0.02]                                                                 |
| HA1, HA2                                      | HA2 supported; HA1 likely supported                                            |
| filled_pause F1 o2026_12 (o2026_11 CV: 0.697) | [0.66, 0.73]; delta vs o2026_11 in [-0.03, +0.03]                              |
| prolongation F1 o2026_12 (o2026_11 CV: 0.619) | [0.59, 0.64]; delta in [-0.02, +0.01]; GP likely passes                        |
| lexical filler vs d:rellenar                  | precision [0.10, 0.40], lenient recall [0.50, 0.85]                            |
| G4 wait after close o2026_12                  | [8, 14] s; passing the 10 s limit is NOT assured                               |
| offline RTF o2026_12                          | [0.35, 0.50]                                                                   |

## 7. What this cannot show

One corpus (telephone, peninsular, train-information domain); literal repetition only; no human-judged function labels;
the P1 effect has an unknown cause (C1 control); weights of the ASR-recovery frame are not used (the full pool is scored
unweighted); turn-level scoring (no timing accuracy); scores are not calibrated probabilities; no confirmatory claim of
any kind. A failed hypothesis or gate is reported as a failure. `o2026_12` stays opt-in whatever the outcome; adopting it
as default would need a new protocol and an untouched evaluation cell.
