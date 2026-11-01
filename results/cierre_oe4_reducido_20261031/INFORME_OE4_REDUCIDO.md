# OE4 reduced: evidence package for a proposed reduction of scope (2026-10-31)

**Status of this document.** Technical evidence package prepared by the authors. It is a SELF-REVIEW product: written by one author, numbers transcribed from existing reports in `results/` (nothing was re-run for this document; heavy runs were forbidden while phase 3b measures CPU time), and not checked by an independent reviewer. Nothing here is signed, sent or approved.

## 0. Compliance statement (verbatim)

> **OE4 reduced: complete evidence package for the proposed scope, pending ratification by the advisor; it does NOT meet the original indicators of `OBJETIVOS.MD`.**

What that means, without softening:

- The scope reduction (section 1) is a **proposal**. The authors cannot state that the advisor approved it. The user decided on 2026-10-31 to drop the three classes without a compatible reference; that decision still requires the advisor's ratification (`APROBACION_ASESOR_OE4.md`, status PENDING).
- None of the original OE4 indicators is met (section 9). The measured F1 values are far below the 0.80 / 0.65 / 0.60 thresholds; several indicators cannot be measured at all under the "no new human annotation" restriction.
- Every number below is EXPLORATORY. No result is confirmatory. The checkpoint under test is a weakly supervised smoke-test checkpoint evaluated zero-shot on a telephone domain; nothing validates it as a trained model.
- Usability (SUS >= 70 with >= 10 real users) was **not measured**. It requires an operable web application, consent and real participants. No local SUS protocol exists in the workspace (searched `mrcd/`, `<plan-package>/`, `paper/`, `Documentos/`; the only SUS text found is the target itself in OBJETIVOS/INFORME_OBJETIVO_4/TASKS).

## 1. Reduced scope (PROPOSAL, requires advisor ratification)

User decision of 2026-10-31: classes without a compatible reference are removed from the evaluated scope of OE4, and OE4 is completed with what exists. The code is NOT deleted.

| Engine class | Treatment in reduced OE4 | What the reference actually is | What is NOT claimed |
|---|---|---|---|
| `filler_word` | EVALUATED | Filled pause only (DIHANA `f`; EPIC FPAUSE). Discourse markers such as "bueno", "este" are tagged `d` in DIHANA, not `f` | Boundaries; DM as a separate class; function |
| `prolongation` | EVALUATED as observable lengthening | DIHANA `a` mark, AMERESCO `<alargamiento/>` | That the lengthening is disfluent (not demonstrated) |
| `repetition` | EVALUATED, literal repetition | DIHANA `r:repeticion` (sub-type A literal = 78% of marks); text-level rules | Rhetorical vs disfluent repetition; cut-word restarts (sub-types B-D are only partly captured) |
| `revision` | APPROXIMATION, NOT CONFIRMED | DIHANA `r:sustitucion/insercion/borrado`, 58% silent edits; `utt74` is one exact example | Any accuracy claim; F1 is shown only as a descriptive screening number |
| `block` | EXCLUDED | No reference. `TRUNCATION`, `l:cortada`, `b`, `<x->`, `CUT`, `-` are truncations or abandonments, not blocks | Anything |
| `rhetorical_pause` | EXCLUDED | No reference. Corpora mark pause presence (DIHANA `p`, EPIC SPAUSE, ES-Port `silence`), never function or duration-validated boundaries | Anything |
| `neutral_pause` | EXCLUDED | Same | Anything |

The three excluded classes stay in the engine as: **out of the evaluated scope, no reference, technical output not validated.** They are never averaged as F1 = 0 and their emissions must not be shown as validated results. Section 7.3 documents their measured technical sensitivity (not validity).

Separate codes of `ALCANCE-MOTOR.MD` that are not separable in the executable taxonomy and are therefore not reported individually: FP and DM (both inside `filler_word`), FS and REP-A (inside `revision`), SP (no one-to-one mapping). See the taxonomy equivalence in `results/cierre_oe1_oe3_20261031/APROBACION_ASESOR_OE1.md`, appendix A.

## 2. Tasks and units (what is actually measured)

- **S task (presence per unit):** is the class present in this turn / segment / sentence? Input is audio plus the engine's OWN ASR. Prediction = at least one 3 s window (0.5 s stride) whose argmax is the class.
- **T task (text):** text-only rules on the human transcript with marks removed. Not comparable with S numbers and not engine input.
- **Units:** DIHANA user turns (62% shorter than 3 s, zero-padded to one window), AMERESCO speech segments of 3-8 s, EPIC sentences, disfluentES sentences, ES-Port utterances.
- **Not measured:** temporal detection (D). No resource on disk has human event boundaries (AMERESCO has human segment times only; EPIC times are automatic). IoU against humans does not exist. The engine's 0.5 s "decision band" is not an event boundary.
- **Absence of a label is never a negative** unless the corpus documents exhaustive marking (DIHANA manual double-annotated; EPIC for filled pauses; disfluentES labelled records). AMERESCO negative coverage is unverified.

## 3. Corpora used (licence, coverage, speakers)

Source: `results/auditoria_corpus_20261030/MATRIZ_CORPUS.md` (counts recomputed from source files there), `results/oe4_fuentes_20261031/FUENTES_OE4.md`.

| Corpus | Used for | Units / support | Speakers or groups | Licence | Negative coverage / limits |
|---|---|---|---|---|---|
| DIHANA | filler, prolongation, repetition, revision (S and T) | 6,279 user turns; folds 1-4: 5,056 units; fold 5: 1,223 | 225 speakers, 5 official speaker-disjoint folds of 45 | CC BY 4.0 | Documented (double annotation `eti1+eti2`, inter-annotator agreement NOT reported). 8 kHz telephone, Wizard-of-Oz train information, peninsular Spanish, no event times, no video |
| AMERESCO | prolongation only | 98 segments (3-8 s pool), 37 positive | 4 conversations, 11 speaker tiers (9 with positives); identity across conversations not established | CC BY-NC-SA 4.0 | Unknown. Positives sit in 2 conversations (33 of 37) |
| EPIC v2 | filler only (development set) | 136 sentences, 68 positive (balanced 50/50 by construction) | 29 speakers | CC BY-NC 4.0 | Positives concentrated; negatives almost all carry silent-pause notation; used in earlier development, NOT a test |
| disfluentES | repetition (T), filler (T) | 4,506 sentences; 633 repetition, 86 filler positives | 329-450 child speakers, text only | CC BY 4.0 | Children reading aloud; no audio; the filler rule that works needs the bare token `e`, found by inspecting this same corpus (optimistic) |
| ES-Port | repetition + false start jointly (T) | 99,298 utterances | 1,170 dialogues (speakers not linkable) | CC BY-SA 3.0 Spain | Joint label, not repetition alone |
| DIME, ESLORA | engineering checks only (importers, throughput, ASR smoke) | 20 dialogues + utt1/utt74; public text | - | per catalogue | DIME has no class labels (XLS are dialogue acts); repair-label directory returned HTTP 403; ESLORA text has no audio or times |
| CHM150, SEP-28k, FluencyBank | not used. SEP-28k/FluencyBank label CSVs were downloaded as external taxonomy contrast (English, stuttering context) | - | - | CHM150: licence conflict unresolved; SEP-28k/FluencyBank labels CC BY-NC 4.0 | Not mixed with Spanish metrics |
| PerLA, PUCP, SOPHIAS | nothing retrieved (auth wall, challenge, DUA pending) | - | - | - | - |

The exclusion of an entire family of corpora is not a finding about the engine; it is a data-access limit (section 11).

## 4. Protocol

- **Partitions by speaker (group).** DIHANA official folds are speaker-disjoint. Folds 1-3 = development, fold 4 = tuning/selection, fold 5 = reserved test as declared in `results/baselines_20261030/BASELINES.md`. Text corpora: `sha256(group) % 5` buckets (0-2 dev, 3 tune, 4 test), group = speaker (disfluentES) or dialogue (ES-Port). AMERESCO partitions by conversation (4 groups, so a bootstrap over groups is not meaningful; per-conversation tables are the honest view).
- **Fold 5 is HISTORICAL, not an untouched test.** It was scored in `baselines_20261030` (checkpoint and comparators) and again once in `mejoras_20261030/final` (versions V0..Vac). No re-selection was made after seeing it, but it cannot count as an independent confirmatory test. The later repetition/revision experiment (`ejecucion_material4_20261030/experimentos`) and the ASR recovery experiment use folds 1-4 only; `ALLOWED_FOLDS == (1,2,3,4)` is pinned by tests.
- **Exploratory status.** Comparators and thresholds were fixed a priori in the baselines (zero tuning); `mejoras` has a written pre-registration (`PREREGISTRATION.md`), but its freeze order is evidenced by file modification times only (`results/` is git-ignored). The rep/rev experiment has a hashed protocol (`PROTOCOL.sha256.json`) and annexes. None of this makes the results confirmatory: one test fold of 45 telephone speakers, one corpus per class family, unknown inter-annotator agreement.
- **Uncertainty.** 95% confidence intervals by cluster bootstrap over speakers (2,000 resamples in baselines; 5,000 paired draws in the rep/rev experiment, whose p floor is therefore p < 0.0004). Rows with fewer than 30 positives are flagged UNSTABLE.
- **Multiplicity.** The rep/rev family is pre-declared (ctx_LR minus a one-line rule, 4 paired tests) and corrected by Holm. The `mejoras` comparison family (about 25 comparisons) was NOT corrected; its two "demonstrated" gains (Vc filler lower CI bound +0.011, Vc prolongation +0.009) would not survive Holm and are reported as not demonstrated here.
- **Zero new human annotation.** No campaign, no double annotation, no automatic judge used as a human reference.

## 5. Results by class x comparator x corpus

Cell format TP/FP/FN, P, R, F1 [95% CI by speaker]. "Support" = positive units. All EXPLORATORY.

### 5.1 DIHANA fold 5 (historical test), S task, audio + engine ASR, 1,223 turns, 45 speakers
Source: `results/mejoras_20261030/final/FINAL_TABLES.md` (frozen caches, OMP=4 set before numpy). V0 = checkpoint with default recipe; Vc = trained head on folds 1-3, selection on fold 4 (experimental, opt-in).

| Class (support) | Comparator | TP/FP/FN | P | R | F1 [95% CI] |
|---|---|---|---|---|---|
| filler_word (145) | V0 | 64/44/81 | 0.59 | 0.44 | 0.51 [0.39, 0.59] |
| | Vc | 90/77/55 | 0.54 | 0.62 | 0.58 [0.46, 0.66] |
| | rule_asr_lexical (one-line lexicon on ASR) | 61/5/84 | 0.92 | 0.42 | 0.58 [0.47, 0.66] |
| | existing auto-labeler | 65/22/80 | 0.75 | 0.45 | 0.56 [0.45, 0.64] |
| | always_present | 145/1078/0 | 0.12 | 1.00 | 0.21 [0.15, 0.28] |
| prolongation (211) | V0 | 95/46/116 | 0.67 | 0.45 | 0.54 [0.44, 0.62] |
| | Vc | 157/130/54 | 0.55 | 0.74 | 0.63 [0.56, 0.69] |
| | existing auto-labeler | 26/6/185 | 0.81 | 0.12 | 0.21 [0.14, 0.29] |
| | always_present | 211/1012/0 | 0.17 | 1.00 | 0.29 [0.25, 0.34] |
| repetition, DIHANA `r:repeticion` mark (60) | V0 | 17/20/43 | 0.46 | 0.28 | 0.35 [0.21, 0.48] |
| | Vc | 14/14/46 | 0.50 | 0.23 | 0.32 [0.18, 0.43] |
| | rule_asr_lexical | 22/18/38 | 0.55 | 0.37 | 0.44 [0.28, 0.57] |
| | existing auto-labeler | 19/17/41 | 0.53 | 0.32 | 0.40 [0.26, 0.52] |
| revision (36), APPROXIMATION | V0 | 2/5/34 | 0.29 | 0.06 | 0.09 [0.00, 0.20] |
| | rule_asr_lexical | 2/1/34 | 0.67 | 0.06 | 0.10 [0.00, 0.21] |

Reading, with the independent reviewer's limits (`results/revision_20261030/REVISION.md`):
- **filler_word:** the checkpoint does not beat a one-line lexical rule on the ASR text (Vc minus rule: -0.001 [-0.072, +0.069]); the checkpoint's only difference is more false positives. The ASR caps recall near 0.5: only about 49% of filler-positive turns contain a vocalic filler in Whisper's text.
- **prolongation:** the only class where the trained model clearly exceeds simple comparators. Vc minus V0 is +0.091 [+0.009, +0.179] (uncorrected; the gain is recall at the cost of precision, FP 46 to 130), and the +0.336 over `always_present` says little because that comparator is weak. V0 F1 varies 0.45-0.54 across folds, comparable to the claimed gain. The label is a lengthening mark, not necessarily a disfluency.
- **repetition:** every model version is worse than the ASR lexical rule (dF1 -0.08 to -0.12, CI below 0). 74% of positive turns show an adjacent repetition in the human transcript but only 54% of those keep it in Whisper's output.
- **revision:** no comparator detects it. Only conclusion: not detectable with this input.
- Dev/tune gains did not carry to test (for example Vb prolongation +0.14 dev, +0.007 test).

The earlier baseline run on a 33% development sample and the pre-OMP-pin ASR gave nearly identical test numbers (`baselines_20261030/BASELINES.md`: V0 checkpoint filler 0.53 [0.40, 0.62], prolongation 0.53 [0.43, 0.62], repetition 0.34 [0.19, 0.47]); the differences (0.51 vs 0.53 and so on) come from running ASR at OMP=4 on unpadded audio.

### 5.2 DIHANA folds 1-4, speaker-grouped nested CV, text models on ASR text (condition B) or reference text (condition A)
Source: `results/ejecucion_material4_20261030/experimentos/TABLES.md`. 180 speakers. Condition A reads the human transcript with marks and depends on the comma convention, so it is NOT engine performance and is shown only to bound the loss caused by ASR. Condition B uses the frozen Whisper-small text.

| Task (support) | Cond | Comparator | TP/FP/FN | P | R | F1 [95% CI by speaker] |
|---|---|---|---|---|---|---|
| rep_literal (214 turns, 84 speakers) | B | rule | 108/77/106 | 0.58 | 0.50 | 0.541 [0.48, 0.59] |
| | B | V0 | 93/75/121 | 0.55 | 0.43 | 0.487 [0.42, 0.54] |
| | B | ctx_LR (contextual logistic model) | 109/83/105 | 0.57 | 0.51 | 0.537 [0.48, 0.59] |
| | A | rule / ctx_LR | 211/32/3 ; 207/6/7 | 0.87 / 0.97 | 0.99 / 0.97 | 0.923 / 0.970 (reference text; not engine input) |
| rep_full, all repetition sub-types (281, 114 speakers) | B | rule / V0 / ctx_LR | 119/77/162 ; 100/75/181 ; 122/99/159 | 0.61 / 0.57 / 0.55 | 0.42 / 0.36 / 0.43 | 0.499 / 0.439 / 0.486 |
| revision (161, 91 speakers) | B | rule | 15/30/146 | 0.33 | 0.09 | 0.146 [0.09, 0.21] |
| | B | V0 | 11/18/150 | 0.38 | 0.07 | 0.116 [0.06, 0.17] |
| | B | ctx_LR | 71/265/90 | 0.21 | 0.44 | 0.286 [0.23, 0.34] |

Paired ctx_LR minus rule (Holm over 4): rep_literal B -0.004 [-0.036, 0.028], p_Holm 0.81, **no difference**; rep_literal A +0.046, revision A +0.137 and B +0.140 (p_Holm 0.0072 / 0.0072 / 0.0016). The revision gain is a screening gain: a length-only model reaches 0.232, so the contextual excess is +0.053 [0.017, 0.091], post hoc and unadjusted. The contextual model gives **no improvement over a one-line rule for literal repetition on ASR text**; the rule also beats V0 (+0.054 [0.011, 0.097], descriptive). Calibration gain: not demonstrated (equal-width and quantile bins; all CIs contain 0). The historical fold-5 numbers for the earlier-study repetition and revision marks are copied unchanged in the same file.

Exploratory ASR-prompt experiment (`results/asr_recovery_20261031/EXPERIMENTO_ASR.md`, same folds): a repetition-bearing prompt (P1) keeps more repetitions in the transcript (survival 105 to 144 of 214) and raises weighted F1 of ctx_LR from 0.581 to 0.690, but a fluent control prompt of equal length and domain (C1) did not reproduce the gain (0.304) and gives the same WER improvement (0.129 vs 0.130), so WER is not a recovery signal. What feature of P1 matters is not proven. P1 is opt-in and experimental, one corpus, one model size. Revisions: not demonstrated (p_Holm 0.104). It is not part of the default engine.

### 5.3 Other corpora
| Corpus / class | Setting | Result | Reading |
|---|---|---|---|
| AMERESCO prolongation (98 segments, 37 pos, 4 conversations) | S, V0 checkpoint | 8/8/29 (TN 53), P 0.50, R 0.22, F1 0.30; existing rule 0.10; always_present 0.55 | `always_present` beats the checkpoint. Pilot-32 subset 0.43 vs the other 66 segments 0.20: the pilot figure was optimistic. No meaningful CI with 4 groups. Vc 0.56 vs always_present 0.55 (McNemar vs V0 p = 0.14): recall inflation toward the 38% prevalence |
| EPIC filler (136 development sentences, 68 pos, 29 speakers) | S | V0 0.72-0.74; Vc 0.77 (+0.046 [-0.019, +0.133]); rule_asr 0.66-0.69; always_present 0.67 | Development set, balanced by construction: no demonstrated gain and not a test |
| disfluentES repetition (633 pos / 4,506 sentences, 329 speakers) | T | adjacent-repetition rule P 0.93, R 0.56, F1 0.70 [0.66, 0.73] (test bucket 0.73 [0.65, 0.79]) | Reading task by children, human text; not comparable with S |
| disfluentES filler | T | a-priori lexicon F1 0.00; lexicon + bare `e` 0.75 (86 pos) | The second rule was informed by inspecting the same corpus: optimistic, not independent |
| ES-Port repetition + false start | T | rule P 0.59, R 0.51, F1 0.55 over 99,298 utterances; always_present 0.19 | Joint label |

Error analysis (`baselines_20261030/BASELINES.md` section 5): Whisper erases or normalises repetitions and fillers; the checkpoint confuses neighbouring phenomena that DIHANA separates (prolongation FPs co-occur with silent-pause mark `p` in 72%; filler FPs with discourse-marker tag `d` in 24%); domain shift (8 kHz telephone vs the YouTube material of the weak supervision).

## 6. Comparison with baselines (summary)

- Against trivial baselines (`always_present`, `always_absent`): the engine is clearly above them only for prolongation on DIHANA (and even there `always_present` equals or beats it on AMERESCO).
- Against a one-line rule on ASR text: filler no difference (Vc), repetition rule is better than every audio/text model, revision all near 0.
- Against the contextual text model (ctx_LR): no gain for literal repetition on ASR text; a screening gain for revision that a length-only model largely explains.
- `rule_existing_autolabeler` is the teacher of the weak labels used to train the base checkpoint; its agreement with the checkpoint is not independent evidence.
- The checkpoint `audio_text_v2.pt` (duration ratio + manual post-softmax thresholds) was rejected (8/60 vs 12/60) and must not be deployed (`results/rejected/README.md`).

## 7. Robustness

### 7.1 Determinism
- Same input, same process: repeat max probability difference 3.5e-10; torch threads 1/2/4 max difference 2.4e-7; 0 argmax changes (`mejoras_20261030/MEJORAS.md`).
- ASR (Whisper small, CPU int8, beam 5, T=0): bit-identical in a fresh process (84/84 units, `asr_cache/determinism_audit.json`) **only when `OMP_NUM_THREADS=4` is set before numpy is imported and the machine is unloaded**. Setting it after `import numpy` gave different text (2 of 32 pilot clips matched; mechanism not profiled). Two concurrent slow test suites produced 6 failures each. Cross-environment determinism is not established.

### 7.2 Parity offline and live
- **Offline:** the current `Engine.analyze` equals the snapshot path (max abs diff 0.0, 0 argmax changes, 820 + 192 windows, `results/paridad_20261030/PARIDAD.md`). The default Engine has no terminal window; with `terminal_window=True` the pilot decisions 5/3/11/13 are reproduced (n = 32, uncalibrated).
- **Live (limits):** the old live path differs from offline in 6 of 192 windows (3.1%, max probability difference 0.951). The stateful tracker with the offline pitch centre injected is exact (622 windows, 0 differences); with the causal centre it is NOT equivalent (20 of 622 argmax changes, 3.2%). On 12 new synthetic 10.5 min sessions (exact frame criterion, Clopper-Pearson 95%) `live_cfg_v2` diverges in 7/12 sessions (58% [27.7, 84.8]) and `live_cfg_v3` in 1/12 (8% [0.2, 38.5]); v3 is not the default. Sessions are synthetic Spanish stress sessions; no real microphone, no streaming ASR equivalence (`results/ejecucion_material4_20261030/INFORME_FINAL.md`, sections (e) and round-5/6/7 addenda). The earlier statement "exact" was withdrawn.
- **Streaming ASR parity and WER: pending 3b.** Predictions exist under `results/cierre_oe1_oe3_20261031/3b/asr/pred/` (offline and chunk-aligned streaming, profiles p0/p1, several chunk periods), but the WER/divergence report had not been generated when this document was written. The numbers will appear in the 3b ASR report next to those predictions. No streaming-ASR figure is claimed here.

### 7.3 Sensitivity, including the excluded classes
- Acoustic perturbation (words fixed): window argmax agreement 0.94 (SNR 30 dB), 0.90 (20 dB), 0.95 (gain x0.5), 0.93 (x2), 0.85 (x0.25).
- **Padding invariance fails for the pause classes.** On 71 short DIHANA turns, zero vs noise-floor padding flips `rhetorical_pause` emission in 46% and `neutral_pause` in 42%; extra zero padding (3 to 4.5 s) flips 28% and 46%. Block is stable (4%).
- 240 DIHANA turns (folds 1-3) with one variable changed (`results/ejecucion_material4_20261030/pausas/PAUSAS.md`): units emitting `rhetorical_pause` 100 (zero padding) vs 42 (noise padding) vs 179 (zero +1.5 s); `neutral_pause` 91 / 124 / 185; `block` 50 / 53 / 65. Emission depends on turn duration and padding (Spearman with duration: block +0.35, rhetorical -0.43, neutral +0.21). This is technical sensitivity only: it says nothing about whether an emission is correct.

## 8. Cost of processing

Source: `results/cierre_oe1_oe3_20261031/3b/cost/COST_REPORT_tables.md` and `video_cost.json` (phase 3b, CPU only, Intel i7-13650HX 14 cores / 20 threads, 23.7 GB RAM, Windows 11, torch 2.9.1+cpu, `OMP_NUM_THREADS=4`; background CPU 2.98-7.62% before each round; round 0 discarded, rounds 1-3 as median [min-max]; 11 files analysable, 2 rejected as `audio_too_short`).

- Cold start (checkpoint + Whisper small + Vc head): 1.47 s wall, 3.17 s CPU; RSS after load 562 MB; peak working set 1,145 MB; peak RSS per run median 795 MB.
- Default path (ASR p0 + V0 head), real-time factor (processing time over audio duration): about 0.45-0.51 for files of 17-60 s; 1.24 for the 13.5 s file, the first file of every round (its ASR p0 took 16.2 s against 7.6 s for the 17 s file; cause not isolated). Service end-to-end (`MrcdAnalyzer.analyze_bytes`): 0.47-0.52 for files of 17-60 s, 1.25 for the 13.5 s file.
- All stages with both ASR profiles: RTF about 0.93-1.07 for files of 17-60 s. Per-stage median RTF: ASR p0 0.44, ASR p1 0.53, acoustic 0.039, linguistic and decision about 0, heads 0.0015, report 0.0002. ASR is more than 90% of the cost.
- Staged events equal `engine.analyze` on all 11 files; 0 schema errors.
- Video (landmarks only): median 6.6 ms/frame (p95 about 12-13 ms) at 10 fps 480x480; face found in 98.6-100% of frames on 4 clips; peak RSS 370 MB. This is decode + face landmark cost only; no fusion, speaker identity or synchrony was measured (`scope_note` of the JSON).
- Streaming (live) cost: partial at the time of writing (3 of 6 seeds logged; tick median 0.09-0.24 s per 0.5 s tick across the logged runs, all configs). Final figures will be in `results/cierre_oe1_oe3_20261031/3b/live/runs.jsonl` and its report. **Pending 3b.**
- Not measured: GPU, other hardware, concurrency, the Speechmatics ASR used by the web application.
- For context, the 2026-10-20 Capstone verification measured 22.9 and 21.0 min of wall time per hour of audio (their own goal was <= 20) and RTF 0.38 and 0.35 on 12 recordings (`results/sync_verification/REPORT.md`). Those figures are a different run and are not reconciled with the 3b table here.

## 9. Modality ablation: what exists and what does not

- **Exists, but against a proxy, not a human.** `results/ablation_table.md` trains on weak `auto` labels and tests on `gold_llm` labels from a multimodal VLM judge (gpt-4o; heuristic-vs-judge kappa 0.422; human validation CSV has no valid labels). 18 units, 18 speakers, k = 2 folds, bootstrap B = 2. Macro-F1: audio_only 0.126, audio_text 0.281, trimodal 0.286, kinesic_only 0.058; trimodal minus audio_text [-0.016, +0.003]. With 18 units and B = 2 the intervals are not credible (several do not contain their own point estimate). It is retained for traceability and is **not** evidence under the reduced OE4.
- **Audio vs audio + text on a human reference:** only indirectly (S task audio+ASR vs T task on human transcript). No controlled ablation of the engine's branches on DIHANA was run; `BASELINES.md` states "no ablation of audio vs text branches was run here".
- **Video: NOT EVALUABLE.** The clip MP4s are face crops at 10 fps without an audio track, so audio-video synchrony (SyncNet) was not run; there is no active-speaker or identity verifier; no compatible human reference accompanies video except EPIC pause marks that have no verified synchrony. The visual policy therefore falls back to exact audio/text outputs (136 exact backups). No visual benefit is accredited and none is inferred. The visual channel must never infer psychological states.
- **Absence is reported with a reason, never as 0:** a modality that was not measured is "not available" with its reason. Known deviation: before the 2026-10-31 service fix `rate_per_minute` returned 0.0 for audio <= 1 s; it now returns None (commit 51cbaab).

## 10. Uncertainty and abstention

What was done: a three-state decision module (`mrcd/decision.py`: present / absent / uncertain; NaN score maps to uncertain; per-class thresholds by speaker-grouped CV), a selective-policy prototype in the earlier plan package (null abstention; nested grouped validation on 102 segments / 16 ids: benefit insufficient for promotion), reliability tables and Platt/quantile calibration of ctx_LR (ECE about 0.002-0.009 for rare positives; gain not demonstrated), and an engine contract that labels every score `max_window_softmax_score`, **not calibrated** (deprecated alias retained).

What was NOT done: no calibrated abstention in the engine; no calibration set; no coverage/risk curve. The experiment that added abstention (`audio_text_v2`) was rejected. Requirement RF-09 ("uncertain" state with calibrated abstention) is **not implemented** (`REQUISITOS_MOTOR_OE1.md`). The pause and block classes have no threshold at all.

## 11. Limits that bound every statement above

One reference corpus family for the audio classes (telephone, peninsular Spanish, Wizard-of-Oz, 8 kHz); no Peruvian corpus; inter-annotator agreement unknown or not reported; only one test fold (45 speakers) and it is historical; AMERESCO has 4 groups; EPIC is development; revision labels are 58% silent edits; ASR (Whisper small, int8) is the cap for filler and repetition; the checkpoint is a weak-label smoke test; probabilities are uncalibrated; live evidence is synthetic; dialectal fairness is not measured; protocol freeze order is evidenced by file times; `results/` is git-ignored and only force-added reports are versioned. Pending access: DIME repair labels (403), ESLORA audio (not requested; licences conflict), PerLA, PUCP, SOPHIAS (DUA), CHM150 (licence conflict), COE San Millán (permission). Independent review of the late corrections (rounds 5-7) is pending; this document has not been independently reviewed.

## 12. Matrix: original indicator -> reduced indicator -> evidence -> state

"Cumplido" in the last column means the reduced indicator was measured and reported as proposed. It never means a quality target was reached. The reduced indicators carry **no numerical acceptance threshold**.

| Original indicator (`OBJETIVOS.MD`, unchanged) | State against the original | Proposed reduced indicator (to be ratified) | Evidence | State of the reduced one |
|---|---|---|---|---|
| F1 >= 0.80 for FP, SP, REP, FS, REP-A, PRO | **Not met.** Best audio+ASR values: filler 0.51 (V0) / 0.58 (Vc, rule), 0.22 to 0.29 below; prolongation 0.54 / 0.63, 0.26 / 0.17 below; repetition 0.35 (fold 5), 0.54 (folds 1-4, B), 0.26 to 0.45 below; revision 0.09-0.29, 0.51 to 0.71 below. Upper CI bounds are 0.59-0.69 for filler/prolongation, still below 0.80. Values >= 0.80 appear only on reference text (condition A) or human-transcript rules (0.78 [0.68, 0.86]), which are not engine input | Per-class metrics (support, speakers, TP/FP/FN, P/R/F1, speaker-bootstrap CI) for filler_word, prolongation, literal repetition; revision descriptive only | Sections 5.1-5.3; `TABLES.md`, `FINAL_TABLES.md`, `BASELINES.md` | Cumplido (reported; exploratory) |
| F1 >= 0.65 for DM | **Not measurable.** DM is not a separate executable class (inside `filler_word`); DIHANA tags discourse markers `d`, not mapped | None (DM not reported separately) | `MATRIZ_CORPUS.md` | No medible |
| F1_context >= 0.60 on the ambiguous subset | **Not measurable.** No ambiguous subset with compatible human labels. Closest evidence: ctx_LR vs rule for literal repetition, no difference | Contextual model vs one-line rule, paired, Holm | Section 5.2 | Cumplido (reported); result: no gain for repetition, screening gain for revision |
| Inter-annotator agreement (kappa/alpha) and temporal IoU >= 0.70 vs double human annotation | **Not measurable.** Requires new annotation (excluded) and human event boundaries (none on disk). DIHANA is double annotated but agreement is not reported | None. Reported instead: decision band 0.5 s labelled as such; no IoU | `MATRIZ_CORPUS.md` | No medible |
| Coverage >= 0.90 with risk <= 0.10 in the "uncertain" state | **Not met / not measurable.** No calibrated abstention exists | Report what was done on abstention and state that calibration does not exist | Section 10 | Cumplido (documented as absent) |
| SUS >= 70 with >= 10 real users | **Not measured.** Needs an operable app, consent and participants; no local protocol found | Defer to future work or keep as an independent objective (advisor decides, question P5 of the sheet) | Section 0 | No medible |
| (no original counterpart) Reduced OE4 text: comparison with baselines and ablations of the modalities | - | Comparison with baselines on the same units; modality ablation where a reference exists | Sections 5-6; ablation vs proxy only (9) | Parcial: baselines cumplido; ablation of audio/text vs human reference no medible; video no medible |
| (no original counterpart) Robustness | - | Determinism, offline parity, live parity with its limits, sensitivity | Section 7 | Cumplido (measured); live parity is NOT exact; ASR streaming parity pending 3b |
| (no original counterpart) Cost of processing | - | Per-stage time, RTF, memory on stated hardware | Section 8 | Cumplido for offline; streaming and ASR-streaming pending 3b |
| Excluded classes `block`, `rhetorical_pause`, `neutral_pause` (inside the F1 indicators) | No reference | Declared out of evaluated scope: "no reference, technical output not validated" | Sections 1 and 7.3 | No medible (by design of the reduction) |

## 13. Criteria proposed for ratification (PROPOSED, to be decided with the advisor; the authors set no thresholds)

1. A class may be called "evaluated" only if the report gives support, number of speakers, TP/FP/FN, P/R/F1 with a speaker-bootstrap interval and at least one non-trivial comparator.
2. A class may be called "improved over a comparator" only if the paired interval excludes 0 after the correction declared for its family; otherwise it is "not demonstrated".
3. Nothing is called "validated" while the only test fold is historical and the reference is a single telephone corpus.
4. The three excluded classes appear in the product only as "technical output not validated".

## 14. Reproduction (read-only for this document)

Reports and raw JSON live under `results/` (git-ignored; the PDF-ready approval sheet and this report are force-added). Re-run commands are in `results/ejecucion_material4_20261030/INFORME_FINAL.md` section (h), `results/baselines_20261030/BASELINES.md` and `results/cierre_oe1_oe3_20261031/3b/` (`scripts/cierre3b/`). Slow suites must not run concurrently with another job.

## 15. Sources read for this document

`Documentos/OBJETIVOS.MD`; `results/ejecucion_material4_20261030/{INFORME_FINAL.md, experimentos/TABLES.md, pausas/PAUSAS.md}`; `results/baselines_20261030/BASELINES.md`; `results/mejoras_20261030/{MEJORAS.md, final/FINAL_TABLES.md}`; `results/asr_recovery_20261031/{EXPERIMENTO_ASR.md, PERFIL_P1.md}`; `results/revision*_*/` (findings as quoted in the addenda); `results/paridad_20261030/PARIDAD.md`; `results/paridad_live_20261030/PARIDAD_LIVE.md`; `results/auditoria_corpus_20261030/MATRIZ_CORPUS.md`; `<plan-package>/MRCD_Plan_Avance_20261030/docs/MATRIZ_OE1_OE4.md`; `results/ablation_table.md`, `results/judge_agreement.md`, `results/sync_verification/REPORT.md`, `results/rejected/README.md`; `results/cierre_oe1_oe3_20261031/{APROBACION_ASESOR_OE1.md, REQUISITOS_MOTOR_OE1.md, 3b/cost/*, 3b/live/*, 3b/logs/*}`; `results/oe4_fuentes_20261031/FUENTES_OE4.md`.

---

## Addendum 2026-11-01 (S3; the text above is unchanged)

Self-reviewed by the author; no independent review. All numbers EXPLORATORY (DIHANA folds 1-4, speaker-grouped nested CV, Holm per declared family; protocol frozen before measuring, `results/mejora_objetivos_20261101/S3/PROTOCOL.md`, details and tables in `S3/README_S3.md`). Nothing becomes confirmed and no original OE4 indicator is met.

- **Scope.** SOPHIAS is discarded as a source. The evaluated classes are `filler_word`, `prolongation` (observable lengthening, no disfluent-function claim) and `repetition` (literal). `revision` is an exploratory screening approximation, **not evaluated** (no S3 number is a revision accuracy). `block`, `rhetorical_pause`, `neutral_pause` are out of scope; the service now hides them by default (contract rev 1.2, marked `out_of_scope_unvalidated` when explicitly requested, never counted in totals or rates).
- **Repetition on P1 text.** `ctx_LR` retrained on P1 text (weighted frame, 1019 turns, 214 positives, 84 speakers): weighted F1 0.699 [0.620, 0.768] vs rule on P1 text 0.682 (delta +0.017, not demonstrated), vs V0 0.516 (+0.183, supported after Holm), vs ctx_LR on P0 text 0.581 (+0.118, supported after Holm) and vs the unretrained P0 model on P1 text 0.690 (+0.009, not demonstrated). The gain is the P1 text, not the retraining. Integrated only as the opt-in experimental route `rep_ctx_p1`.
- **Fusion rule + model.** No gain: for `filler_word` the learned OR equals the lexical rule (F1 0.661 vs 0.661) and Vc-based fusion on fold 4 is worse than frozen Vc; for repetition fusion does not exceed A1 or the rule. Not integrated.
- **Calibration and abstention.** Cross-fitted Platt/isotonic reduce ECE (quantile bins) of the V0 scores inside this pool (e.g. filler 0.084 to 0.017) but not of the learned text scores; the optional three-state decision in the service is experimental (`calibration_claim: "none"`). Coverage 0.89-0.97 with error among accepted 0.02-0.15; all-cases F1 with uncertain counted as absent is lower than without abstention (repetition 0.222 vs 0.496 for V0). Exploratory result on DIHANA (folds 1-4, 5,056 user turns, 180 speakers, speaker-grouped nested cross-validation; labels are corpus marks, not the double human annotation the original indicator requires). Coverage = share of turns on which the engine emits present/absent; risk = error rate (false positives and false negatives) among those emitted. For `filler_word`, coverage 0.932 (95% speaker-bootstrap CI 0.923-0.940) and risk 0.097 (0.083-0.110): the original thresholds (coverage >= 0.90, risk <= 0.10) are met at the point estimate, but the upper bound of the risk interval exceeds 0.10, so compliance is not demonstrated. For `repetition`, coverage 0.962 (0.954-0.970) and risk 0.029 (0.022-0.036), but the risk criterion is not informative there (always answering 'absent' gives risk 0.043). The F1 over all cases falls (filler 0.566 to 0.480; repetition 0.496 to 0.222). Not a confirmation of the OE4 indicator.
- **Prolongation.** Fusing V0/Vc with observable-lengthening and pause features does not reduce pause-associated false positives and does not change F1 demonstrably.
