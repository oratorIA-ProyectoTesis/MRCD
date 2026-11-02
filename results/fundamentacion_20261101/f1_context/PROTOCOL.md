# F1_context on ambiguous Spanish forms: frozen protocol (2026-11-01)

Status: EXPLORATORY. Frozen before any model, rule or metric was computed. Only the Step-0 reference audit (counts, no predictions) had been seen when this file was written; that is disclosed, and it influenced the scope choices in section 3 (declared as such).
Self-review by the executing agent; no independent review. DIHANA folds 1-4 only (telephone, peninsular Spanish, train-information Wizard-of-Oz). Fold 5 is never opened. No existing default (V0, P0, `live_cfg_v2`, service, extractors, optimize) is changed.

## 0. Step 0: is the reference valid? Decision

Evidence (`step0_audit.json`, sha256 `b320a601c60b06258d66350d3274d64defc81317fcef8c4c0db7c39ce2209d61`; 5056 user turns, 180 speakers, folds 1-4) and the annotation guide (`doc/Guia_Anotacion_Disfluencias_Basurde.pdf`, sections 2.4 and 2.7):

1. The filled-pause mark `<f>` covers only vocalic/nasal pauses (guide 2.4): no lexical forms. It is NOT the reference here.
2. A different family covers lexical forms: the discourse-marker mark `<d tipo=...>` (guide 2.7). Its value `fill` ("relleno": examples "pues", "entonces", "mire", "verá"; XML `tipo="rellenar"`) is the human "filler" function. The other values give the legitimate functions of the same words: `accept` ("bueno", "vale"), `explain` ("o sea", "es decir"; XML `editar`), `request` ("¿eh?"), `open`, `close`. The guide states the same word may or may not act as a marker ("bueno" can be an adjective) and gives a deletion test for the decision. The corpus audit lists `d` negative coverage as accredited by protocol (`MATRIZ_CORPUS.md`), with the note "do not evaluate as filler_word without a mapping decision". The mapping decision is made here: construct = "lexical filler function of an ambiguous form", which is NOT the engine's `filler_word` (filled pause).
3. Counts per form (occurrences in the mark-free text, folds 1-4): pues 159 (fill 152), mira/mire 42 (38), a ver 37 (16), bueno 32 (fill 9, accept 17, open 2, unmarked 4), entonces 16 (9), o sea 11 (fill 3, explain 7, unmarked 1), vale 15 (fill 0), este 59 (marked 0; all inspected followers are nouns/dates: demonstrative), eh 6 (all `request`/`exclaim` "¿eh?", never a hesitation).
4. Consistency evidence: by site, pues is marked 96-97% at all three sites and a ver 42-44% at all three (consistent); bueno-as-fill occurs only at one site (9/22 there, 0/10 at the other two); unmarked occurrences exist that look like fillers by position (e.g. turn-initial "Pues" 3 times, "entonces" after "Sí,"), so UNMARKED is a noisy negative for pues/entonces/mira. No inter-annotator agreement for `<d>` was found in the DIHANA documents.

Decision: the reference is VALID WITH DECLARED LIMITS for a restricted scope; it is NOT valid for "este" and lexical "eh" (zero positives: no use of either as filler is annotated, so recall and F1 are undefined; they are kept as negatives and reported for specificity only). The design question for "este" cannot be answered with DIHANA. Positives are reliable (a human actively marked them); negatives carry unquantified noise. Because the check was on counts only, measurement proceeds; if the data contradict this decision a stop report is the outcome, not a remap. Alternative resources (ES-Port, AMERESCO, EPIC, disfluentES) were reviewed in the corpus matrix: ES-Port marks `<%pues>`-type continuers and fillers jointly with no function, no audio; EPIC marks only FPAUSE; AMERESCO and disfluentES have no lexical filler function. None gives a function-level reference, so none is used.

## 1. Unit and label

Unit = one occurrence of a form in the mark-free input view of a user turn (tokens of `"".join(<t>.itertext())`, punctuation tokens dropped, accent-free lowercase). Forms: este, bueno, o sea, pues, entonces, mira/mire, vale, a ver, eh. Matching: longest form first, non-overlapping. y = 1 iff every token of the occurrence is inside one `<d tipo="rellenar">` element; y = 0 otherwise (other `<d>` types, unmarked). `<f>`, `<a>`, `<p>`, `<l>` marks and `d` types are reference-only fields and are never model inputs.

## 2. Scopes (fixed)

- POOLED: all forms except `eh` (n counts above; `este` and `vale` are all-negative).
- CONTESTED (primary headline, "ambiguous subset"): forms where both classes are present in non-trivial numbers in folds 1-4: bueno, a ver, entonces, o sea. Chosen after seeing the Step-0 counts (declared). Expected n = 96, positives 37.
- Per form: reported with raw support; forms with fewer than 10 positives or fewer than 10 negatives are DESCRIPTIVE only, never tested.

## 3. Conditions

- A: reference text without marks (pure classification). Text features only.
- B: frozen ASR cache (`results/mejoras_20261030/asr_cache/dihana`, profile P0, all turns; P1 only on the 1025-unit asr_recovery subset: descriptive). Reference occurrences are aligned to ASR words with `difflib.SequenceMatcher(autojunk=False)` on normalised tokens; an occurrence is RECOVERED iff every token falls in an `equal` block. Not recovered = ASR deleted or changed it. Reported: (B-e2e, primary B) all reference occurrences, non-recovered ones count as predicted 0 by every system; (B-rec) recovered ones only. ASR words not in the reference (insertions such as "Eh,") have no label and are only counted.

## 4. Comparators (fixed before running)

- R1 lexical: every occurrence of a POOLED form is filler (A and B).
- R1L lexical with list: filler iff form in the guide's fill list {pues, entonces, mira/mire}.
- R2 simple context (uncalibrated, thresholds are placeholders): filler iff form in {pues, entonces, mira/mire, bueno, a ver, o sea} AND the turn is not just the form (stand-alone "bueno"/"vale" is an answer); in B also AND (gap_before + gap_after >= 0.10 s).
- RF form prior: logistic regression on the form one-hot only, trained on the training folds (shows whether context adds anything beyond the word).
- V0 projected (B only): filler iff some 3 s window of the frozen V0 feature cache (`features/dihana_zero_*.pkl`) that overlaps the occurrence ASR interval has argmax `filler_word`. Vc (head trained on folds 1-3) is evaluated on fold 4 only and flagged descriptive; V0's class is the filled pause, so this is a cross-construct comparator, declared.
- CTX (proposed): L2 logistic regression. Text features: form one-hot, tokens at -2,-1,+1,+2 (vocabulary: train-fold words with count >= 3, else UNK), form x neighbour interactions for the form prior, relative position, is_first, is_last, stand-alone flag, turn length bucket. Condition B adds acoustic features of the aligned ASR word: duration, duration per character, gap before/after, ASR word probability, relative RMS, F0 mean/slope (semitones from the turn median, 8 kHz u-law decode, `librosa.yin`). No punctuation features in the headline (the guide makes annotators put commas around markers; punctuation would leak the label). Sensitivity run with punctuation features is reported separately and flagged as leaky.
- Validation: outer leave-one-official-fold-out over folds 1-4 (speaker-disjoint); inner speaker-grouped 3-fold (`group_folds`, seed 13) selects C in {0.03, 0.1, 0.3, 1, 3} and the decision threshold (max F1 on inner OOF scores) using the training folds only. Reported predictions are outer out-of-fold.

## 5. Metrics and tests

F1_context = F1 of class "filler" over the scope; P, R, TP/FP/FN/TN; support per form. 95% CI: percentile bootstrap by speaker (2000, seed 13). Paired delta CTX minus each comparator on identical units: speaker bootstrap (5000, seed 13), `mrcd.context_eval.metrics`; Holm over the family {CTX-R1, CTX-R1L, CTX-R2, CTX-RF} computed separately for condition A and condition B-e2e on the CONTESTED scope. POOLED and per-form results are descriptive. Error analysis by form with local example indexes only (turn id + token offsets); no third-party text is committed.

## 6. Pre-declared reading rules

- "The route distinguishes by context" is claimed only if, in A on CONTESTED: F1_context(CTX) >= 0.60 (the original OE4 bar), AND CTX minus R1 and CTX minus RF are supported after Holm with the CI lower bound > 0. Otherwise the report says "not demonstrated" or "not measurable" with the reason. Note: R1 on CONTESTED has F1 = 2*37/(96+37) = 0.556 by construction, so 0.60 is a weak bar and the Holm comparisons are the substantive test.
- The CTX artifact is exposed as an opt-in function only if that criterion holds. Otherwise it is stored for reproducibility but not exposed.
- A positive result is bounded by: one telephone domain, peninsular speakers, noisy negatives, tiny contested support, label construct = discourse-marker function (not acoustic disfluency).

## 7. Limits declared in advance

Single domain; guide-based labels without a published agreement; contested n about 96 (CIs will be wide); "este"/"eh" cannot be tested; V0/Vc detect filled pauses, not lexical fillers; fold 5 untouched; CPU shared with other processes (runs serial; load recorded in the run log).
