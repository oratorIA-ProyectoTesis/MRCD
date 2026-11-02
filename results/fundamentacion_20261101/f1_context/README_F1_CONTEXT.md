# F1_context on ambiguous Spanish forms (DIHANA folds 1-4, 2026-11-01)

EXPLORATORY. Self-review by the authors; no independent review. Protocol frozen before measuring (`PROTOCOL.md`, sha256 `6d1aa0c35a03bc73...`, `PROTOCOL.sha256.json`; annex `PROTOCOL_ANNEX_1.md`). DIHANA folds 1-4 only (telephone, peninsular Spanish, train information); fold 5 never opened. No existing default, service, extractor or optimize module was touched. No annotation was added: labels are the existing DIHANA marks.

## 1. Validity of the reference (Step 0)

- The filled-pause mark `<f>` covers vocalic/nasal pauses only; it is not a lexical-filler reference.
- Lexical forms are covered by the discourse-marker mark `<d tipo=...>` (annotation guide, section 2.7). Its value `rellenar` ("fill": examples pues, entonces, mire, verá) is the human filler function; the other values give legitimate functions of the same words (`aceptar` for bueno/vale, `editar` for o sea).
- Construct measured: "lexical filler function of an ambiguous form", NOT the engine's `filler_word` (filled pause).
- Verdict: VALID WITH DECLARED LIMITS, restricted scope. Positives are reliable; negatives (unmarked or other `d` types) are noisy. Evidence of noise: near-identical contexts of turn-initial "bueno" are marked `aceptar` in some turns and `rellenar` in others (ids in `errors_A_ctx.csv`); bueno-as-fill occurs at one recording site only (9 of 22 there, 0 of 10 elsewhere); some turn-initial "pues" and "entonces" after "sí" are unmarked. No inter-annotator agreement for `<d>` exists in the DIHANA documents.
- NOT measurable: "este" (59 occurrences, 0 marked; all follow-words are nouns/dates, so it is a demonstrative here) and lexical "eh" (6 occurrences, all "¿eh?" request tags). The thesis example "este" cannot be tested with DIHANA. ES-Port, AMERESCO, EPIC and disfluentES give no function-level lexical reference (see protocol), so none was used.

## 2. Results

Unit = one occurrence in the mark-free text. y=1 iff covered by `<d tipo=rellenar>`. 371 occurrences, 227 positive (180 speakers). CONTESTED subset (bueno, a ver, entonces, o sea; chosen after seeing counts, declared): 96 occurrences, 37 positive. F1 CI = 95% speaker bootstrap. A = reference text; B-e2e = frozen ASR P0, occurrences the ASR drops count as predicted 0.

CONTESTED, F1_context (primary):

| system                                  | A: P / R / F1 [CI]               | B-e2e: P / R / F1 [CI]                              |
| --------------------------------------- | -------------------------------- | --------------------------------------------------- |
| R1 lexical "every occurrence is filler" | 0.39 / 1.00 / 0.556 [0.44, 0.64] | 0.38 / 0.92 / 0.540 [0.41, 0.63]                    |
| R1L list {pues, entonces, mira}         | 0.56 / 0.24 / 0.340 [0.18, 0.50] | same                                                |
| R2 simple context rule                  | 0.39 / 1.00 / 0.561 [0.45, 0.64] | 0.41 / 0.81 / 0.545 [0.42, 0.64]                    |
| RF form prior (no context)              | 0.39 / 0.24 / 0.300 [0.12, 0.46] | 0.42 / 0.22 / 0.286 [0.10, 0.46]                    |
| V0 projected (filled-pause class)       | n/a                              | 0.40 / 0.57 / 0.467 [0.30, 0.60]                    |
| CTX contextual classifier               | 0.53 / 0.46 / 0.493 [0.27, 0.68] | 0.49 / 0.46 / 0.472 [0.32, 0.65] (text only: 0.204) |

Paired CTX minus comparator on CONTESTED (speaker bootstrap, Holm over 4):

| vs  | A: delta [CI], p Holm        | B-e2e: delta [CI], p Holm    |
| --- | ---------------------------- | ---------------------------- |
| R1  | -0.064 [-0.273, 0.135], 1.00 | -0.067 [-0.185, 0.105], 0.68 |
| R1L | +0.153 [-0.053, 0.329], 0.53 | +0.133 [-0.080, 0.343], 0.64 |
| R2  | -0.068 [-0.275, 0.130], 1.00 | -0.073 [-0.193, 0.100], 0.68 |
| RF  | +0.193 [-0.066, 0.466], 0.53 | +0.187 [-0.024, 0.448], 0.37 |

POOLED (all forms except eh; descriptive; pues and mira/mire are 96% and 90% filler, so word identity dominates): A: R1 0.759, R1L 0.896, RF 0.882, CTX 0.900 [0.86, 0.94]; CTX minus R1 +0.141 [0.095, 0.197]; CTX minus RF +0.018 [-0.010, 0.050]. B-e2e: R1 0.748, R1L 0.881, RF 0.871, CTX 0.877 [0.83, 0.92], V0 0.594.

Per form (`per_form_table.md`): pues 0.98 and mira/mire 0.95 (word identity suffices); a ver is the only form where context helps (CTX F1 0.73 vs form prior 0.39 vs R1 0.60; 16 positives, descriptive); entonces 0.63; bueno 0.00 (9 positives; the CTX never fires, consistent with the label inconsistency above); o sea 3 positives (no inference). Sensitivity with punctuation features (leaky by annotation procedure): CONTESTED A 0.538, no significant gain.

ASR side: the standard Whisper pipeline (P0, with prompt) keeps these lexical forms: 356 of 371 occurrences recovered (96%; a ver 30/37, mira/mire 39/42). P1 on its 1025-turn subset does not recover more of them (no gain). Lexical forms are not what the ASR deletes; the filled pause is.

## 3. Conclusion

Not demonstrated, and below the original bar. On the contested subset F1_context of the contextual classifier is 0.49 [0.27, 0.68] against 0.56 [0.44, 0.64] for the lexical rule "every occurrence is filler"; the difference is -0.06 [-0.27, 0.14] (not supported after Holm), and the 0.60 target is not reached. On the pooled set, context adds nothing beyond word identity (+0.018 over a form-only model, CI includes 0). The only positive signal is "a ver" (descriptive, 37 occurrences). The classifier is therefore NOT exposed as an opt-in route and no artifact is shipped. The question stays open because the reference is the limiting factor (few contested positives, inconsistent negatives, one telephone domain), not because context was shown useless.

Sentence for the thesis statement: "Con las etiquetas existentes de DIHANA (marcadores de discurso con función de relleno), la distinción contextual de formas ambiguas como 'bueno', 'a ver' o 'entonces' no pudo demostrarse (F1_context = 0,49 [0,27-0,68] frente a 0,56 [0,44-0,64] de la regla léxica, n = 96, 37 positivos); 'este' y 'eh' léxico no son medibles con este corpus, por lo que el aporte de desambiguación contextual queda como hipótesis que requiere una referencia anotada específica."

## 4. Limits and what would change the answer

Single domain; construct = discourse-marker function (not acoustic disfluency); noisy negatives; contested n = 96; V0 detects filled pauses, so its row is a cross-construct comparator; Vc not run (annex). A valid test of "este"/"bueno" needs a reference with several tens of hesitation uses per form and agreement statistics; that requires new annotation, which was out of scope.

## 5. Reproduce

`PYTHONPATH=. python -I scripts/f1_context/audit_reference.py` (Step 0); `build_table.py` (alignment, acoustics, V0 projection; ~65 s); `run_eval.py`; `error_analysis.py`. Tests: `tests/test_f1_context_ambiguity.py`. CPU was shared (about 67% busy from other processes at start); everything ran serially. Intermediate pickles (`_table.pkl`, `_preds.pkl`) hold transcripts and are not committed; `errors_A_ctx.csv` has ids and offsets only.
