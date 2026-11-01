# Protocol annex 3 (2026-10-31): control prompt C1 for the attribution of the O1 gain (EXPLORATORY; written and hashed BEFORE any C1 decode)

Motivation: REVISION4 sections 3.7 and 4.1 note that P1 differs from P0 in style (repetitions, self-corrections, ellipses), length and domain words, so the O1 gain cannot be attributed to "verbatim decoding style" without a control prompt. This annex adds ONE control, C1. Nothing else of the protocol changes; PROTOCOL.md is not edited (its sha256 is unchanged). This annex is hashed (`PROTOCOL_ANNEX_20261031_3.sha256.json`) and committed to git before the first C1 decode, which removes the "evidence by mtime only" caveat of REVISION4 section 3.1 for this part.

## 1. Control prompt C1 (written once; no other prompt is tried)
`"Buenos días, quería preguntar por el horario de las salidas de mañana. Me interesa conocer la duración del trayecto hasta la ciudad vecina y las tarifas para estudiantes. Además, quisiera elegir un asiento junto a la ventanilla, en un vagón silencioso. Muchas gracias por su ayuda."`
Defined in `mrcd/asr_recovery/decode.py` as `C1_PROMPT` (variant `C1`: identical to P1 except `initial_prompt`; `temperature=0.0`, `condition_on_previous_text=False`, beam 5, chunking 8 s / 2 s, `ctranslate2.set_random_seed(13)` per unit, OMP_NUM_THREADS=4 before numpy).
Design: fluent Spanish, the same domain words as P1 (`horario`, `preguntar`, `quería`; DIHANA is a train-timetable domain) plus ticket/seat vocabulary, comparable length, and NO adjacent repeated words, fillers, ellipses or self-corrections. Prompt properties are measured by `scripts/asr_recovery/s8_c1_prompt_check.py` (output `c1_prompt_check.json`, run before any decode, no ASR involved):
* P1: 45 words, 216 characters, 71 Whisper tokens, 4 adjacent identical word pairs, 5 ellipses, fillers present.
* C1: 46 words, 281 characters, 75 Whisper tokens (+5.6 %), 0 adjacent identical pairs, 0 ellipses, no fillers.
* n-gram overlap with the reference text of DIHANA turns (normalised tokens): C1 has 0 five-grams and 0 four-grams with the 1025 evaluated units (frame); 3-grams: "buenos dias queria", "el horario de" in 42 frame turns (generic phrases of the domain); 1 four-gram ("la duracion del trayecto") in 3 pool turns outside the frame. P1 for comparison: one four-gram ("si si es el") in 1 frame turn. Statement used in the report: "no five-gram or four-gram overlap with the evaluated turns; generic three-gram overlap".
The prompt was written knowing the P1 result (declared), but without looking at any C1 output.

## 2. Subset, variants, reuse
Same frozen subset (`subset.json`, sha256 `d1b1c7be...`, 1025 units: 346 positives + 679 sampled negatives, 122 speakers, same weights). P0 and P1 are NOT re-decoded: their stored word lists (`asr/P0`, `asr/P1`) are used (P0 re-decode equals the frozen cache in 1025/1025; the re-decode of 20 turns by the reviewer reproduced P0 and P1 exactly). Only C1 is decoded (`s2_decode.py C1 <shard> <nshards>`).
Same models: ctx_LR(B) per outer fold fitted on P0-text TRAIN speakers (not retrained for C1), same thresholds, same parent rules.

## 3. CPU budget
RTF of the protocol (P0 settings, 1 process, OMP=4, `timing.json`) = 0.772 on 20 probe turns (not re-measured with C1 to avoid decoding with C1 before this annex is hashed). Expected single-process time = 0.772 x subset audio (about 4577 s) = about 59 min; to stay within about 90 min wall time with margin, C1 is decoded with 2 shards (2 x OMP=4 = 8 of 20 logical cores) and no other heavy job of this session is run concurrently beyond 2 more processes. Measured wall time and per-unit RTF are reported with the results.

## 4. Outcomes and family (pre-declared)
Per variant P0, C1, P1 on the frame: (a) survival of the positives (rep_literal A-marks; revision) with speaker-bootstrap CI (2000, seed 13) and paired differences; (b) ctx_LR and rule weighted F1 (speaker bootstrap CI), TP/FP/FN/TN raw and weighted, FP on sampled negatives and paired new/removed FP; (c) WER of each variant on the sampled negatives against reference text A (a damage control only) and prompt-echo rate (>= 4 consecutive prompt tokens; each variant against its own prompt, plus P1-prompt echo in C1 output and vice versa); (d) counts of adjacent identical word pairs on sampled negatives (loop hallucinations).
Family (EXPLORATORY, Holm over 4): paired difference of ctx_LR weighted F1 on identical units, speaker bootstrap 5000, seed 13, `p = 2*min(P(d<=0), P(d>=0))` (reported as `< 0.0004` when it equals the bootstrap floor 2/(5000+1)), for {rep_literal, revision} x {P1 - C1, C1 - P0}. Everything else is descriptive. The share `(F1w(C1) - F1w(P0)) / (F1w(P1) - F1w(P0))` of the P1 gain that C1 reproduces is a point estimate with a speaker-bootstrap CI (2000), reported only if the denominator > 0.02.
## 5. Decision rule (primary: rep_literal, ctx_LR)
* "P1 > C1; the gain is attributable to the verbatim content of the prompt beyond a generic prompt" only if the CI of (P1 - C1) excludes 0 on the positive side and p_Holm < 0.05.
* "C1 ~ P1; effect of a (domain, fluent) prompt in general" if the CI of (P1 - C1) includes 0 AND (C1 - P0) is positive with CI excluding 0 and p_Holm < 0.05. Absence of a detectable difference is NOT proof of equivalence; the CI of the difference is given so the reader can see which differences remain compatible.
* Otherwise "inconclusive". `revision` is read the same way and reported separately. Whatever happens is reported; no further prompt is tried.

## 6. Limits
Single control prompt (one draw of style/length/domain); P1 vs C1 differ also in characters and in specific words; the weighted frame, the best-of-2 selection of the parent family, one corpus and one ASR model/size apply as in PROTOCOL.md section 9. Fold 5 is not indexed or used.
