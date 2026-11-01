# S2 PROTOCOL - ASR repetition loops vs literal-repetition survival (EXPLORATORY)

Frozen before any alternative is decoded. The hash of this file is in `PROTOCOL.sha256.json` (created by `scripts/s2_noloop/freeze_protocol.py`).
Date of writing: 2026-11-01. Everything here is EXPLORATORY; defaults (V0, P0 CPU int8, `live_cfg_v2`) never change. DIHANA fold 5 is never read.

Already known to the author BEFORE freezing (declared, not hidden): the offline p0 8 s chunked decode loops on about 8-12 of 75 AMERESCO clips
(3b report); P1 raises survival 105/214 -> 144/214 on CPU (asr_recovery); the cached p0 / p1 CPU transcripts exist. The numeric thresholds of the
loop metric below were fixed after looking at the cached p0/p1 loop counts of the CONTROLS only (never at any alternative).

## 1. Units
* `U_W` (loops, WER): the 135 units of `results/cierre_oe1_oe3_20261031/3b/asr/units.json` = 75 AMERESCO clips (human ELAN reference; 4 recordings = 4 groups,
  66 flagged `noisy_ref`) + 60 DIHANA folds 1-4 turns >= 10 s (39 speakers; reference = orthographic transcript). Group = recording (AMERESCO) or speaker (DIHANA).
* `U_S` (survival, rule false positives): the frozen frame of `results/asr_recovery_20261031/subset.json`: 1025 DIHANA folds 1-4 turns (346 positives of either task,
  of which 214 `rep_literal` positives; 679 sampled negatives with `rep_literal==0 and revision==0`). Group = speaker.
* The union of both is decoded once per branch. No human annotation is added; no automatic judge is used as reference. Absence of label is never a negative.

## 2. Branches (at most 3 alternatives + 1 conditional combination; all fixed now)
All branches use Faster-Whisper `small` (same snapshot as the earlier experiments), `beam_size=5`, `word_timestamps=True`, `condition_on_previous_text=False`,
`vad_filter=False`, language `es`, and `ctranslate2.set_random_seed(13)` before each unit. Unless stated: p0 prompt, `temperature=0.0`, chunk 8 s / overlap 2 s, keep only the new region of each chunk (the engine's `transcribe_chunked`).
* `C0` control = profile `p0_cuda` (engine path, float16 CUDA) on GPU; the cached CPU int8 `p0` transcripts on CPU.
* `P1` = profile `p1_cuda` / cached CPU `p1_disfluent_prompt` (comparison, not a mitigation candidate).
* `A` = p0 prompt + `temperature=(0.0,0.2,0.4)`, `compression_ratio_threshold=2.4`, `log_prob_threshold=-1.0`, `no_speech_threshold=0.6` (faster-whisper's own fallback).
* `B` = per-chunk loop detection + re-decode of THAT chunk. A chunk is "looped" if its normalized tokens contain a back-to-back n-gram run reaching the
  thresholds of `mrcd.asr_profiles._scan_loops` (n=1: >=5 repeats; n=2,3,4: >=4) or its text (>= 80 chars) has zlib compression ratio > 2.4. Ladder (first attempt that is not
  looped wins): attempt 1 = same chunk, `initial_prompt=None`, temperature 0; attempt 2 = same chunk, `initial_prompt=None`, `temperature=0.4`, `best_of=5`, seed 13 set before the attempt.
  If no attempt is clean, the ORIGINAL chunk text is kept and the chunk is recorded as `unresolved`. Every re-decoded chunk is recorded (unit, chunk index, attempt used).
* `C` = p0 prompt, chunk 13 s / overlap 2 s (same `transcribe_chunked` semantics).
* `D` (conditional) = the single best mitigation X among A/B/C (rule below) combined with the P1 prompt. Executed only if BOTH hold: (i) X reduces loops (rule 4.1 on GPU) and
  (ii) P1 shows its survival effect on GPU (`P1` vs `C0` survival delta with CI lower bound > 0). Otherwise D is NOT run and reported as "not run: precondition failed".

## 3. Metrics (definitions)
* **Loop excess** of a transcript = sum over maximal back-to-back n-gram runs (n=1..4, greedy left to right, largest coverage at a position) with >= 4 repeats of `(repeats-1)*n` tokens, for the hypothesis minus the same quantity on the human reference text (floored at 0). Tokens = frozen normalization of `scripts/cierre3b/asr_analyze.py::norm_text`.
  **Loop clip** = loop excess >= 20. Secondary (lenient) flag: any `repeated_ngram_loop` finding of `detect_asr_anomalies`. Primary loop outcome = fraction of `U_W` clips that are loop clips (also the count and the total excess tokens).
* **WER** vs the human reference, pooled over `U_W` with the frozen normalization (accents kept) and the same alignment as `asr_analyze.py`. Never read as evidence of disfluency recovery.
* **Survival** of literal repetition = `mrcd.context_eval.survival.turn_survives('repetition', ref['rep_marks'], text, {'A_literal_repeat'})` on the 214 positives of `U_S` (audit definition, heuristic, not a label).
* **Rule FP** = fraction of the 679 sampled negatives of `U_S` on which the text rule `mrcd.context_eval.textfeat.rule_rep` fires (raw, unweighted).
* Secondary descriptive: WER vs reference text A on the negatives, empty outputs, mean words/unit, prompt echo, `unresolved` chunks, wall time / RTF.

## 4. Decision rule (fixed)
1. **Loop reduction (GPU)**: a branch "reduces loops" if its loop-clip count on `U_W` is <= (C0 count - 3) and C0 has >= 3 loop clips (otherwise: "not evaluable").
2. **Qualifies** (no harm) if additionally: pooled WER delta on `U_W` (branch - C0) point estimate <= +0.005 and upper 95 % CI bound < +0.02; AND survival delta point >= -0.02 with CI lower bound > -0.05; AND rule-FP delta point <= +0.01.
3. **Best** = among branches that reduce loops, the one with the lowest loop-clip count (ties: lowest WER, then simplest). If none reduces loops, the best-by-loop-count is still reported as the negative result.
4. Qualifying winner -> implemented as an opt-in experimental profile. No qualifying branch -> negative result recorded, nothing implemented.
5. **CPU confirmation**: the best branch is decoded on CPU int8 (the default path, `OMP_NUM_THREADS=4`) on the SAME units and compared with the cached CPU p0 / p1. Report whether the sign of every delta, the loop-clip reduction and the branch ranking hold between devices. Implementation requires the rule above to hold ALSO on CPU.
6. Determinism checks: C0 GPU decoded twice on the first 100 units of the union; CPU C0 re-decoded on 20 units and compared word for word with the cached p0 transcripts.

## 5. Statistics
Paired differences on the same units. 95 % percentile CIs by bootstrap over groups (speakers; recordings for AMERESCO), 5000 resamples, seed 20261101. Two-sided bootstrap p (floor 1/(B+1)).
Family for Holm: {A, B, C, D} x {loop-clip rate (`U_W`), WER (`U_W`), survival (`U_S` positives), rule FP (`U_S` negatives)}, each vs `C0` = 16 comparisons (D counted even if not run: conservative).
Comparisons against P1 and per-corpus rows are descriptive and outside the family. AMERESCO alone has 4 groups (recordings): its CI is NOT informative and is labelled so. Scores are not probabilities; nothing is calibrated.

## 6. Limits declared in advance
Single-corpus survival (DIHANA, telephone, peninsular); AMERESCO references are partly noisy (`noisy_ref`); `ctranslate2` sampling at temperature > 0 depends on the seed; GPU float16 text differs from CPU int8 text (S1);
survival is a heuristic; loop thresholds are conventions, not calibrated; the cached units that revealed the loops define `U_W` (selection of the problem, not tuning of an alternative); one model size / quantization pair.
