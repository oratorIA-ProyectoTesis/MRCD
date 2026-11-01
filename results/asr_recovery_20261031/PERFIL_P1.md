# ASR profile `p1_disfluent_prompt` (EXPERIMENTAL, opt-in)

Implementation dated 2026-10-31. Code: `mrcd/asr_profiles.py`, `VerbatimASR(profile=...)`, `DisfluencyEngine.load(..., asr_profile=...)`; tests: `tests/test_asr_profiles.py`. **This implementation has NOT been independently reviewed** (single writer, self-checked by tests only).

## Safe statement
En este experimento exploratorio de DIHANA, un control fluido de igual longitud y dominio no reprodujo la ganancia observada con P1; un prompt escrito y fluido suprime repeticiones y P1 las conserva; qué rasgo importa (registro oral vs escrito, puntuación, solape de dominio, lista de muletillas ya presente en P0) no está probado.

Calificadores (revisión 6, autorrevisión): P0 ya es un prompt con lista de muletillas; no hay control sin prompt; C1 tiene más solape de dominio (42 vs 2 turnos con 3-grama compartido); un solo control y una sola longitud.

## What it is
* `p0` (default, unchanged): historical decoding. Its kwargs are pinned in a test against the literal pre-change call, and the slow test re-decodes 3 DIHANA turns and matches the frozen P0 cache word for word (text, start, end, prob).
* `p1_disfluent_prompt`: identical to `p0` except `initial_prompt`. Selected only by explicit request; an unknown name raises `ValueError` before any model is loaded.

| parameter | value (both profiles unless noted) |
|---|---|
| model | faster-whisper `small` (Systran snapshot `536b0662742c02347bc0e980a01041f333bce120`) |
| `model.bin` sha256 | `3e305921506d8872816023e4c273e75d2419fb89b24da97b4fe7bce14170d671` |
| compute_type / device | int8 / CPU (what the experiment used; the engine records the actual values) |
| beam_size / temperature | 5 / 0.0 (no fallback) |
| condition_on_previous_text / vad_filter | False / False |
| word_timestamps / suppress_tokens | True / `[]` |
| chunking | 8 s chunks, 2 s overlap (only audio > 8 s is chunked) |
| OMP_NUM_THREADS | 4, set in the process environment BEFORE the first numpy import (recorded; the engine warns otherwise) |
| P0 prompt | `eh, este, o sea, mmm, bueno, tipo, ah, este...` (sha256 of the string via `get_profile("p0").prompt_sha256`) |
| P1 prompt | `Eh... yo quería, quería preguntar por el, por el... horario. Es que no, no sé si... si es el mismo. Perdone, digo, perdón. La mesa está en la ca... en la casa, o sea, en la casa de mi tía. Mmm, este... bueno, sí, sí.` |
| P1 prompt sha256 (UTF-8) | `fe49e930ccc9f94847b7bedaec65f8c347cecaccbcd7283545618bc194c6a8b2` |

## How to activate
```python
# OMP_NUM_THREADS=4 must already be in the process environment (shell / service unit)
engine = DisfluencyEngine.load("models/ckpt_audio_text.pt", asr_profile="p1_disfluent_prompt")
result = engine.analyze_file("audio.wav")
result.metadata["asr_profile"]   # profile, prompt + sha256, model + model.bin sha256, parameters, OMP, cache key, anomalies
result.warning                   # provenance warning + EXPERIMENTAL note + anomaly note
```
Shell (Windows Git Bash): `OMP_NUM_THREADS=4 python your_script.py`. Direct ASR use: `VerbatimASR("small", profile="p1_disfluent_prompt")`. Tests: `OMP_NUM_THREADS=4 MRCD_SLOW=1 python -m pytest tests/test_asr_profiles.py -q` (the slow tests need the snapshot above and the DIHANA wav16k files, and are skipped otherwise).

## What is recorded in every result
`AnalysisResult.metadata["asr_profile"]` (also in `to_dict()` / JSON): profile name, `experimental`, `validation` ("experimental; not validated outside DIHANA" for P1), literal prompt and its sha256, temperature, beam, condition_on_previous_text, vad_filter, chunk/overlap (and the chunk actually requested), model size/dir/`model.bin` sha256/compute_type/device, effective `OMP_NUM_THREADS` and whether it is verifiably effective, `asr_fingerprint` (`profile_fingerprint` + namespace; `asr_cache.key` is a deprecated alias, there is no cache), `anomalies`. For P1, `result.warning` additionally carries the experimental note. P0 results keep the historical `warning`; they only gain the additive metadata block.

## ASR cache
The engine has no ASR cache. `asr_cache_key(...)` (sha256 over profile name, prompt sha256, all decode parameters, chunking, `model.bin` sha256, compute_type, device, OMP, language, audio sha256) is reported as `asr_fingerprint.profile_fingerprint` (alias `asr_cache.key`, kept for compatibility; it does NOT imply that a cache exists) and must be used by any cache an integrator adds; `profile.cache_namespace()` (`<name>-<prompt sha12>`) is a safe directory name. P0 and P1 can never share a key (tested). Pre-existing script caches (`results/mejoras_20261030/asr_cache/*`) hold P0 text only and are untouched; P1 decodes of the experiment live in `asr/P1` and are not a cache of this API.

## Output anomalies (reported, never filtered)
`detect_asr_anomalies` runs on every transcript against the active profile's prompt: `prompt_echo` (>= 4 consecutive prompt tokens), `repeated_ngram_loop` (1-gram >= 5 back-to-back repeats; 2-4-grams >= 4), `high_compression_ratio` (zlib ratio > 2.4 on >= 80 chars). Thresholds are conventions of this module, not calibrated; a genuine "la la", "no no no" or "sí, sí" does not trigger. The detector is a HEURISTIC and reports POSSIBLE anomalies (see addendum 2026-10-31 R7 below). Anomalies go to `metadata["asr_profile"]["anomalies"]` and, for P1, into `warning`. The transcript is not modified. In the experiment the original detector flagged 2 of 1025 units (corrected in the addendum below: 1 genuine loop and 1 false positive, not 2 echoes).

## Limitations (read before using the output)
* The 0.690 weighted F1 is `ctx_LR` on DIHANA (rep_literal, 346 positives + 679 sampled negatives), trained on P0 text and NOT retrained for P1. It is not a result of the engine (`DisfluencyEngine` uses a different model).
* WER does not demonstrate recovery of disfluencies: C1 lowered WER almost as much as P1 (0.153 -> 0.129 vs 0.130) while destroying repetitions.
* Only one control (C1) was tested. It is unknown which component of P1 (repetitions, ellipses, fillers, corrections, length, vocabulary) causes the effect.
* Not validated: no evaluation on another compatible reference. Single telephone peninsular corpus, one Whisper size/quantisation. Do not present P1 output as validated disfluency detection.
* Per-class conclusions are limited to `filler_word`, observable lengthening (does not demonstrate disfluent function) and literal repetition. `revision` is an unconfirmed approximation. No F1 for `block` nor for the two pauses. Scores are not calibrated probabilities, and the absence of a label is not a negative.
* Checkpoints of the engine were trained on P0-style text; feeding them P1 text is a distribution shift that has not been measured.


## Addendum 2026-10-31 (round 7, self-review of the author; not an independent check)
* **"P1 echoed its prompt in 2 of 1025 units" is withdrawn.** What happened: the module's own detector flagged 2 stored P1 outputs. `V118_VA2c1_U_005` ("[quote withheld]") is a loop (a repeated-n-gram hallucination, not a prompt echo); `B501_BA7a2_U_007` ("[quote withheld]") is a genuine repetition that matches the prompt only through function words ("si si es el") and is a false positive. So: at most 1 plausible anomaly in 1025 units, and 1 false positive. Source: REVISION7 F-1 (reviewer), accepted.
* **The detector is a heuristic, not a classifier.** Changes in this round: a prompt echo now needs >= 2 distinct non-function words inside the matching 4-gram (`FUNCTION_WORDS`, `ECHO_MIN_CONTENT`); loop detection also runs after dropping vocalised fillers (`LOOP_FILLERS`), so "no se no se eh no se no se eh" is detected. Known remaining false positives: honest speech sharing content words with a prompt 4-gram ("la mesa esta en la casa de mi tia" is the prompt's own tail; "queria preguntar por el horario"). Known false negatives: loops interleaved with other words ("si claro si bueno si claro si bueno"), hallucinations unrelated to the prompt ("suscribete al canal"), echoes made only of function words. "eh" x5 and "ja" x6 DO flag (they are loops by definition). Thresholds (4-gram, 5/4 repeats, zlib 2.4) are conventions; 2.4 equals faster-whisper's own default, the 80-character minimum is an invention. Not re-run on the 1025 stored units in this round.
* **Decode tests are load-sensitive: "bit-identical" holds only on an unloaded machine with `OMP_NUM_THREADS=4` exported BEFORE numpy is imported.** With two concurrent suites (or 24 CPU burners on a 20-core machine) 6 slow tests failed (prob 0.256 vs 0.272, "tarde," vs "tarde") and passed again alone; the failing tests also exercise P0, so the cause is environmental (oversubscribed CPU), not the profile code. The slow tests now skip (they do not fail) when more than 50 % of the CPU is busy at fixture time, or when `OMP_NUM_THREADS != 4`; `MRCD_SKIP_LOAD_CHECK=1` forces them. Never run a slow suite concurrently with another job. The engine reports `omp_effective_verified` (may be `None`: not verified).
* **Marking.** For non-default profiles every event carries `asr_profile` and `asr_experimental: true` (`Event.asr_profile`, emitted in `to_dict()` only when set), and live rows get the same two keys when `LiveScorer(asr_mark=StreamingASR.asr_mark())` is used (`scripts/live_demo.py` does so). p0 events/rows keep exactly their historical schema. An ASR object without `.profile` is reported as `profile: unknown` (`experimental: null`), never as verified p0. `get_profile` rejects a custom `ASRProfile` that is not experimental and not equal to p0, and one that reuses a registered name with different parameters. `VerbatimASR.model_info()` now logs a hash/resolve failure and exposes `model_info_error`. With `model_bin_sha256 = None` the fingerprint does not contain the model identity beyond `compute_type`/`device`.
