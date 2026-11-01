# S2 - ASR quality: repetition loops vs literal-repetition survival (2026-11-01)

**Self-review by the executing agent; no independent review.** EXPLORATORY. Defaults unchanged: V0, P0 (CPU int8), `live_cfg_v2`. DIHANA fold 5 never read. No human annotation added, no automatic judge used. Protocol frozen before measuring: `PROTOCOL.md` (sha256 `7ee91bfec81f182ec65308b17eb2fb73a3c645443607d8c635e3fe26746d9b92`, frozen 2026-11-01T06:29:06Z, commit `dce2133`).

## Answer
Yes, a mitigation exists that cuts loops without hurting WER or destroying repetitions: **re-decode only the chunk that looks looped, without the initial prompt** (branch `B`), and the same combined with the P1 prompt (`D_B`). Both qualified under the pre-registered rule on GPU and on CPU int8. Temperature fallback (`A`) and 13 s chunks (`C`) did not. Two caveats weigh on how strongly this can be stated: the loop evidence comes from **4 AMERESCO recordings** (CI over recordings is not informative), and the loop *count* ranking between B and D_B flips between devices (below).

## Units and metrics (see PROTOCOL)
`U_W` = 135 units (75 AMERESCO clips, 4 recordings; 60 DIHANA folds 1-4 turns >= 10 s, 39 speakers). `U_S` = frozen frame of the ASR-recovery experiment (1025 DIHANA turns: 214 literal-repetition positives, 679 sampled negatives). Loop clip = n-gram run excess >= 20 tokens beyond the human reference. Survival = audit heuristic `turn_survives`. Rule FP = `rule_rep` on the 679 negatives. WER = frozen normalization of the 3b report (never evidence of disfluency recovery). Branches: `C0` control (p0), `P1`, `A` = temperature (0,0.2,0.4) + compression/logprob thresholds, `B` = per-chunk loop detection + re-decode ladder (no prompt T=0; no prompt T=0.4 best_of=5), `C` = 13 s chunks, `D_B` = B with the P1 prompt (precondition met: B reduces loops and P1 has its survival effect on GPU).

## GPU float16 (all branches same device/precision; control `p0_cuda`) - `analysis/tables_cuda.md`
| branch | loop clips /135 (AMERESCO 75 / DIHANA 60) | WER all | survival /214 | rule FP /679 |
|---|---|---|---|---|
| C0 | 8 (7/1) | 0.546 | 107 (0.500) | 12 |
| P1 | 11 (11/0) | 0.520 | 148 (0.692) | 14 |
| A | 7 (6/1) | 0.524 | 107 (0.500) | 12 |
| B | 4 (4/0) | 0.420 | 105 (0.491) | 11 |
| C | 12 (12/0) | 0.553 | 100 (0.467) | 10 |
| D_B | 1 (1/0) | 0.384 | 147 (0.687) | 14 |

Paired vs C0 (bootstrap over groups: 43 for U_W = 4 recordings + 39 speakers; 84 for survival; 122 for FP; 5000 resamples), Holm over a fixed family of 16 (4 branches x 4 metrics; D counted):
| comparison | loop rate | WER | survival | rule FP |
|---|---|---|---|---|
| B | -0.030 [-0.065, 0.000] (Holm 1.0) | -0.126 [-0.206, -0.048] (Holm 0.10) | -0.009 [-0.029, 0.000] (1.0) | -0.0015 (1.0) |
| D_B | -0.052 [-0.119, -0.007] (Holm 0.44) | -0.161 [-0.266, -0.060] (Holm 0.012) | +0.187 [+0.133, +0.244] (Holm 0.006) | +0.003 (1.0) |
| A | -0.007 [-0.025, 0.000] (1.0) | -0.021 [-0.042, 0.000] (1.0) | 0.000 (A never fell back on these units) | 0.000 |
| C | +0.030 [-0.016, +0.069] (1.0) | +0.007 [-0.123, +0.157] (1.0) | -0.033 [-0.073, +0.005] (1.0) | -0.003 |

Decision rule: A not reduce loops (7 vs 8, needs <= 5); C worse; B and D_B qualify. Best by the rule (fewest loop clips): D_B.

## CPU int8 confirmation (default path; same units; controls = cached CPU p0/p1) - `analysis/tables_cpu.md`
| branch | loop clips | WER all | survival | rule FP |
|---|---|---|---|---|
| C0 (cached p0) | 11 (11/0) | 0.583 | 105 (0.491) | 11 |
| P1 (cached) | 9 | 0.531 | 144 (0.673) | 14 |
| B | 0 | 0.385 | 103 (0.481) | 10 |
| D_B | 2 (2/0) | 0.400 | 143 (0.668) | 12 |

Paired vs C0: B loops -0.082 [-0.154, -0.012], WER -0.198 [-0.275, -0.059], survival -0.009 [-0.029, 0.000], FP -0.0015. D_B loops -0.067 [-0.148, -0.010], WER -0.183 [-0.271, -0.073] (Holm 0.024), survival +0.178 [+0.129, +0.228] (Holm 0.006), FP +0.0015. Both qualify by the rule on CPU too.
**Does the ranking hold between devices?** The qualitative result does (loops down, WER down, survival preserved or, with the P1 prompt, +0.18). The *order of B vs D_B by loop count* does not (GPU: D_B 1 < B 4; CPU: B 0 < D_B 2). Absolute loop counts of the control also differ (GPU 8 vs CPU 11): float16 and int8 loop on different clips. With 1-4 clips these differences are within noise; the profile choice should rest on the survival column (D_B recovers the P1 gain, B does not), not on the loop count.

## What the numbers do NOT show
* The WER gain is almost entirely AMERESCO (0.79 -> 0.50 on CPU, 4 recordings, noisy references); DIHANA WER does not change (0.154 -> 0.154 B, 0.140 D_B). Loop rate Holm-adjusted p is 0.42-0.44 for both (not significant after correction); the WER and survival effects of D_B are.
* The survival gain of D_B is the P1 gain already known (D_B vs P1: -0.005 [-0.015, 0.000]); the re-decode adds no survival, it removes loops. The control prompt experiment (C1) showed P1's mechanism is unknown.
* Re-decode ladder rung 2 (T=0.4 sampling) is **not bit-reproducible** even with `ctranslate2.set_random_seed(13)` (a unit decoded 3 times in one process gave 3 texts); rung 1 is deterministic. It was used on 2-3 chunks per run (GPU B: 16 rung-1, 2 rung-2, 5 unresolved; CPU B: 19 / 3 / 3; chunks that stay looped keep their original text and are recorded). A re-run can therefore differ on those few units. Determinism checks: GPU C0 re-run 100/100 identical; CPU C0 20/20 identical to the cached p0; custom decoder == engine path on 40/40 units.
* Loop thresholds are conventions (not calibrated); survival is a heuristic; DIHANA is telephone peninsular only; timing columns are not clean (shared machine, load gate not applied): CPU RTF of B/D_B (0.48 / 0.45) is not comparable to S1's.

## Implemented (opt-in, experimental): `p0_noloop` and `p1_noloop`
`mrcd/asr_profiles.py` (profiles, ladder, `chunk_looped`, warning text), `VerbatimASR.transcribe` (`mrcd/extractors/linguistic.py`: first pass identical to the base profile; if `profile.loop_redecode` and the chunk text loops, re-decode per ladder; else untouched) and `engine._asr_profile_meta` (lists every replaced/unresolved chunk under `metadata.asr_profile.loop_redecode.events` and adds a warning). p0, p1, p0_cuda, p1_cuda: `describe()` and cache keys bit-identical (pinned by tests). New profiles have their own fingerprint (the ladder is part of the cache key). The transcript is never altered silently: replacement is logged, unresolved chunks keep the original. Not selected by default; the service/app do not expose it (S3 owns `mrcd/service`).
Tests: `tests/test_asr_noloop_profile.py` (19: registration, config == used kwargs, fingerprints, synthetic detector, replacement, unresolved keeps original, clean chunk untouched, p0/p1 never re-decode, engine metadata) and `tests/test_asr_noloop_metrics.py` (8: loop metrics and branch decoders with a fake model). Real-model spot check: profile path == experiment output on a rung-1 unit (identical words); differs on a rung-2 unit (sampling).
Engine edit scope: `engine.py` got 2 additive blocks (reset of the event log; metadata) - outside the files listed for S2 but necessary for "no silent change"; flagged for the reviewer.

## Commands (all from `<repo>`, OMP_NUM_THREADS=4)
```
python scripts/s2_noloop/freeze_protocol.py
bash scripts/s2_noloop/queue_gpu.sh        # C0 P1 A B C + determinism re-run (GPU DLL PATH from run_gpu.sh)
bash scripts/s2_noloop/queue2.sh           # D_B on GPU, B on CPU, CPU determinism
bash scripts/s2_noloop/queue3.sh           # D_B on CPU
python scripts/s2_noloop/analyze.py --device cuda --branches C0 P1 A B C D_B --out results/mejora_objetivos_20261101/S2/analysis
python scripts/s2_noloop/analyze.py --device cpu  --branches C0 P1 B D_B   --out results/mejora_objetivos_20261101/S2/analysis
```
Versions: faster-whisper 1.2.1, ctranslate2 4.8.2 (CUDA 12 DLLs from `_venv_gpu`: cublas 12.9.2.10, cudnn 9.27.0.42), model `small` sha256 `3e305921...d671`, RTX 5060 Laptop float16; CPU i7-13650HX int8. Transcripts (`asr/`, 35 MB, texts of third parties) stay local and are not committed; committed: protocol, analysis tables/JSON, this README, scripts, code, tests.

## Limits / pending
Self-review only. Four AMERESCO recordings carry the loop result. Rung 2 not reproducible. Profiles not exposed in the service or app; not integrated with `ctx_LR` (S3 retrains on P1 text).

## Suite
`OMP_NUM_THREADS=4 .venv\Scripts\python.exe -m pytest -q -rs` -> **897 passed, 10 skipped** in 185 s (`logs/pytest_full.log`; skips = MRCD_SLOW). Touched modules: 27 new tests plus the existing profile/chunking/recovery tests pass. The first full run had 1 failure (a duck-typed test double calling `VerbatimASR.transcribe` unbound); fixed by making the decode helper module-level, then re-run.

## Addendum 2026-11-01 (independent review S; nothing above was edited)
* **Headline, restated.** The earlier "Answer: Yes ... cuts loops" is stronger than the evidence. Restated: B and D_B **reduce loops in this sample** and qualify under the pre-registered, non-Holm rule, but the Holm-adjusted p of the loop *rate* is 0.42-0.44 (**not significant**). Only the WER (D_B: Holm 0.012 GPU / 0.024 CPU) and survival (D_B: Holm 0.006) effects survive correction.
* **WER gain is concentrated in AMERESCO** (4 recordings): B goes 0.794 -> 0.499 there and loops 11 -> 0 on CPU, while DIHANA WER is unchanged (delta 0.0 for B; D_B vs C0 on DIHANA only: -0.0146 [-0.030, -0.001], i.e. the P1 prompt effect). It is **robust in leave-one-recording-out** (WER delta stays between -0.14 and -0.19), so no single recording drives it, but all of it comes from 4 noisy-reference recordings.
* **Service exposure (corrects the final line "not exposed in the service").** `MrcdAnalyzer.validate_options` has always accepted every key of `PROFILES`, so `p0_noloop`, `p1_noloop`, `p0_cuda` and `p1_cuda` were reachable through the service. Decision: they are **accepted and explicitly marked experimental** (`experimental.components`, `experimental_components` warning, plus `asr_redecode_experimental` / `asr_gpu_text_unvalidated`); defaults are untouched. `p0_cuda`/`p1_cuda` without a usable GPU return the structured error `asr_unavailable` with a message that names the CUDA device (no CPU fallback). See `docs/contracts/ASR_PROFILES_CONTRACT.md` section 1b.
* **Re-decode events now record** `chunk_offset_s` and `reproducible` (false only for text that came from the sampled T=0.4 rung); the run metadata gets `n_non_reproducible` / `reproducible` and the `warning` says so. Results of S2 were not recomputed (the analysis scripts use their own decoder). The process-global `ctranslate2.set_random_seed(13)` is a documented side effect.
