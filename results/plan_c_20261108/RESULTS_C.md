# Plan C results: CrisperWhisper 2.0 large on CTranslate2 (EXPLORATORY, DIHANA folds 1-4)

Protocol: `PROTOCOL_EXPLORATORY_C.md` (committed in 28437bc before any conversion or decode). Aggregates:
`RESULTS_C.json`. Code: `scripts/plan_c/`, shared metrics `scripts/verbatim_ft/vft_metrics.py`. Every figure below is
**exploratory (DIHANA folds 1-4)**; 48 turns, 48 speakers, 237.2 s of audio; 95 % CIs are speaker-bootstrap
percentile intervals (5000 resamples, seed 20261108). Per-unit decodes stay in the git-ignored `cache/`.

## Headline

- **The float16 conversion keeps the verbatim output exactly**: CT2_fp16 text is identical to the HF `crisperwhisper`
  pipeline (profile `cw2`) on **48 / 48** units (raw text, tags included). It runs at **RTF 0.092 vs 0.190** on the
  RTX 5060 Laptop GPU, a **2.06x** speed-up (decode only; load 3.3 s vs 8.3 s).
- int8_float16 is also acceptable but no better: 44 / 48 raw-identical (46 / 48 on scoring tokens), all four metrics
  unchanged, and slower than fp16 on this GPU (RTF 0.114, 1.66x).
- Both CT2 arms pass the pre-declared criterion A1-A4. Profile **`cw2_ct2`** (float16, lowest RTF) was added, opt-in.
- The dropped `encoder_blank_head.{weight,bias}` were never used at inference: the HF pipeline itself loads the
  checkpoint into `WhisperForConditionalGeneration` and reports the same two keys as UNEXPECTED
  (`logs/decode_hf.log`). The conversion therefore removes nothing the reference pipeline uses.

## Metrics (same 48 units)

| arm                        | WER [95 % CI]        | literal-repetition survival | filler-token recall [CI]      | filler turn survival | RTF (sum) | RTF (median unit) |
| -------------------------- | -------------------- | --------------------------- | ----------------------------- | -------------------- | --------- | ----------------- |
| HF_full (reference, `cw2`) | 0.069 [0.042, 0.096] | 24 / 24 = 1.00 [1.00, 1.00] | 23 / 26 = 0.885 [0.741, 1.00] | 21 / 24 = 0.875      | 0.190     | 0.205             |
| HF_greedy                  | 0.069                | 24 / 24                     | 0.885                         | 0.875                | 0.195     | 0.235             |
| CT2_fp16                   | 0.069                | 24 / 24                     | 0.885                         | 0.875                | **0.092** | 0.101             |
| CT2_int8f16                | 0.069                | 24 / 24                     | 0.885                         | 0.875                | 0.114     | 0.131             |

Paired CT2 - HF_full: 0.000 on every metric, CI [0, 0] (identical scores per unit). Exact match: CT2_fp16 vs HF_full
and vs HF_greedy 48 / 48 (raw and tokens); CT2_int8f16 44 / 48 raw, 46 / 48 tokens. HF_full equals HF_greedy on all 48
(no fallback or repair fired on these short turns). Determinism: 8 / 8 identical re-decodes for every arm. HF_full is
also identical (48 / 48) to the 2026-11-04 cw2 cache decoded in another environment.

Acceptance (section 5, point estimates): A1 survival tolerance max(0.02, 1/24) = 0.042, A2 WER +0.01, A3 filler
recall tolerance max(0.02, 1/26) = 0.038, A4 RTF lower. CT2_fp16: all true; CT2_int8f16: all true.

## Descriptive, added after the protocol (not used for acceptance)

- Same units, cached small-model texts: P0 WER 0.151 [0.115, 0.195], survival 12 / 24 = 0.50 [0.30, 0.70], filler-token
  recall 0.385 [0.20, 0.59]; P1 WER 0.158, survival 14 / 24 = 0.58 [0.39, 0.79], filler recall 0.423. This only
  checks that the metric code separates systems; this 48-unit sample is not the frame of EXPERIMENTO_ASR.
- CPU probe (G4 relevance): CT2 int8 on CPU, 4 threads, 6 timed units (61 s audio): **RTF 1.47**, i.e. about 176 s for
  a 2-minute session. cw2 on CPU cannot meet G4 (<= 10 s wait after close); 4 / 6 texts identical to HF_full.
- Profile check: `VerbatimASR(profile="cw2_ct2")` (faster-whisper feature extractor with the waveform padded to 30 s,
  CT2 cross-attention word times) gives a word sequence identical to HF_full on 48 / 48 units, RTF 0.114 including
  word alignment; a 95 s concatenation (12 sample turns) runs through the 30 s / 4 s windowing (239 words, times
  monotonic). Windowing and word times are smoke-tested only, not validated.

## Conversion record

`ct2-transformers-converter` (ctranslate2 4.8.2, transformers 5.22.1, torch 2.9.1+cu130, env `_venv_cw2`), source the
HF cache snapshot at revision `f4334f6e8193f2691212d49b20fa12d370e13896`, `--quantization float16` and
`--quantization int8_float16`, `--copy_files tokenizer.json preprocessor_config.json added_tokens.json
generation_config.json special_tokens_map.json tokenizer_config.json vocab.json merges.txt normalizer.json LICENSE.md`.
No config edit was needed (the custom architecture name was accepted). Only warning: the LOAD REPORT listing
`encoder_blank_head.weight` and `encoder_blank_head.bias` as UNEXPECTED (`logs/convert_*.log`). Output (outside the
repo): `<ct2_models_dir>/cw2_fp16\model.bin` 3.1 GB sha256 `3e47089c...754b2e`;
`cw2_int8f16\model.bin` 1.6 GB sha256 `e96b0560...1c9fbe`. Upstream ctranslate2 is a CUDA 12 build: it needs the CUDA 12
cuBLAS/cuDNN DLLs of `<venv_gpu>` on `PATH` (POSIX form in Git Bash).

## Limits

48 short turns (<= 30 s) of one telephone corpus; survival is at ceiling for every cw2 arm (24 / 24), so the sample
cannot show a small loss; CIs are wide (filler recall n_ref = 26). The long-form path (> 30 s) of `cw2_ct2` differs from
the package's continuation strategy and is not validated; word times are not validated. Shared GPU: RTFs are
indicative. The package's own CT2 backend (forked `ctranslate2-crisperwhisper`) was not tested.

## Licence

The CT2 conversions are Derivative Works under the nyra health Non-Commercial Research License (section 1(b): "fine-tuned,
quantized, pruned, merged, converted"), inherit it (section 3), and may not be used in production or offered as a hosted
service/API (sections 1, 2.3, 5). They stay outside the repository; `cw2_ct2` carries the same licence note as `cw2`.
No cw2 output is used as a training label or teacher signal (Plan B rule).
