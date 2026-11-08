# Plan C — CTranslate2 conversion of CrisperWhisper 2.0 large (EXPLORATORY, DIHANA folds 1-4)

Written and committed BEFORE any conversion or decode of this plan. Status: exploratory. Nothing here is a
confirmatory result; every figure produced under this protocol is labelled "exploratory (DIHANA folds 1-4)".

## 1. Question

Does a CTranslate2 (CT2) conversion of CrisperWhisper 2.0 large (`cw2`, `nyralabs/CrisperWhisper2.0_large`,
revision `f4334f6e8193f2691212d49b20fa12d370e13896`) keep the verbatim output of the Hugging Face (HF)
`crisperwhisper` pipeline that the MRCD `cw2` profile uses, and how much faster is it on this GPU?

Known before running (not a result):

- The checkpoint carries extra keys `encoder_blank_head.{weight,bias}` that are not part of
  `WhisperForConditionalGeneration`; a CT2 conversion drops them. Whether verbatim behaviour survives is measured
  here, not assumed.
- `config.json` declares the architecture `WhisperForConditionalGenerationWithAttentionLoss`. If the converter
  refuses that name, the conversion is run on a local copy whose `config.json` names
  `WhisperForConditionalGeneration` (all other files linked unchanged); the edit is recorded.
- The verbatim mode of cw2 is selected by a decoder prompt: the tokens `[verbatim_1]..[verbatim_5]` followed by
  `<|startoftranscript|><|es|><|transcribe|><|notimestamps|>` (`crisperwhisper/prompt.py`). The CT2 arms pass the same
  token ids to `ctranslate2.models.Whisper.generate`.
- The package's own CT2 backend needs a forked `ctranslate2` build (`ctranslate2-crisperwhisper`); this plan uses the
  upstream `ctranslate2` 4.8.2 and its `generate`, i.e. the path an MRCD profile on stock CT2 would use.

## 2. Sample (fixed before decoding)

Universe: the 1025-unit frame of `results/asr_recovery_20261031/subset.json` (DIHANA user turns, folds 1-4 only,
122 speakers), restricted to turns with audio duration <= 30.0 s (one Whisper window, so long-form strategies
cannot differ). Fold 5, NOCANDO, AMERESCO, EPIC and CHM150 are never read.

Strata (counts fixed): **R** = 24 turns with `rep_literal == 1`; **F** = 12 turns with `rep_literal == 0`,
`revision == 0` and >= 1 non-lexical `<f>` tag (`f:a`, `f:e`, `f:m`, `f:sin_identificar`); **N** = 12 sampled
negatives (`kind == "neg"`) with no non-lexical `<f>` tag. Total 48 (the requested "about 40" rounded up so that
stratum R has 24 survival opportunities).

Selection rule: each unit gets the key `sha256("20261108:" + unit_id)`; strata are filled in the order R, F, N by
ascending key, skipping any unit whose speaker is already in the sample (one turn per speaker, so the speaker
bootstrap below resamples units). Seed string `20261108`. The selected ids are written to `cache/sample.json`
(git-ignored); only counts are published.

Audio: decoded on the fly from the original `.ul` files with the confirmed codec of
`scripts/baselines/dihana_decode.py` (mu-law 8 kHz, `resample_poly(2, 1)` to 16 kHz, PCM16 round trip in memory).
The sha256 of every reconstructed PCM16 WAV is checked against `results/baselines_20261030/dihana/wav16k_manifest.json`;
a mismatch aborts.

## 3. Systems (all on the RTX 5060 Laptop GPU, 8 GB, shared with other processes)

| arm         | runtime                                                                                                | decoding                                                                                                                                                                                                                                                                                                                |
| ----------- | ------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| HF_full     | `crisperwhisper` 2.0.3, transformers backend, float16 (env `_venv_cw2`)                                | `CrisperWhisperModel.transcribe(audio, language="es", mode="verbatim", word_timestamps=False, sr=16000)`, package defaults (temperature fallback, hallucination mitigation, early-EOT recovery). Same call as `scripts/asr_survival_models/decode_crisper.py`. `torch.manual_seed(13)` per unit. This is the reference. |
| HF_greedy   | same loaded engine                                                                                     | one greedy `generate` with the verbatim prompt, `max_new_tokens=256`, the model's default suppress list; no fallback or repair. Decomposes pipeline effects from conversion effects.                                                                                                                                    |
| CT2_fp16    | `ctranslate2` 4.8.2 (upstream), `compute_type="float16"`, model `<ct2_models_dir>/cw2_fp16` | `Whisper.generate(features, [prompt_ids], beam_size=1, max_length=448, suppress_blank=False, suppress_tokens=<same default list as HF>)`; features from the same `WhisperFeatureExtractor` (80 mel bins).                                                                                                               |
| CT2_int8f16 | same, `compute_type="int8_float16"`, model `<ct2_models_dir>/cw2_int8f16`                   | as CT2_fp16.                                                                                                                                                                                                                                                                                                            |

Determinism: the first 8 sample units are decoded a second time by every arm; the identical count is reported.

## 4. Metrics (per arm, on the same 48 units)

Scoring text: CrisperWhisper bracketed tags removed (`[UH]`, `[laughter]`, ...), as in
`results/asr_survival_models_20261104/PROTOCOL.md` section 4.3(a).

1. **WER** vs the human transcript with marks stripped (`ref_text_marks_stripped` of
   `results/baselines_20261030/dihana/turns.csv`), pooled over the 48 units, `mrcd.asr_recovery.stats.wer`. Text A
   has no non-lexical fillers, so an emitted filler counts as an insertion (descriptive).
2. **Literal-repetition survival** on stratum R: `turn_survives("repetition", rep_marks, text, {"A_literal_repeat"})`
   (`mrcd/context_eval/survival.py`, same definition as `results/asr_recovery_20261031/EXPERIMENTO_ASR.md`).
3. **Filler-token recall** on every sample unit with >= 1 non-lexical `<f>` tag: reference count = number of those
   tags; hypothesis count = filled-pause tags (`uh um eh em er ah ahh hm hmm mm mmm`) + non-lexical filler words of the
   stripped text (`transcript_reference.is_nonlexical_filler`); recall = sum(min(hyp, ref)) / sum(ref)
   (count-based, position-free). Also reported: turn-level filler survival as in the survival-models protocol.
4. **Exact-match rate** of each CT2 arm against HF_full and against HF_greedy: raw output text, whitespace-normalised,
   tags kept. Also the rate on normalised scoring tokens.
5. **RTF** = sum(decode seconds) / sum(audio seconds) on the 48 units after 2 untimed warm-up units; also the median
   per-unit RTF. Load time and peak GPU memory reported separately. Shared GPU: RTF is indicative.

Uncertainty: 95 % percentile bootstrap over speakers (= units, one per speaker), 5000 resamples, seed 20261108, for
each arm and for the paired differences CT2 - HF_full.

## 5. Acceptance criterion (fixed now)

A CT2 arm V is **acceptable** if, against HF_full on the same units, all hold on point estimates:

- A1 survival(V) >= survival(HF_full) - max(0.02, 1/n_R) (at most 2 points, or one mark, lost);
- A2 WER(V) <= WER(HF_full) + 0.01 (at most 1 WER point worse);
- A3 filler recall(V) >= recall(HF_full) - max(0.02, 1/n_ref_fillers);
- A4 RTF(V) < RTF(HF_full).

The criterion uses point estimates because a non-inferiority test with 24 survival opportunities has no power; the
intervals are reported next to them. If at least one arm is acceptable, the opt-in profile `cw2_ct2` is added to
`mrcd/asr_profiles.py` for the acceptable arm with the lowest RTF (same licence note as `cw2`, `experimental=True`,
never a default), with unit tests. If none is acceptable, no profile is added and the failed criteria are stated.
If the conversion itself fails, every CT2 metric is `not_available` with the error, and work moves to Plan B.

Expectation (author, not a result): the blank head is a training-time auxiliary and is not read by either
`crisperwhisper` backend at inference, so CT2_fp16 should match HF_greedy closely; HF_full may differ from both on a
few units because of its fallback/repair; int8_float16 may lose a few points of verbatimness.

## 6. Licence

CrisperWhisper 2.0 weights AND outputs are under the nyra health Non-Commercial Research License. A converted or
quantised model is a Derivative Work (licence section 1, item (b)) and stays under that licence (section 3); it is
kept outside the repository (`<ct2_models_dir>/`) and is never distributed with MRCD. Outputs are not
published (only aggregates). No cw2 output is used as a training label or teacher signal anywhere (see Plan B).

## 7. Outputs

`results/plan_c_20261108/RESULTS_C.md` and `RESULTS_C.json` (aggregates only, committed with `git add -f`);
per-unit decodes under `results/plan_c_20261108/cache/` (git-ignored, never committed). Code: `scripts/plan_c/`.
