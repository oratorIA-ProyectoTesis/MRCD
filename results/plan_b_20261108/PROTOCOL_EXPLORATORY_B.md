# Plan B — verbatim fine-tune of Whisper small on DIHANA folds 1-4 (EXPLORATORY)

Written and committed BEFORE any training or decode of this plan. Status: exploratory; every figure is labelled
"exploratory (DIHANA folds 1-4)". Code: `scripts/verbatim_ft/`.

## 1. Question

Plan C and the confirmatory secondary arm show that the verbatimness of the ASR is the main lever, but CrisperWhisper's
weights and outputs are non-commercial. Can a licence-clean model (`openai/whisper-small`, MIT, fine-tuned only on
DIHANA human transcripts, CC BY 4.0) keep more literal repetitions and filled pauses than the production p0 (and p1)
decode of the same size, at the CPU int8 cost that gate G4 needs?

## 2. Data and the no-teacher rule

- Only DIHANA user turns of folds 1-4 (`mrcd.context_eval.pool.ALLOWED_FOLDS`): the 5056-turn pool of
  `results/asr_recovery_20261031`. Fold 5, the 14 aligned NOCANDO recordings, AMERESCO, EPIC 136 and CHM150 are never
  read, decoded or scored.
- **No cw2 output is used as a label, a filter, a teacher signal or for checkpoint selection.** Targets come only from
  the human `<t>` transcripts. The CrisperWhisper licence (section 4.2/4.3) makes any model trained on its outputs a
  Derivative Work under its non-commercial licence; this rule keeps the fine-tuned model licence-clean.
- Target text (`vtext.verbatim_target`): every written word of `<t>` (literal repetitions, cut-word fragments as
  pronounced, mispronunciations as pronounced, words inside `<d>`, `<b>`, `<a>`, `<n>`); empty filled-pause elements
  `<f tipo=a|e|m|sin_identificar/>` become `ah | eh | mm | eh` (the transcripts carry no filler word, only the tag; the
  `sin_identificar` -> `eh` choice is a convention); `<p/>` and point noises dropped; punctuation spacing closed;
  case and punctuation otherwise as transcribed. Unit tests: `tests/test_verbatim_ft_text.py`.
- Audio: the original `.ul` files decoded on the fly (mu-law 8 kHz -> 16 kHz), sha256-checked against the historical
  wav16k manifest. All turns are <= 30 s (none is dropped for length).

## 3. Speaker-independent split (inside folds 1-4)

Rule (`vtext.make_split`, seed `20261108`): a speaker is "positive" if it has >= 1 `rep_literal` turn. **val** = 20 % of
the positive and 20 % of the other speakers, each stratum by ascending `sha256("20261108:" + speaker)`; **dev** = 10 % of
the remaining speakers by the same rule with seed `20261108:dev`; **train** = the rest. A DIHANA dialogue has one user
speaker, so the split is also dialogue-disjoint. Sizes (counted before training): train 129 speakers / 3627 turns /
188.6 min (164 `rep_literal` positives); dev 15 / 446 / 23.4 min (20); val 36 / 983 / 50.8 min (30 positives, 146 turns
with a non-lexical `<f>` tag). dev is used only for checkpoint selection; val only for reporting.

## 4. Training (fixed now)

`openai/whisper-small` (MIT) with transformers 5.22.1 / torch 2.9.1+cu130 in the new env `<venv_ft>`
(`BUILD.md` there). Full fine-tuning, fp16 autocast with gradient scaling, gradient checkpointing, AdamW lr 1e-5,
weight decay 0, 50 warm-up steps then linear decay, effective batch 16 (micro-batch 8 x 2 accumulation; if it does not
fit in 8 GB: 4 x 4; if that fails: encoder frozen; any fallback is recorded). Decoder prefix
`<|startoftranscript|><|es|><|transcribe|><|notimestamps|>`, no prompt, labels end with `<|endoftext|>`. Seed
20261108 for torch, numpy and the data order. Checkpoints (model, optimiser, scaler, RNG, step) every 100 steps in the
smoke run and every 200 in the capped run, outside the repo (`<ft_models_dir>/`); resumable.

- **Smoke run**: 200 optimiser steps from the base model; dev loss at steps 0, 100, 200; val metrics of the step-200
  checkpoint through the final pipeline (section 5).
- **Capped run** (fresh from the base model): at most 6 epochs or 2 GPU hours, whichever comes first; dev loss every
  200 steps and at the end. **Best checkpoint = lowest dev loss** (pre-declared; no val metric is used for selection).
  If the run does not fit in the session, the smoke result is reported with the exact resume command.

## 5. Evaluation (val only)

The best checkpoint is converted to CTranslate2 int8 (`ct2-transformers-converter --quantization int8`) and decoded with
faster-whisper on **CPU, int8, 4 threads** (`OMP_NUM_THREADS=4`) with the p0 pipeline settings except the prompt:
no `initial_prompt`, `beam_size=5`, `temperature=0`, `condition_on_previous_text=False`, `word_timestamps=True`,
`vad_filter=False`, `suppress_tokens=[]`, 8 s chunks with 2 s overlap, `ctranslate2.set_random_seed(13)` per turn.

Baselines on the SAME 983 val turns: **p0** = the frozen P0 cache (`results/mejoras_20261030/asr_cache/dihana`, equal to a
re-decode on 1025/1025 frame units); **p1** = the P1 decodes of `results/asr_recovery_20261031/asr/P1` for the 156 val turns
in that frame plus new decodes of the other 827 with exactly the same code and settings
(`mrcd.asr_recovery.decode.transcribe_chunked(asr, audio, "P1")`, CPU int8 4 threads, seed 13).

Metrics (`scripts/verbatim_ft/vft_metrics.py`, same definitions as Plan C):

1. literal-repetition **survival** on the 30 val `rep_literal` positives (definition of EXPERIMENTO_ASR.md);
2. **filler-token recall** on the 146 val turns with a non-lexical `<f>` tag; turn-level filler survival;
3. **WER** vs the verbatim target (with reconstructed fillers) and vs text A (human transcript without fillers, the
   historical reference; an emitted filler is an insertion there);
4. descriptive: filler words emitted per 100 reference words on val turns WITHOUT a `<f>` tag (false-filler rate),
   repetition-loop flags (`detect_asr_anomalies`), empty outputs;
5. **RTF** on CPU int8 4 threads (sum decode s / sum audio s) for the fine-tuned model and the p1 re-decodes, on the
   same machine; G4 relevance (wait-after-close <= 10 s for a 2-minute session) is discussed from it, not measured.

Uncertainty: speaker-cluster percentile bootstrap (5000 resamples, seed 20261108), paired deltas FT - p0 and FT - p1.

## 6. Expectation (author, before running; changes no rule)

In-domain training (same corpus, channel and task vocabulary) should lower WER and raise survival: survival 0.65-0.90
(p0 about 0.49, p1 about 0.67 on the O1 frame), filler-token recall 0.6-0.9 (p0 about 0.4), WER vs the verbatim target
3-10 points lower than p0. Risks: false fillers, loops, forgetting outside DIHANA. The smoke run may change little.
"Worth pursuing" reading (descriptive, not a claim): survival FT - p1 CI lower bound > 0, filler recall FT - p0 > 0 and
WER vs text A not more than 2 points worse than p0. Any gain is **in-domain** and says nothing about other corpora,
presentations or NOCANDO.

## 7. Licences

- `openai/whisper-small`: MIT. DIHANA: Creative Commons Attribution 4.0 International (Zenodo record 5572470; local copy
  of the record page `01_DIHANA/documentacion/pagina_oficial.html`, field `rights`: `cc-by-4.0`); CC BY 4.0 permits
  reuse, adaptation (derivative models) and publication for any purpose with attribution. It covers copyright only, not
  data protection: the recordings are voices of identifiable participants, so the weights stay outside the repository
  and are not published by this plan.
- No transcript, audio or per-utterance text enters the repository; only aggregates (`RESULTS_B.md`, `RESULTS_B.json`).
