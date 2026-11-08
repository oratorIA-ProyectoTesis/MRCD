# Plan B results: verbatim fine-tune of Whisper small (EXPLORATORY, DIHANA folds 1-4)

Protocol: `PROTOCOL_EXPLORATORY_B.md` (committed in fffdb1f before any training or decode). Aggregates: `RESULTS_B.json`.
Code: `scripts/verbatim_ft/`. Every figure is **exploratory (DIHANA folds 1-4)**, measured on the 983 val turns of 36
speakers held out from training (speaker-independent); 95 % CIs are speaker-cluster bootstrap intervals (5000
resamples, seed 20261108). Labels were the human transcripts only; no CrisperWhisper output was read by any Plan B script.

## Headline

- The fine-tuned model (best checkpoint by dev loss = step 200 of the capped run, CTranslate2 int8, CPU 4 threads, p0
  pipeline without prompt) keeps **26 / 30 literal repetitions (0.867 [0.759, 1.00])** against 14 / 30 for p0 (0.467
  [0.292, 0.684]) and 21 / 30 for p1 (0.700 [0.553, 0.889]): FT - p1 **+0.167 [+0.056, +0.269]**, FT - p0 +0.400 [+0.233, +0.560].
- It also keeps filled pauses, which p1 trades away: **filler-token recall 0.864 [0.810, 0.913]** (152 / 176) vs p0 0.494
  [0.395, 0.583] and p1 0.256 [0.199, 0.323]; FT - p0 +0.369 [+0.281, +0.468].
- WER falls: vs text A 0.052 [0.041, 0.063] (p0 0.133, p1 0.111; FT - p0 -0.081 [-0.118, -0.057]); vs the verbatim target
  0.034 [0.026, 0.043] (p0 0.131, p1 0.119). This is **in-domain** (same corpus, channel and task vocabulary as training)
  and says nothing about other speech.
- Cost: false fillers on turns without a `<f>` tag 0.171 per 100 words [0.071, 0.277] vs 0.086 for p0 (FT - p0 +0.086
  [-0.033, +0.201], not resolved); repetition-loop flags 0 (p0 1, p1 0); empty outputs 4 / 983 (p0 3, p1 2).
- The pre-declared "worth pursuing" reading (survival FT - p1 CI lower > 0, filler recall FT - p0 > 0, WER vs text A at
  most 2 points worse than p0) holds, in-domain only.

## All arms (val, 983 turns)

| arm                                | survival (30)           | filler-token recall (176) | filler turn survival (146) | WER vs text A            | WER vs verbatim target | false fillers /100 w | loops |
| ---------------------------------- | ----------------------- | ------------------------- | -------------------------- | ------------------------ | ---------------------- | -------------------- | ----- |
| p0 (frozen cache)                  | 0.467 [0.292, 0.684]    | 0.494 [0.395, 0.583]      | 0.534                      | 0.133 [0.107, 0.169]     | 0.131                  | 0.086                | 1     |
| p1 (156 cached + 827 decoded)      | 0.700 [0.553, 0.889]    | 0.256 [0.199, 0.323]      | 0.301                      | 0.111 [0.097, 0.124]     | 0.119                  | 0.086                | 0     |
| FT_smoke (200 steps)               | 0.833 [0.688, 0.964]    | 0.864 [0.811, 0.907]      | 0.911                      | 0.080 [0.043, 0.143]     | 0.062                  | 0.120                | 1     |
| **FT_best** (capped run, step 200) | **0.867 [0.759, 1.00]** | **0.864 [0.810, 0.913]**  | 0.897                      | **0.052 [0.041, 0.063]** | **0.034**              | 0.171                | 0     |

p1 - p0 on this val set reproduces the known trade-off: survival +0.233 [+0.115, +0.370], filler recall -0.239
[-0.314, -0.151]. FT_smoke - p1 survival +0.133 [-0.042, +0.273] (not resolved); FT_smoke has one loop-flagged turn, a
plausible but unchecked cause of its wider WER interval.

## Training (GPU: RTX 5060 Laptop 8 GB, shared; peak 5.3-5.4 GB allocated)

- Smoke: 200 steps (lr decayed to 0 at 200), dev loss 1.133 (base) -> 0.141 (100) -> 0.127 (200); **0.133 GPU h**.
- Capped (fresh from base, 6 epochs = 1362 steps, cap 2 GPU h not reached): dev loss 0.1215 at 200, then rising
  (0.122 / 0.130 / 0.143 / 0.157 / 0.159 / 0.163 at 400-1362) while train loss fell to about 0.002: overfitting after
  about one epoch. Best = step 200 by the pre-declared dev-loss rule; **0.739 GPU h**. Total GPU for Plan B 0.87 h.
- No fallback was needed (micro-batch 8 x 2 accumulation, full fine-tuning, encoder not frozen).
- Converted with `ct2-transformers-converter --quantization int8` (no warnings): `<ft_models_dir>/ct2\
capped_best_int8\model.bin` 248 MB sha256 `6d4e61de...dd1427`; smoke `edf07e07...b7326c`. Weights stay outside the repo.

## CPU speed (int8, OMP 4 threads), G4 relevance

- Quiet probe (`cache/rtf_probe.json`, 100 seeded val turns, run with no other job of this plan): **FT RTF 0.483 vs p1
  0.615**; one synthetic 120 s session decoded offline in 8 s / 2 s chunks: **42.3 s vs 45.1 s**. The FT model has the
  architecture and chunking of p0/p1, so its incremental-mode cost should match theirs; FT avoids the prompt tokens.
- The RTFs of the full val decodes (FT 0.94, p1 1.60) were measured while training and other decodes shared the CPU and
  are not comparable; they are kept in the JSON only.
- **G4 itself (wait-after-close <= 10 s for a 2-minute session, incremental service) is `not_available`**: not measured
  for the FT model in this plan (needs the service run of `results/mejora_servicio_20261102`).

## not_available (with reason)

- p0 RTF: frozen cache from an earlier campaign; not re-decoded.
- Any figure outside DIHANA (NOCANDO, AMERESCO, EPIC, CHM150, fold 5): forbidden by the data rules of this plan;
  generalisation of the fine-tune is therefore unknown.
- An MRCD ASR profile for the FT model: not added (out of scope of this plan; would be a separate opt-in step).

## Limits

One corpus (telephone, peninsular Spanish, train-information domain) for training and evaluation; speakers are disjoint
but domain, channel and vocabulary are shared, so the WER and survival gains are an in-domain upper bound. 30 survival
opportunities and 176 reference fillers: wide intervals. The filler target words are a reconstruction convention
(`<f tipo=e>` -> `eh`, `m` -> `mm`, `a` -> `ah`, unidentified -> `eh`): filler recall rewards emitting any non-lexical filler,
not its exact form. Survival is a transcript heuristic, not a human judgement. One seed, one learning rate, one schedule.

**Evaluation scope.** The fine-tuned model was trained on 129 DIHANA folds 1-4 speakers (train split; a further 15 dev
speakers drove checkpoint selection), so it may only be evaluated on its 36 held-out val speakers or on non-DIHANA data. It must never be run on the
frozen 1025-unit frame, nor on any folds 1-4 cross-validation of the MRCD heads: both contain its training speakers.

**Interval note.** Percentile bootstrap CIs at the ceiling (k = 30 survival opportunities, upper bound 1.00) are
anti-conservative: the resampled proportion cannot exceed 1, so the upper bound is truncated and the interval understates
the uncertainty.

## Licences

- `openai/whisper-small`: MIT. DIHANA: **CC BY 4.0** (Zenodo record 5572470; local copy of the record page
  `corpus-repository/01_DIHANA/documentacion/pagina_oficial.html`, `rights` = `cc-by-4.0`: "allows re-distribution and
  re-use of a licensed work on the condition that the creator is appropriately credited"; `0readme.txt` has no licence
  text). CC BY 4.0 permits training and derivative models, including commercial use and publication, with attribution
  (credit: DIHANA consortium, Universidad de Zaragoza / Universidad del País Vasco / Universitat Politècnica de València,
  project TIC2002-04103-C03). It does not address data protection of the recorded voices; the weights are not published.
- The fine-tuned model never saw a CrisperWhisper output, so it is not a Derivative Work under the nyra health licence.

## Reproduce / resume

```
set PYTHONPATH=<worktree> & set OMP_NUM_THREADS=4 & set HF_HUB_OFFLINE=1
<repo .venv> scripts/verbatim_ft/prepare_data.py
<venv_ft>/python scripts/verbatim_ft/train.py scripts/verbatim_ft/configs/capped.json   (re-run = resume from last/)
<venv_ft>/python scripts/verbatim_ft/convert_ct2.py <ft_models_dir>/capped/best <ft_models_dir>/ct2/capped_best_int8
<repo .venv> scripts/verbatim_ft/decode_val.py FT_best --model <ft_models_dir>/ct2/capped_best_int8 [--shard k/2]
<repo .venv> scripts/verbatim_ft/decode_val.py P1
<repo .venv> scripts/verbatim_ft/rtf_probe.py P1 ; ... rtf_probe.py FT_best --model ...
<repo .venv> scripts/verbatim_ft/evaluate.py FT_smoke FT_best
```
