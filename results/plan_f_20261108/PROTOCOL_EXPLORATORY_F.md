# Exploratory protocol F: fine-tuned verbatim ASR profile `ft_small_v1` + decision profile `o2026_13` (2026-11-08)

Status: **exploratory ("Plan F"), DIHANA folds 1-4 validation speakers only. This is NOT the confirmatory protocol of
`results/confirmatoria_20261107`** (that cell is spent and is not used here; nothing in this plan is a confirmatory claim).
Written and committed BEFORE any decode, feature extraction, fit or metric of this plan. The sha256 of this file (CRLF
normalised to LF) is recorded in every result JSON of `results/plan_f_20261108/`. Any later change is a dated annex with
its own hash; this file is not edited after the commit.

## 0. Why

Plan B (`results/plan_b_20261108`, exploratory) fine-tuned `openai/whisper-small` (MIT) on DIHANA human transcripts only
and measured, on 36 held-out speakers, 26/30 literal repetitions kept (p0 14/30, p1 21/30) and filler-token recall 0.864
(p0 0.494, p1 0.256), at a CPU int8 speed no worse than p1. Plan A (`results/plan_a_20261108`) showed that a P1 single
pass (`o2026_12`) with a demotion layer did not beat the plain lexical rule and reached G4 at 9.885 s median over only
two measured rounds (limit 10 s). Plan F asks the next question: if the Plan B model is made an opt-in MRCD ASR profile and
the repetition decision is the plain primary lexical rule on its words (no demotion layer), is the decision better than the
same rule on P0 or P1 words, are fillers read from its lexical tokens at least as good as `o2026_11`, and does the
profile pass the G4 latency gate with a stricter measurement (1 cold + 5 measured rounds)?

## 1. Candidate under evaluation (opt-in; the default stays `o2026_11`)

- ASR profile `ft_small_v1`: CTranslate2 (faster-whisper) backend, the Plan B step-200 checkpoint converted with
  `ct2-transformers-converter --quantization int8` (best checkpoint by dev loss, chosen in Plan B before any val metric),
  loaded from the directory named by the environment variable `MRCD_FT_CT2_DIR`; CPU int8; decoding exactly as Plan B
  section 5: no `initial_prompt`, `beam_size=5`, `temperature=0`, `condition_on_previous_text=False`,
  `word_timestamps=True`, `vad_filter=False`, `suppress_tokens=[]`, 8 s chunks with 2 s overlap, `OMP_NUM_THREADS=4`.
- Decision profile `o2026_13`: ONE ASR pass with `ft_small_v1` feeds every class.
  - repetition = the **primary lexical rule**: the generator of `mrcd.service.contextual` (`norm_tokens` of every ASR
    word, non-lexical filled pauses removed, every immediate exact repetition of 1-4 tokens is an event, interval =
    second copy), **with no demotion layer** (all four rules of `o2026_12` are off). It is the same rule as `p0_rule`
    and `p1_rule` of Plan A.
  - filler_word = the lexical filler tokens of the ASR output, i.e. the non-lexical filled-pause tokens the model was
    trained to write (Plan B convention: DIHANA `<f tipo=a|e|m|sin_identificar/>` -> `ah | eh | mm | eh`). Recognised by the
    regex `^(e+h*|eh+|ehm+|em+|mm+|hm+|ah+|aa+h*)$` on `norm_tokens` (the engine's `NONLEXICAL_FILLER_RE`, the same
    predicate Plan B used to score filler recall), with the conjunction-`e` exception (`e` before a token starting `i` or
    `hi` is not a filler). Each such word is one `filler_word` event (subclass `filled_pause`).
  - prolongation = the shipped `o2026_11` head, **unchanged and not refitted**. It reads engine features and ASR word
    timing/text, so its behaviour may change with the words of another ASR (declared; measured descriptively, section 4).

## 2. Evaluation set (and what it is NOT)

- ONLY the **36 Plan B validation speakers, 983 turns** of DIHANA folds 1-4 (`scripts/verbatim_ft/vtext.make_split`,
  seed `20261108`, same rule and same pool as Plan B; the id lists are recomputed, never copied from another worktree,
  and their counts must equal Plan B's: 36 speakers, 983 turns). The FT model saw 129 train speakers and 15 dev speakers;
  it never saw these 36.
- **It is not an untouched cell.** The Plan B val metrics of the FT model (survival 26/30, filler-token recall 0.864) and
  of p0/p1 were already seen before this protocol was written. No checkpoint, threshold or rule is selected or tuned on
  them here.
- **o2026_11 is in-sample on this set and biased in its favour:** its pinned filler and prolongation heads were fitted on
  all of DIHANA folds 1-4 (5056 turns, 180 speakers), including these 36 speakers. Any o2026_11 figure of this plan that
  comes from the shipped head is in-sample. A second, out-of-fold reading (the nested-CV predictions of the O-campaign) is
  reported next to it for the filler class; it is less biased but its configuration was still selected with all folds in
  view. V0 is the frozen engine checkpoint (its provenance is documented with the checkpoint and not re-audited here).
- Never read, decoded or scored: DIHANA fold 5, any NOCANDO recording (including the 19 unlabelled mp3), AMERESCO, EPIC 136,
  CHM150. The FT model is never run on the frozen 1025-unit frame or on any folds 1-4 cross-validation of MRCD heads (it
  saw those speakers). No cw2 output is used as a label. The G4 clip (section 5) is the S8 benchmark recording (a YouTube
  seed clip), not any of the forbidden corpora.
- Reference per turn: repetition = `ref.rep_literal` of the frozen pool (1 / 0; `None` = indeterminate, excluded);
  filler = `filler_present` (all DIHANA `f` marks, i.e. every non-lexical filled pause); prolongation =
  `lengthening_present`.

## 3. Systems (turn = unit; a turn is positive if the system emits >= 1 event of the class)

Repetition (fixed rules, nothing is fitted): `ft_rule` (generator on FT words), `p1_rule` (same on P1 words), `p0_rule`
(same on P0 words), `V0` (engine argmax windows on P0 features, the product's repetition). All word sets are the
`transcribe_chunked` output of the respective profile on the same audio.

Filler: `ft_filler` (FT filler tokens, section 1) vs `o2026_11` (shipped pinned head, turn-level records built from P0
words and P0 engine features, the same unit convention the head was fitted with; in-sample, section 2) and its
out-of-fold reading `o2026_11_oof`. Descriptive text-only comparators with the same token predicate: `p0_text`, `p1_text`.

Prolongation (descriptive, not in the Holm family): the shipped `o2026_11` head on P0 words/features (`o2026_11`, in-sample)
vs the same head, unchanged, on FT words and engine features recomputed with FT words (`o2026_13`).

Word sources and provable reuse (read-only; nothing of another plan is modified):

- P0 words: the frozen campaign cache (`results/mejoras_20261030/asr_cache/dihana`); P0 engine features: Plan A's
  `feat_p0` cache (recomputed from audio with the service path).
- P1 words: the frozen ASR-recovery decodes (1025 turns) and the Plan A decodes (4031 turns), both produced with
  `asr_recovery.decode.transcribe_chunked(asr, audio, "P1")`, CPU int8, `OMP_NUM_THREADS=4`, `ctranslate2` seed 13 per turn.
  They are reused only if, on the val turns that Plan B also decoded with P1 (same code and settings), the word text is
  identical for every turn they have in common; any mismatch is reported and the affected source is re-decoded.
- FT words: **decoded fresh** by this plan through the `ft_small_v1` profile (CPU int8, 4 threads, seed 13 per turn,
  resumable cache under `results/plan_f_20261108/cache/`, untracked). Plan B's FT_best decodes of the same turns are used
  only as an identity check of the profile path (word text equal on every turn is expected; differences are reported, not
  hidden). The primary results use the fresh decodes.

## 4. Metrics and statistics

Per system and class: precision, recall, F1 with TP/FP/FN/TN. **Speaker-cluster percentile bootstrap, 5000 resamples,
seed 13, 95 % CI**, paired deltas on the same resamples (the machinery of `scripts/plan_a/evaluate.py`: `Boot`, `describe`,
`paired`). **False-filler rate per 100 words** = filled-pause tokens emitted per 100 reference words (text A) on turns
without any `f` mark, pooled, speaker-bootstrap CI, for `ft_text`, `p0_text`, `p1_text` (Plan B definition; `o2026_11` emits
turn decisions, not tokens: reported `not_available` in that unit, with its turn-level false-positive rate on reference
negative turns instead).

Holm family (3 hypotheses; Holm-adjusted p < 0.05 AND the stated CI bound):

- **HF1** repetition F1 `ft_rule` - `p1_rule` > 0 (two-sided bootstrap p; supported iff CI lower > 0 and p_Holm < 0.05).
- **HF2** repetition F1 `ft_rule` - `p0_rule` > 0 (same criterion).
- **HF3** filler **recall** `ft_filler` - `o2026_11` non-inferiority, margin **0.05**: supported iff CI lower > -0.05 and the
  one-sided bootstrap p of H0 "delta <= -0.05" (floor 1/(B+1)) is < 0.05 after Holm.

Pre-declared secondary readings (outside the family, descriptive): filler F1 delta `ft_filler` - `o2026_11` with the
non-inferiority rule delta >= -0.02 and CI upper >= 0 (same form as gate GF of Plan A; recall-only non-inferiority can be met
by over-flagging, so precision, F1 and the false-filler rate decide how to read HF3); the same against `o2026_11_oof`;
prolongation F1 delta (o2026_13 head on FT words) - (o2026_11 on P0), reported as "degraded" if the point delta < -0.02;
per-system repetition precision/recall; `V0` repetition for context; counts of repetition candidates per system.

Absence is never 0: any quantity not measured is `not_available` with a reason.

## 5. Latency gate G4 (stricter than Plan A)

Same procedure and metric as Plan A (`scripts/plan_a/g4_o2026_12.py` / `scripts/mejora_servicio/g4_benchmark.py`): a real
service process (`mrcd.service serve`, test port 8791, `--incremental --device cpu --preload p0,ft_small_v1 --warmup`,
`OMP_NUM_THREADS=4`, `HF_HUB_OFFLINE=1`), the same 120 s recording of the S8 benchmark (the same seed clip as Plan A's G4),
real-time pace (2 s batches), metric = wait after close (seconds from closing the incremental session to the completed
document), limit **10 s**. Per profile: **1 cold round discarded + 5 measured rounds**; report each round, the median and
the maximum of the 5. **G4 passes only if the median AND the maximum of the measured rounds are <= 10 s** (and every
document completed and equals the offline document). Profiles measured in the same process: `o2026_13` and, as
same-session reference, `o2026_11`. Background CPU load is recorded before every round (flagged if >= 20 %, never
excluded). Offline RTF reported. Measured after all decoding is finished, one heavy job at a time. No claim beyond this machine.

## 6. Expectations written before any run (changes no rule)

| quantity                                            | expected                                                                |
| --------------------------------------------------- | ----------------------------------------------------------------------- |
| repetition F1 `ft_rule` (30 positive turns)         | [0.55, 0.85]                                                            |
| repetition F1 `p1_rule`                             | [0.45, 0.75]                                                            |
| repetition F1 `p0_rule`                             | [0.35, 0.65]                                                            |
| HF2                                                 | supported                                                               |
| HF1                                                 | point delta > 0 but likely NOT resolved (30 positives; wide CI)         |
| filler recall `ft_filler` vs `o2026_11` (in-sample) | [0.80, 0.95] vs [0.60, 0.85]; HF3 supported                             |
| filler precision `ft_filler`                        | [0.60, 0.90]; false-filler rate FT above p0/p1 (Plan B: 0.171 vs 0.086) |
| prolongation F1 on FT words vs P0                   | delta in [-0.06, +0.02]; may be "degraded"                              |
| G4 `o2026_13`                                       | median [8, 10] s; max-of-5 <= 10 s NOT assured (Plan A measured 9.89 s) |

**What counts as failure:** HF2 unsupported (the FT words do not beat plain P0 + rule); G4 verdict "fail" (median or max
of the 5 measured rounds above 10 s); filler HF3 unsupported or filler F1 delta below -0.02 with CI upper < 0; prolongation
reported "degraded". Any of these is reported as a negative result. HF1 unresolved is reported as "not resolved", not as
evidence of equality. `o2026_13` stays opt-in whatever the outcome; adopting it as default needs a new protocol and an
untouched evaluation cell.

## 7. No tuning, deviations, public safety

- No threshold, token set, regex, margin, rule or checkpoint is changed after any result is seen. Any deviation (a crash
  that forces a re-run, an environment difference, a skipped step) goes into a dated annex with its own hash.
- Tracked files contain aggregates only: no transcript, per-turn text, speaker or unit id, audio, weights or absolute local
  path. Caches stay under the git-ignored `results/` tree. The model weights are never committed or published
  (DIHANA is CC BY 4.0; it does not address data protection of the recorded voices).

## 8. What this cannot show

One corpus (telephone, peninsular, train-information domain) for training and evaluation; the speakers are disjoint but the
domain, channel and vocabulary are shared, so every FT gain is an in-domain upper bound and says nothing about
presentations, other corpora or non-clinical settings; 30 repetition positives and 146 filler turns give wide intervals;
turn-level scoring (no timing accuracy); the filler reference rewards emitting any non-lexical filler, not its form; scores
are not calibrated probabilities; the G4 clip is one non-DIHANA recording on one machine; `environment_is_pinned` remains
False. No confirmatory claim of any kind.
