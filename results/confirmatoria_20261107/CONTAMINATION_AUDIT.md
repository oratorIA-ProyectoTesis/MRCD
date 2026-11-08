# Contamination audit of every labelled corpus (2026-11-07)

Purpose: decide which cells can still carry a CONFIRMATORY figure. A confirmatory cell needs (a) the protocol frozen before,
(b) data never consulted for any decision, (c) a single run. This audit classifies every labelled corpus x partition x class.

How it was built: reading the repository at `6da2d43` (scripts, protocols, reports; evidence is `path:line` of the committed
tree unless marked LOCAL). No corpus label was read for this audit. The only data-side look was counts of rows of
`12_AMERESCO/source_annotation_index.csv` by recording and duration (labels column not opened), file durations/sample rates
of the NOCANDO and AMERESCO audio (`soundfile.info`), the number of intervals of the NOCANDO TextGrids (text not printed; protocol section 10) and file listings. Everything else comes from earlier reports.

Status vocabulary (precise):

* **used for decisions**: the cell fed a fit, a selection, a threshold or a go/no-go; cannot be confirmatory, ever.
* **read for reporting, several times**: scored by frozen systems more than once; no declared selection on it, but it is
  historical (a figure on it is a replication of a known number).
* **read once for reporting**: scored once with frozen rules, nothing tuned on it; a second scoring is no longer first-look.
* **counts only**: only marginal counts/inventories were computed; no system output was ever compared with its labels.
* **untouched**: no label, no count of the relevant mark, no system output.

## 1. Table

| # | Corpus | Partition / bucket | Class (reference) | Experiments that read it (evidence) | Status |
|---|---|---|---|---|---|
| 1 | DIHANA | folds 1-3 (dev) | filler `f`, prolongation `a`, repetition `r`, revision, pauses, DM | `scripts/baselines/dihana_prepare.py:9` (declared dev); `scripts/improve/select_dev.py:45-48`; `scripts/improve/train_head.py:3`; `mrcd/context_eval/pool.py:12`; `scripts/asr_recovery/*` (4 commits `bfc6852..7eca778`); `scripts/rep_rev/run_experiment.py`; `scripts/s3/s3_common.py:1`; `scripts/o_campaign/run_campaign.py` (nested CV); `scripts/mejora_servicio/common.py:3`; `scripts/control_noprompt/decode_pnone.py`; `scripts/asr_survival_models/decode_*.py`; `scripts/f1_context/build_table.py` | used for decisions |
| 2 | DIHANA | fold 4 (tuning/selection) | same | `scripts/baselines/dihana_prepare.py:9`; `scripts/improve/select_dev.py:48`; `scripts/improve/final_eval.py` (selection of Vb/Vc); pool folds 1-4 in all the experiments of row 1 | used for decisions |
| 3 | DIHANA | **fold 5** | filler, prolongation, repetition (mark), revision; cross-class matrix (all classes) | scored in `results/baselines_20261030` (`scripts/baselines/evaluate_audio.py:5`, commit `fec7968`); again for V0..Vac in `scripts/improve/final_eval.py:1,335,376` (`fec7968`/`cbc6ec2`); regression read `scripts/live_parity/rerun_comparisons.py:6-8,235`; `scripts/o_campaign/run_fold5_historical.py:1,24` via `mrcd/optimize/data.py:136` (o2026_11 pipelines, `db28004..c86cdce`); copied in `scripts/rep_rev/historical_fold5.py:4,17`; support counts in `scripts/audit2/b1_dihana_rep_rev_dev.py:154,488`. Declared historical: `results/cierre_oe4_reducido_20261031/INFORME_OE4_REDUCIDO.md:62` | read for reporting, several times (HISTORICAL, contaminated; `ALLOWED_FOLDS == (1,2,3,4)` pinned in `mrcd/context_eval/pool.py:12`) |
| 4 | DIHANA | all folds, task T (text) | repetition sub-types, revision | `scripts/audit2/b1_dihana_rep_rev_dev.py`; `scripts/rep_rev/*`; `scripts/s3/*` (folds 1-4; fold 5 counts only) | used for decisions (folds 1-4); fold 5 as row 3 |
| 5 | disfluentES | buckets 0-2 (`sha256("MRCD-bl-13\|"+speaker) % 5`) dev, 3 tune | repetition `REP*`, filler `FILL*` (text, children reading) | `scripts/baselines/text_baselines.py:2,30-32,76,102-151` | read for reporting; the filler rule "lexicon + bare `e`" was informed by inspecting this corpus (`INFORME_OE4_REDUCIDO.md:126`) so it is used for decisions on the filler rule |
| 6 | disfluentES | **bucket 4 (test)** | repetition, filler (text only, **no audio**) | `scripts/baselines/text_baselines.py:76,270` prints the test partition; `scripts/audit/audit_a1_disfluentes.py` and `scripts/audit2/b3_other_corpora_rep_rev.py:1` inspected the whole corpus (counts) | read once for reporting (rep F1 0.73 [0.65,0.79], `INFORME_OE4_REDUCIDO.md:125`); no audio exists, so it cannot test the audio engine in any case |
| 7 | ES-Port | buckets 0-4 (group = dialogue) | repetition U false start (joint `< >`), text only | `scripts/baselines/text_baselines.py:203-244,270`; `scripts/audit/audit_a2_esport.py`; `scripts/audit2/b3_other_corpora_rep_rev.py` | bucket 4 read once for reporting (`INFORME_OE4_REDUCIDO.md:127`); no audio; joint label |
| 8 | AMERESCO | 98-segment pool (3-8 s, no exclusion reason; 37 pos / 61 neg) | prolongation `<alargamiento/>` | `scripts/baselines/ameresco_prepare.py:1-20`, `evaluate_audio.py:174`; `scripts/improve/final_eval.py:21`; `scripts/o_campaign/run_campaign.py:356`; `scripts/live_parity/rerun_comparisons.py:3,51`, `margin_study_v2.py:6`; `scripts/cierre3b/prepare_units.py:3` (cost/ASR) | read for reporting, several times |
| 9 | AMERESCO | segments OUTSIDE the pool of the SAME 4 conversations / 11 tiers (<3 s: 2090 rows; >8 s: 17; both counts include rows that also carry an automatic exclusion reason; plus 156 rows of 3-8 s with an exclusion reason) | prolongation `<alargamiento/>` | marginal counts only (`scripts/audit/audit_a3_ameresco.py`, `scripts/audit2/b3_other_corpora_rep_rev.py:153`); no system was ever scored on these segments (pool definition `ameresco_prepare.py:18`) | counts only at segment level; **conversation- and speaker-level contaminated** (the same recordings fed rows 8 and the pilot-32 clips) |
| 10 | AMERESCO | all | repetition, filler | no label exists (repetitions transcribed verbatim, no tag) | no reference |
| 11 | EPIC v2 (Spanish originals) | 136-sentence set (78/24/34 by 29 speakers) and v4 dev 184 | filler `FPAUSE` | `scripts/improve/final_eval.py:20`; `scripts/improve/train_head.py` (never loaded, comment `:3`) and the v2/v3 abstention study (`MATRIZ_CORPUS.md` row "EPIC 136": "ALREADY USED in development"); `scripts/o_campaign/run_campaign.py:356`; `scripts/baselines/epic_prepare.py` | used for decisions (v2/v3, abstention) / read for reporting |
| 12 | EPIC v2 | v4 external diagnosis (42 sentences, 10 speakers, 7 positives from one actor) | filler | `results/auditoria_etiquetas_20261030/EPIC_INVENTARIO.md:37,56` (LOCAL) | read once for reporting; 1 unused positive left in those speakers |
| 13 | EPIC v2 | the other 1307 sentences (74 `FPAUSE` sentences, 22 videos; 28+27 in two actors) | filler | `scripts/audit2/b4_epic_inventory.py`, `b4b_epic_unused_subset.py` (counts per actor; LOCAL `epic_unused_subset.json`) | counts only at sentence level; **speaker-level contaminated**: all 39 speakers were used (`EPIC_INVENTARIO.md:56`: `actors_never_in_any_set = 0`); positives concentrated in 2 speakers |
| 14 | EPIC v2 | all | pauses `SPAUSE`, truncation | `scripts/s6_video/s6_01_units.py:1` (video units), audit scripts | audit only; no class mapped (pause presence, not function) |
| 15 | EPIC-UdS | all | none (normalised text, no marks) | `MATRIZ_CORPUS.md:123-135` | no reference |
| 16 | CHM150 | all 2663 utterances, 150 speakers | **repetition (literal, transcript-derived)** | `scripts/external_chm150/s0_reference.py`, `s1_run.py`, `s2_eval.py` (commit `6c68ae7`; protocol frozen 2026-11-02, `results/external_chm150_20261102/PROTOCOL.md`); `scripts/asr_survival_robustness/make_chm_frame.py:3` + `decode_chm150.py` (500-utterance sub-frame, `82e4a02`/`6da2d43`); `scripts/mejora_servicio/c1_chm150.py` (LOCAL, uncommitted in the main checkout; no output found) | **read once as a confirmatory test, then re-read for the survival robustness study** -> repetition cell is spent |
| 17 | CHM150 | all | filled pauses `e`/`em` (transcript tokens) | descriptive counts only: `results/external_chm150_20261102/README_CHM150.md` ("em 9, e 118 ... Not evaluated"); `mrcd/context_eval/transcript_reference.py:1-24` | counts only, no system compared; BUT utterances are 1.85 s median (all < 13 s = service minimum), reference is weak (`e` ambiguous) and the cell has been seen alongside the spent repetition cell; not used here |
| 18 | CHM150 | all | prolongation, revision, `este` | not marked in the transcription | no reference |
| 19 | NOCANDO (Spanish) | 14 aligned recordings (TextGrid), 5 speakers; 33 mp3 / 11 speakers total | repetition (literal, transcript-derived), filled pauses `hm`/`eh` (transcript tokens) | `scripts/audit/audit_a6_nocando.py:51-59` computed segment counts of `hm\|eh\|em\|mm` and `_`/`#` marks over the transcripts (published: 165 segments / 175 tokens; LOCAL `verified/a6_nocando.json`); `audit_a9_build_matrix.py`, `audit_a10_render_md.py` (matrix text). No system was ever run on NOCANDO audio; repetition was never derived | counts only (filled-pause token marks); **repetition cell untouched**. Excluded as reference by the earlier audit ("no disfluency labels"), which concerns annotation, not the transcription tokens used here |
| 20 | DIME | 56 files, 20 dialogues | none (XLS are dialogue acts; repair labels HTTP 403) | `scripts/rep_rev/utt74_check.py` (one exact example), importers | engineering only |
| 21 | ESLORA | public text | none; no audio | `mrcd/corpora/eslora.py` | no audio |
| 22 | Spanish in Texas, NOCANDO TSA, SOPHIAS, Bangor, PRAUTOCAL, PUCP, others | - | no usable label / not obtained | `corpora_manifest.csv` | no reference / not available |
| 23 | Internal YouTube clips (87) | - | automatic labels (weak supervision) and VLM-judge `gold_llm` | `scripts/data_mining.py`, `scripts/train_checkpoints.py:2`, `scripts/run_ablation.py:2`, `scripts/llm_judge.py:2` | training source of the checkpoint; judge labels are not human |

## 2. Reading by class (audio engine)

| Class | Human-labelled audio cells | Untouched cell? |
|---|---|---|
| filler_word | DIHANA 1-5, EPIC (all 39 speakers used), CHM150 `e`/`em` (transcript tokens, counts only) and NOCANDO `hm`/`eh` (transcript tokens, counts only) | **Weak yes**: NOCANDO `hm`/`eh` (transcript tokens, counts known). Recall-only: absence of a transcribed filled pause is not a verified negative |
| prolongation | DIHANA 1-5, AMERESCO | **No.** The only other human cell is AMERESCO outside the pool, which shares recordings and speakers with the spent pool (row 9); NOCANDO `_` has no documented meaning (`MATRIZ_CORPUS.md:137-150`) |
| repetition (literal) | DIHANA 1-5, CHM150 (spent), NOCANDO (transcript-derived) | **Yes, with a circular reference**: NOCANDO transcript-derived literal repetition (never derived, never scored) |
| revision, block, rhetorical_pause, neutral_pause | DIHANA (revision 58 % silent edits); none for the others | No (out of the evaluated scope, `INFORME_OE4_REDUCIDO.md:20-30`) |

Nothing human-labelled for prolongation is left that is both audio and speaker-new. Nothing confirmatory is possible for
revision/block/pauses. Neither of the two weak untouched cells (NOCANDO) is a human disfluency annotation: both are
transcription-derived (see the protocol for what they can and cannot claim).

## 3. Residual risk of the "untouched" label for NOCANDO

* Counts of `hm|eh|em|mm` tokens and of `_`/`#` marks were published (segment level, all 40 transcripts including 26 without
  audio/alignment). The filled-pause sample size is therefore known in order of magnitude; the specific positives and the
  repetition reference are not known.
* The transcripts were opened by the earlier audit script only through regexes for those counts. No MRCD output was ever
  compared with them.
* NOCANDO was catalogued as "EXCLUDE as reference" (`MATRIZ_CORPUS.md:137-150`) because it has no disfluency scheme. That
  decision was about annotation, not about the evidence used here (token-level transcript reference, declared as such).
* The author of this audit and of the protocol did not read any NOCANDO transcript text or audio sample: only durations and
  sample rates, the encoding header bytes of the TextGrids and, to validate the parser, the count of non-empty first-tier
  intervals (1 531 over the 14 files) and the last interval end time; no interval text was printed (protocol section 10).
