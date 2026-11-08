# Annotation kit 2026-11-08 (Phase 0.5): what to do next

Goal: a new human reference cell (filled pause, lexical filler, repetition, prolongation, plus the function of discourse markers: functional / disfluent / uncertain, revision 2026-11-08) on NOCANDO audio no system has touched. Primary path: **blind self-annotation** under a declared deviation. Preferred alternative if resources appear: two external annotators (`PROTOCOL_ANNOTATION.md`, EXT).

## Contents

| File                                | Role                                                                                                                                                                                         |
| ----------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ANNEX_20261108_SELF_ANNOTATION.md` | Declared deviation from PREREGISTRO §7 and its controls. Commit first.                                                                                                                       |
| `PROTOCOL_ANNOTATION.md`            | Full protocol (both variants).                                                                                                                                                               |
| `GUIA_ANOTACION.md`                 | Guide for annotators (Spanish) with the calibration checklist.                                                                                                                               |
| `TOOL_RECOMMENDATION.md`            | Audacity label workflow.                                                                                                                                                                     |
| `manifest_nocando_unlabeled.csv`    | 150 items (seed 20261108).                                                                                                                                                                   |
| `manifest_nocando_calibration.csv`  | 20 disjoint calibration items.                                                                                                                                                               |
| `annotation_sheet_template.csv`     | CSV schema (incl. `marker_token`, `marker_function`).                                                                                                                                        |
| `../../scripts/annotation/`         | `sample_items.py`, `cut_clips.py`, `blind_session.py`, `labels_to_csv.py`, `unblind.py`, `agreement.py` (events; `--field marker_function` for the marker layer), `markers.py` (closed list) |

Run from the repo root with `PYTHONPATH` set to the repo root and the project venv (`.venv\Scripts\python.exe`, see `comandos-ejecutar/COMANDOS.MD`).

## Order of work (primary path)

1. **Sign and commit the annex.** Fill the sign-off; commit; record the commit hash here: `________`. Nothing below starts before this.
2. **Licence check.** Confirm the NOCANDO terms at the source page (see status below). Do not publish clips. Optional: decide now whether to drop speaker `maria` (also in the spent cell): `sample_items.py --exclude-speaker maria` redraws the manifest; that is a deliberate author decision and a new committed manifest, never done silently.
3. **Calibration (20 clips), solo check.** A solo author has no second rater, so calibration is a self-consistency check: annotate the 20 calibration clips (blinded like a pass, own seed), wait at least 48 h, re-annotate them with a fresh blind id and order, and run `agreement.py --mode intra`. Stop rule: class-level κ >= 0.60 **and** per-class Dice >= 0.50 (only for classes with at least 5 events; smaller classes are listed as "not assessable"). If it fails, revise the guide, draw 20 new calibration clips (protocol §4) and repeat once. Rater 2 on the 20 clips, if available, replaces the re-pass. Then commit the frozen guide and record its hash. This step (a few days) happens **before** pass 1; the 7-day gap below applies only between the two main passes.
   **Then freeze protocol G, stage 1** (hypotheses, thresholds, gates, statistics; `run_protocol_g.py freeze-stage1`, commit with `git add -f`) and register it on OSF (`results/protocol_g_draft/OSF_PREREGISTRATION_CHECKLIST.md`). This must happen **before** pass 1 starts.
4. **Cut clips locally, pass 1.** Set `MRCD_NOCANDO_ROOT` (the `02_NOCANDO_ES` folder) or pass `--root`. Per pass:
   - `python -m scripts.annotation.blind_session --manifest docs/annotation_20261108/manifest_nocando_unlabeled.csv --seed <PASS_SEED> --key-out <LOCAL>/key_pass1.csv --list-out <LOCAL>/list_pass1.csv` (pass 2: add `--prev-key <LOCAL>/key_pass1.csv`)
   - `python -m scripts.annotation.cut_clips --manifest <LOCAL>/key_pass1.csv --out-dir <LOCAL>/clips_pass1` (needs ffmpeg; aborts if a source mp3 does not match the manifest sha256; folder must be outside the repo)
   - **Back up the key file outside the repo** (second disk or private cloud). Losing it makes the CSVs unusable.
   - Annotate in Audacity (`TOOL_RECOMMENDATION.md`), export one label file per clip.
   - `labels_to_csv.py`, then `unblind.py --write-hash`; commit the CSV and its `.sha256`.
5. **Wait at least 7 days.** Do not look at pass 1.
6. **Pass 2.** Same as step 4 with a different `--seed` and `--prev-key`, new key, new clip folder.
7. **Intra-rater reliability.** Unblind both CSVs to original ids, then `python -m scripts.annotation.agreement pass1_orig.csv pass2_orig.csv --mode intra` (also with `--tol 0.1` and `--tol 0.5`) and, for the marker layer, `... --mode intra --field marker_function` (not available below 30 matched spans). Save the JSON and commit it **before adjudicating**.
8. **Rater 2** (native speaker, unpaid, never exposed to system output) on at least 40 items. Draw them first: `python -m scripts.annotation.draw_subset --manifest docs/annotation_20261108/manifest_nocando_unlabeled.csv --seed <DECLARED_SEED> --n 40 --out docs/annotation_20261108/rater2_subset_ids.csv` (declare the seed in the annex before drawing). Blind them with `blind_session.py --only-ids rater2_subset_ids.csv`. Then compute `agreement.py pass1_orig.csv rater2_orig.csv --mode subset` and `agreement.py pass2_orig.csv rater2_orig.csv --mode subset`. **Never against the adjudicated file** (circular). Report both as is.
9. **Adjudicate your own disagreements** with the written rules (protocol §7), never looking at system output. Keep raw CSVs untouched; write a separate adjudicated CSV with rule numbers in `note`.
10. **Commit hashed CSVs** (pass 1, pass 2, rater 2, adjudicated, agreement reports).
11. **Freeze protocol G, stage 2** (harness code, `SYSTEM_FREEZE_G.json`, the OD-4 decision; stage 1 was frozen after step 3) and commit it. **Only then** may any system touch the cell, once.

## Time estimate

150 items × ~8 s is about 20 min of audio. At ~10× real time: about 3.3 h per pass, about 4 h with breaks. Calibration about 0.5 h plus the 0.5 h re-pass. Two passes about 8 h, rater 2 about 1 h (40 items), adjudication about 1 h, computation and freeze about 1 h. Discourse-marker function layer (labelled in the same passes, rater-2 subset and calibration): about +2 h. Total about 14 h of author time (0.5 + 0.5 + 8 + 1 + 1 + 1 + 2), over at least 2-3 weeks because of the 48 h and 7-day gaps. Use sessions of at most 1 h to limit fatigue drift.

## NOCANDO licence status

- Recorded as **CC BY-NC-SA 3.0**: `THIRD_PARTY_NOTICES.md:15` ("NOCANDO (Spanish recordings) | as downloaded 2026-10 | CC BY-NC-SA 3.0 | https://recursos-glif.upf.edu/nocando/recordingscorpus?field_language_value=spanish | obtained | no"); `docs/corpora/README.md:29` ("NOCANDO | CC BY-NC-SA 3.0 | obtained (audit only)"); `results/auditoria_corpus_20261030/MATRIZ_CORPUS.md:147` ("Licence: CC BY-NC-SA 3.0").
- The local corpus folder contains no licence or terms file (`02_NOCANDO_ES/` has `ESTADO.md`, `FUENTE.url`, `audio/`, `alineamientos/`, `anotaciones/`); the licence text was **not** read from a primary source here. Note that `docs/corpora/README.md:29` says "audit only": using the audio for a new annotation campaign extends the previously recorded use, so confirm at the source page that non-commercial research annotation is covered.
- Internal 87-clip YouTube corpus: licence "no declarada" (`MATRIZ_CORPUS.md:217`), so no second manifest was produced.

## Known limits of the sample

- 19 recordings, 7 speaker labels, one narration task, no dialect stratification. Item counts per speaker are unequal (proportional to duration; `cristina-mem` has 44 items, `jessica3-mem` and `maria` 5 each).
- Speaker `maria` also appears in the spent cell; `enrique2` vs `enrique` identity is unknown. Do not claim speaker independence from the spent cell.
- Sampling is uniform in time, not by event: rare classes may end up with fewer than the 20 events needed for per-class statistics.
