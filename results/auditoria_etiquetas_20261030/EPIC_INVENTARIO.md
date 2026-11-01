# EPIC v2 local video inventory (2026-10-30)

Read-only on media. New files only. Scripts: `scripts/audit2/b4_epic_inventory.py` (inventory, decode tests, 3-frame face sample), `scripts/audit2/b4b_epic_unused_subset.py` (derived tables). Per-video data: `EPIC_INVENTARIO_por_video.csv` (74 rows, 44 columns incl. sha256). Per-speaker: `EPIC_INVENTARIO_por_hablante.csv` (39 rows). Totals: `epic_inventory_summary.json`, `epic_unused_subset.json`. Actor names present in the corpus metadata are deliberately not copied; only `actor_id`, sex, country and mother tongue.

## 1. Why old reports said 1 video

`corpus-repository/16_EPIC_v2/ESTADO.md` and `video_sample_manifest.json` describe the first sample, `epic_st_20-07-04-p-003-org-es.mp4` (438,301 B, 11.28 s, written 29 Oct 10:43). The other 73 files in `media_selected/` were written later the same day (10:57 to 18:12). The reports were not updated. Verified now: 74 files, 1 at the folder root + 73 in `media_selected/`.

## 2. Technical inventory (74 videos, all Spanish originals `-org-es`)

| Item                                    | Result                                                                                                                                                                                                  |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Location                                | `<CORPORA_ROOT>\16_EPIC_v2\` (root: 1; `media_selected\`: 73)                                                                            |
| Total                                   | 241.4 min, 544,764,769 B (0.545 GB); per video 11.3 s to 1512 s, median 162.5 s                                                                                                                         |
| sha256                                  | computed for all 74, 0 duplicates (column `sha256` in the CSV)                                                                                                                                          |
| Video                                   | H.264 (all); 64 at 384x288, 10 at 352x288; 25 fps; yuv420p                                                                                                                                              |
| Audio                                   | AAC (all), 44.1 kHz, stereo                                                                                                                                                                             |
| Decode tests (ffmpeg, errors to stderr) | first frame OK 74/74; middle frame OK 74/74; last-second frame OK 74/74; full audio decode OK 74/74 (0 errors)                                                                                          |
| Format                                  | all `monologue`; delivery: read-out 33, mixed 24, impromptu 17; 21 events (parliamentary sessions), 2004-02-09 to 2004-07-22                                                                            |
| Speakers                                | 39 `actor_id`; 55 videos male, 19 female; mother tongue Spanish in 74/74 (no L2 speakers); videos per actor 1 to 7                                                                                      |
| Licence                                 | CC BY-NC 4.0: `01_README.html` lines 130-131; `mrcd-codex/MRCD_Plan_Avance_20261030/LICENSE_NOTICES.md` lines 3-5 (non-commercial, attribution kept). Raw media stay out of any redistributable package |

Video decoding is verified on three frames per file and audio fully; the whole video stream was not decoded.

Speaker / speech id: each video is one speech (`source_text_id` = file stem) mapped to `actor_id` through `04_metadata_v2.0.zip`. Speaker identity is the corpus' own metadata; it was not verified by face or voice.

## 3. Coverage of the existing sentence sets

Sentences come from `08_alignments_v2.0.zip` (automatic WebMAUS alignment, not human boundaries). FPAUSE = `-x-` filled-pause notation.

| Set                            | Total                                                          | In local videos                  | Notes                                                                                                                                         |
| ------------------------------ | -------------------------------------------------------------- | -------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| All Spanish original sentences | 2384 (130 speeches)                                            | 1533 (74 speeches, 39 speakers)  |                                                                                                                                               |
| FPAUSE sentences               | 214                                                            | **173** (51 videos, 31 speakers) | 41 FPAUSE sentences are in the 56 speeches without local media                                                                                |
| Codex 136 reference set        | 136 (68 pos / 68 neg, 29 speakers)                             | 136/136, 62 videos               | exact-text match, 0 ambiguous                                                                                                                 |
| v4 development set             | 184 (29 speakers; includes the 136)                            | 184/184                          | `github_upload/repo/experiments/epic_f1_v4_20261029/references.json`                                                                          |
| v4 external diagnosis          | 42 reserved, 41 accepted (10 speakers, 7 pos from one speaker) | 42/42                            | this is the "41 segments" set that `MATRIZ_CORPUS.md` could not locate; provenance is `cohort_inputs.json` + `references.json` in that folder |

Videos with at least one sentence used in an earlier set: 73 of 74 (the exception is the 11 s sample video, 3 sentences, whose speaker is in the development set).

### FPAUSE concentration by speaker (local media, 173 sentences)

| Speaker                       | Videos | Min  | Sentences | FPAUSE | FPAUSE never used in an earlier set |
| ----------------------------- | ------ | ---- | --------- | ------ | ----------------------------------- |
| epic_actor_010                | 4      | 22.5 | 126       | 36     | 28                                  |
| epic_actor_173                | 5      | 20.4 | 106       | 35     | 27                                  |
| epic_actor_138                | 3      | 11.3 | 62        | 12     | 4                                   |
| epic_actor_119                | 3      | 7.8  | 27        | 11     | 6                                   |
| epic_actor_021                | 7      | 18.1 | 172       | 8      | 1                                   |
| other 26 speakers with FPAUSE |        |      |           | 71     | 8                                   |

The top two speakers hold 71/173 (41%) of the positives; 31 of 39 speakers have any. Any pooled metric is dominated by two voices; speaker-macro results rest on few positives per speaker.

## 4. Speaker overlap with the development set and visual evidence

- **All 39 speakers have already been used**: 29 in the 136/184 development sets, 10 in the v4 external diagnosis (`actors_never_in_any_set = 0`). `MATRIZ_CORPUS.md` says "the 10 speakers never used contribute only 8 FP sentences"; those 10 are the v4 external speakers, which were scored once (41 segments, 7 positives, all from `epic_actor_062`). Their 325 sentences hold 8 FPAUSE in total; 283 sentences were never used, containing 1 positive.
- Existing visual evidence: `mrcd-codex/MRCD_Plan_Avance_20261030/results/visual_gate.json` (legacy masks, 136 sentences, 62 videos): 0/136 admitted, 136 `active_speaker_unknown`, 136 `synchronization_unverified`, 70 coverage below 0.5, 10 with no visual observations. Mean legacy face coverage per video (62 videos with evidence): 0.45, 36 videos below 0.5. The other 12 videos have no visual-gate evidence at all (column `legacy_visual_n = 0`).
- **New small sample (described as such)**: MediaPipe FaceLandmarker (`models/face_landmarker.task`, same model family the project uses), 1 frame at 25%, 50% and 75% of each video, up to 4 faces. Result: face found in 3/3 frames in 21 videos, 2/3 in 18, 1/3 in 21, 0/3 in 14. This is presence of a face in 3 frames, not tracking, not active-speaker detection, not synchronisation; a face in frame can be another person in a parliament shot. Among the 51 videos with FPAUSE: 14 with 3/3, 13 with 2/3, 13 with 1/3, 11 with 0/3. Columns `sample_frames_with_frontal_face_of_3`, `sample_max_face_px` in the CSV.
- Therefore video is **not evaluable** in EPIC today: active speaker and synchronisation are unverified for every video, and the speaker-visibility sample is weak.

## 5. Subset usable for an evaluation without reusing inspected speakers

Criterion: speaker-disjoint from every speaker already used (development 29 + external 10), local media, FPAUSE positives.

| Option                                                              | Speakers                            | Positive sentences                                                       | Verdict                                                                                                                                                                                                                                                                                                     |
| ------------------------------------------------------------------- | ----------------------------------- | ------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Speakers never inspected                                            | **0**                               | **0**                                                                    | **Not evaluable.** No untouched speaker exists in the 74 videos.                                                                                                                                                                                                                                            |
| The 10 external-diagnosis speakers, sentences not yet used          | 10                                  | 1 (283 unused sentences, 8 FPAUSE in all their speech, 7 already scored) | Not evaluable: 1 positive.                                                                                                                                                                                                                                                                                  |
| Unused sentences of already-seen speakers (sentence-level hold-out) | 12 speakers with an unused positive | **74** (1307 unused sentences in total)                                  | Usable only as "new sentences, seen speakers": not speaker-independent, so it cannot support a generalisation claim. 55/74 positives come from 2 speakers (28 + 27); excluding them: 19 positives, 10 speakers. 22 videos contain these positives; 9 of them have a face in at least 2 of 3 sampled frames. |

Conclusions: (1) a speaker-independent EPIC test does not exist locally; (2) the sentence-level hold-out is the best available and must be reported as speaker-reused, per-speaker, with the two dominant speakers shown separately and also removed; (3) any video-gain claim is not evaluable (no verified active speaker or sync); (4) EPIC labels remain filled-pause presence per automatic-aligned sentence, nothing about repetition, revision, block or function.

## 6. Marked not evaluable

- Speaker-disjoint evaluation on EPIC (0 untouched speakers).
- Video benefit (active speaker and synchronisation unverified in 74/74; face sample weak in 14/74).
- Any temporal-boundary metric (sentence times are automatic WebMAUS).
- Repetition, revision, block, prolongation, pause function on EPIC (no labels).
- Using the 10 external speakers as a fresh test (1 unused positive).
- Redistribution of the media (CC BY-NC 4.0).
