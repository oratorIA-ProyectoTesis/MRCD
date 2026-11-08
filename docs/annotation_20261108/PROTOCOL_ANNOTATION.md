# Annotation protocol: new confirmatory reference cell (Phase 0.5)

Two variants are specified. **Primary path (no resources): blind self-annotation** under `ANNEX_20261108_SELF_ANNOTATION.md`. **Preferred alternative, if resources appear: two external annotators** (marked EXT). The annotator guide is `GUIA_ANOTACION.md` (Spanish).

Sources read for this protocol: `PREREGISTRO.md` §7, `Documentos/ALCANCE-MOTOR.MD` §8-10, the engine's `docs/taxonomy/ANNOTATION_MANUAL.md`, `docs/oe4/INFORME_OE4_REDUCIDO.md` §12 (agreement "not measurable"), `Documentos/CAMINO-A-PUBLICACION.MD` §1.

## 1. Purpose and scope

Produce a human reference for filled pauses, lexical fillers, repetitions and prolongations on audio no system has touched, with measured reliability. Out of scope this round: false starts, repairs, silent pauses, the visual channel, any psychological or clinical inference.

## 2. Roles

| Role        | Primary path                              | EXT (if resources appear)                                  |
| ----------- | ----------------------------------------- | ---------------------------------------------------------- |
| Rater 1     | author, two passes >= 7 days apart        | external annotator E1                                      |
| Rater 2     | native speaker, random subset >= 40 items | external annotator E2, all items                           |
| Adjudicator | author, by the written rules in §7        | third expert; the author neither annotates nor adjudicates |

## 3. Items

- Source: the 19 NOCANDO mp3 with no TextGrid. The 14 recordings with a TextGrid are the spent confirmatory cell and are excluded by name (stem match); no TextGrid is opened. The internal 87-clip corpus is **not used**: its licence is "no declarada" for all 87 videos (`results/auditoria_corpus_20261030/MATRIZ_CORPUS.md:217`).
- Sampling: `scripts/annotation/sample_items.py`, seed 20261108. Items per recording proportional to duration; inside a recording, equal time strata with one uniformly placed 6-10 s window per stratum. Sampling uses only file names and durations; no system output is consulted.
- 150 items (`manifest_nocando_unlabeled.csv`) plus 20 disjoint calibration items (`manifest_nocando_calibration.csv`).
- Known limits: 19 recordings, 7 speaker labels, items per speaker unequal (proportional to recording time). Speaker `maria` appears in the spent cell (`maria_es_2`, `_3`) and here (`maria_es_1`); whether `enrique2` is the same person as `enrique` cannot be determined from file names. Results must not claim speaker independence from the spent cell. One corpus, one narration task: no dialect stratification is possible.
- Rare classes: sampling is uniform in time, not by event, so rare classes (prolongation especially) may have few reference events; see §7 and the annex for the consequence.

## 4. Calibration

20 calibration clips; then the guide is frozen by commit and its hash recorded. Raters annotate them, agreement is computed, and the raters meet to resolve **guide** ambiguities (not individual labels). In the primary path there is no second rater at calibration, so the check is **solo**: the author annotates the 20 clips, waits at least 48 h, and re-annotates them (fresh blind ids and order); the stop rule below is applied to that intra-rater agreement. Rater 2 on the same 20 clips, if available, may replace the re-pass. This happens before the main passes and the guide is frozen after it; the 7-day gap of §2 applies only between main pass 1 and pass 2.

- Stop rule: if class-level κ on calibration (EXT: κ between raters; solo: intra-rater κ between the two calibration passes) is below 0.60 (a pre-declared working threshold, not a literature value) or any class with at least 5 events has Dice below 0.50 (smaller classes are reported as not assessable), revise the guide and run one more round on 20 new calibration clips (draw with `--n-calibration 40` and use items 21-40). After two rounds the campaign proceeds and the low agreement is reported as a finding, not hidden.
- Calibration clips are excluded from the confirmatory cell and from all reported reference statistics.

## 5. Blinding and order

- Raters never see system output, transcripts, file names, speakers or offsets.
- Each pass uses a different seed (`blind_session.py`): fresh blind ids and a fresh order. Clips are cut locally (`cut_clips.py`, WAV 16 kHz mono) outside the repository and named by blind id.
- The key (blind id to manifest) lives outside the repository. `unblind.py` requires a `.sha256` sidecar of the annotation CSV.
- EXT: raters work independently, no discussion until all CSVs are submitted.

## 6. Annotation schema and tool

CSV columns: `item_id, annotator, class, state, start_s, end_s, note, marker_token, marker_function`. The last two are the discourse-marker function layer (§6.1); the first seven are unchanged, and sheets without the last two remain valid (no marker spans).
`class` in `filled_pause, lexical_filler, repetition, prolongation, inaudible, discourse_marker, none`; `state` in `disfluent, functional, uncertain, none`; times in seconds relative to the clip start. An item with no events has exactly one row with `class=none, state=none` and empty times, so "looked, found nothing" differs from "not annotated". Tool: `TOOL_RECOMMENDATION.md`.

### 6.1 Discourse-marker function (`marker_function`)

Revision of 2026-11-08 (pre-annotation; see the annex). Purpose: a human label for the function of discourse markers, so a later contribution can compare functional vs disfluent markers, with an explicit uncertain state and measured agreement.

- For every occurrence the annotator hears of a token in the closed list (`este, o_sea, bueno, pues, a_ver, tipo`; `eh`, `em` and `mm` are not in the list and remain `filled_pause`), one row is added with `class=discourse_marker`, `state=none`, the span of the token, `marker_token` (normalized form from the list) and `marker_function` in `functional, disfluent, uncertain`. Other rows leave both columns empty. The definitions use observable cues only (`GUIA_ANOTACION.md` §3); Decision order (exhaustive, `GUIA_ANOTACION.md` §3.2): `functional` if there is an observable syntactic role; else `disfluent` only if there is no syntactic role AND the token is deletable without changing content AND at least one adjacency cue is present (silent or filled pause, prolongation, restart or repair); else `uncertain`, including no role but no adjacency cue, and unclear audio. This layer is separate from and stricter than `LF`: "no own tone" alone can justify `LF`=disfluent but gives `MK`=uncertain without an adjacency cue; disagreement between the two layers is expected by design and both are analysed separately.
- The layer is labelled in **both main passes** and in the **rater-2 subset**, and in the calibration passes. It is independent of the existing classes: an `este` can carry both an `LF` row and a marker row.
- Audacity label: `MK <token> <d|f|i>` (`labels_to_csv.py` validates the closed list and the value).
- Agreement (`agreement.py --field marker_function`, same `--mode` options): spans are matched per item and token with the same greedy IoU >= 0.10 matcher as events. On the **matched** spans we report intra-rater Cohen κ and Krippendorff α (nominal) for `intra`, the same for `subset` against each pass, the proportion of `uncertain` (per rater and pooled, over all spans) and the unmatched span counts. Pre-declared minimum: 30 matched spans; below it κ and α are reported as `not_available` with the reason, never as 0. κ and α are **always reported together with** `match_rate` (matched / max(n spans A, n spans B)) and detection F1 (2·matched / (n spans A + n spans B)), pooled over common items, in a token-aware version and in a token-agnostic version (IoU only) with the count of token mismatches. A low match rate makes κ and α non-interpretable: pre-declared, if detection F1 < 0.5 the function κ (and α) is reported as descriptive only (`function_kappa_descriptive_only`). In `subset` mode the same coverage flag as for events applies (`subset_below_minimum`, fewer than 40 common items). Computed before any adjudication, on raw CSVs.
- Marker rows do not enter the event statistics of §7 (they are skipped by the event reader), so existing results keep their meaning.
- Forbidden here too: any statement about intention, emotion or mental state; `marker_function` is a label of observable use in the sentence.

## 7. Agreement, then adjudication

Computed by `scripts/annotation/agreement.py` **before** any adjudication; the pre-adjudication values are the reported reliability.

- Alignment: per item, events matched one-to-one by temporal IoU >= 0.10 (greedy by IoU); unmatched events pair with `none`.
- Categorical: Cohen κ and Krippendorff α (nominal) on `class:state` (primary) and on `class` alone; item-level κ for "contains an event"; per-class Dice (2·both / (nA + nB)).
- Temporal: mean IoU of matched pairs and the share of matched pairs whose start **and** end differ by <= tolerance.
- **Tolerance ±250 ms.** A working compromise: wide enough to absorb the ambiguity of where a hesitation sound starts and ends in running speech, narrow enough to stay below a typical event. It is large relative to very short events (a 200-300 ms filled pause could fall inside the tolerance), so IoU is reported alongside and ±100 / ±500 ms are reported as sensitivity. The value is not derived from a cited study; it is declared in advance and may change only before the first agreement computation, through a dated annex.
- Undefined statistics (e.g. zero variance) are "not available", never 0. Classes with fewer than 20 reference events get counts only.
- Rater-2 agreement is computed against pass 1 and against pass 2 (raw, pre-adjudication), never against the adjudicated file, which already contains the author's resolutions (circular).
- Modes: `inter` (EXT), `intra` (pass 1 vs pass 2, both already unblinded to original ids), `subset` (rater 2 on a subset; flags n < 40).

**Adjudication rules (fixed now, applied mechanically, never by consulting system output).** For each aligned unit where sources disagree:

1. Event present in one source and absent in the other: keep it with state `uncertain`, unless a third source (rater 2, if the item is in the subset) sides with one of them; then take the majority.
2. Same event, different class: majority of three if rater 2 labelled the item; otherwise `uncertain`, with the pass-2 class recorded in `note`.
3. Same class, `disfluent` vs `functional`: `uncertain` (third-source majority applies as above).
4. Boundaries: mean of the two starts and mean of the two ends.
5. `inaudible` in any source: keep `inaudible`.
   Each adjudicated row keeps a `note` with the rule number. Raw CSVs are never overwritten.

## 8. How the result becomes a NEW untouched confirmatory cell

It is untouched only if all of the following are frozen and committed **before any system sees the audio**:

1. This protocol, the annex, `GUIA_ANOTACION.md` (hash) and the commit of `sample_items.py` / `agreement.py`.
2. The manifest (hash), the seed and the list of excluded recordings.
3. Raw annotation CSVs with `.sha256` sidecars, the adjudicated CSV, the agreement report and the adjudication notes.
4. A **protocol G**: classes evaluated, matching rule (IoU/tolerance), metrics, minimum events per class, decision rule, pinned system version and configuration, and the statement that the cell is spent after one run. Written and committed before the run.

After the single run the cell is spent. Results carry the label "blind single-rater reference (author)" (or "two external raters" under EXT) together with the reliability values. Tuning on the cell invalidates it.

## 9. Storage, licence and ethics checklist

- Repository: manifests (file name, sha256, offsets, ids), guide, protocol, scripts, annotation CSVs (ids and labels only). **Never** audio, transcripts or personal data. Clips and the blind key stay outside the repository.
- NOCANDO: CC BY-NC-SA 3.0 as recorded in `THIRD_PARTY_NOTICES.md:15` and `docs/corpora/README.md:29` (source `https://recursos-glif.upf.edu/nocando/recordingscorpus?field_language_value=spanish`). The local folder has no licence file (only `ESTADO.md` and `FUENTE.url`), so the licence text itself was **not verified** here. To check: non-commercial research use only; annotations are an adaptation, so keep them under the same terms with attribution; do not publish clips or transcripts. Confirm at the source page before inviting anyone.
- Internal 87 YouTube clips: licence "no declarada"; excluded.
- Second rater: send only the blind clips needed, under the same non-commercial terms, and ask for deletion afterwards.
- Raters give no personal data; annotator ids are codes. If external people take part, check whether the university requires an ethics or consent record.
- Forbidden in any field or note: emotion, anxiety, intention, personality, diagnosis.

## 10. Time estimate

150 items × ~8 s is about 20 min of audio; at ~10× real time that is about 3.3 h per pass, about 4 h with breaks (sessions <= 1 h). Calibration (20 clips) about 0.5 h. Discourse-marker layer (revision 2026-11-08): about +2 h in total across the two passes and the rater-2 subset. Primary path: 2 passes about 8 h, rater 2 on 40 items about 1 h, adjudication about 1 h, computation and freeze about 1 h: roughly 12-13 h of author time over at least 2 weeks. EXT: two raters × ~4 h plus 1-2 h for the adjudicator.
