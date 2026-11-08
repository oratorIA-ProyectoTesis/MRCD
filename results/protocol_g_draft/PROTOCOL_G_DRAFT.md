# Protocol G - confirmatory evaluation against a HUMAN reference (NOCANDO, 19 unlabelled recordings) - DRAFT

**Status: DRAFT, NOT FROZEN, NOT REGISTERED (revision 2 of 2026-11-08, see the revision log at the end).** Written by a
software agent and revised after one independent fresh-context review; no advisor ratification. No human annotation of
the cell exists yet (pass 1 has not started), no system and no aligner has touched its audio, and no transcript of these
recordings was opened to write this draft (only file names were listed, section 3.3).

**Two-stage freeze (binding, section 14).**

- **Stage 1 - before annotation pass 1.** Everything that defines the test is frozen: hypotheses, predicted directions,
  thresholds, evaluability floors, reference quality gates, statistics, matching and mapping rules, the timing method of
  the transcript-derived reference (closed decision OD-2), every default and every closed decision. The frozen text is a
  NEW file `PROTOCOL_G.md` (this draft with OD-1 closed and "DRAFT" removed), committed and hashed
  (`PROTOCOL_G.stage1.sha256.json`), then registered by the author on OSF as a **pre-annotation registration** with its
  timestamp (`OSF_PREREGISTRATION_CHECKLIST.md`). Pass 1 does not start before the registration exists.
- **Stage 2 - after the annotations are committed, before any system or aligner run.** Only three things may be added:
  (a) the harness code and its tests, implementing the stage-1 text; (b) `SYSTEM_FREEZE_G.json` with the pins of
  section 14; (c) the OD-4 inclusion record, decided by merge status alone. Anything else changed after pass 1 has started
  is a **dated annex declared post-annotation** that states which annotation counts, agreement figures or gate values
  the author had seen when writing it; it may add, never alter, a family, gate or threshold.

The reference is produced with the annotation kit v1.1 (`docs/annotation_20261108/`: `PROTOCOL_ANNOTATION.md`,
`ANNEX_20261108_SELF_ANNOTATION.md`, `GUIA_ANOTACION.md`, `README.md`; code `scripts/annotation/agreement.py` and
`scripts/annotation/markers.py`). The kit's README step 11 ("freeze protocol G") corresponds to stage 2 here; stage 1 is an
additional, earlier freeze. Structure, statistics, locks and single-shot rule mirror protocol F
(`results/confirmatoria_20261107/PROTOCOL_CONFIRMATORIO.md`, `PROTOCOL_ANNEX_20261107_SECONDARY_CW2.md`).

Hypothesis ids carry the prefix `PG` (primary `PG1`-`PG3`; secondary `PGS*`, cw2 arm `PGC*`, FT arm `PGF*`, absolute
criteria `PGA*`) so they never collide with the G4 latency gate of the engine documentation.

## Decisions

### The one open decision (must be closed by the author before stage 1, i.e. before pass 1)

| id   | Decision                                                                                             | Default (frozen if the author does not change it)                                                                                                                                                                        | Why it must be closed before pass 1                                                                                                                                                                      |
| ---- | ---------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| OD-1 | Speaker `maria` (5 items, recording `maria_es_1`) also spoke in the spent cell (`maria_es_2`, `_3`). | **Keep + sensitivity:** the 5 items stay in the primary analysis; pre-declared sensitivities S-MARIA (exclude them) and S-MARIA-ENRIQUE (also exclude `enrique2`, 25 items, identity with spent-cell `enrique` unknown). | Exclusion means redrawing the manifest (`sample_items.py --exclude-speaker maria`, new committed manifest), which is only possible before pass 1. After pass 1 the only remaining option is the default. |

If the author chooses exclusion, the redrawn manifest replaces the one cited in section 3.1 in the stage-1 commit, the
item and speaker counts of section 3.1 are updated there, S-MARIA and S-MARIA-ENRIQUE become S-ENRIQUE only, and the
leave-one-speaker-out criterion of section 10 reads "at least 5 of 6".

### Closed decisions (rules, frozen at stage 1)

| id   | Closed as                                                                                                                                                                                                                                                                                                                                                                                                                     | Basis                                                                                                                        |
| ---- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| OD-2 | `R_T` timing by a **pinned CTC forced aligner independent of every tested system** (section 4.3). Anchoring to the `p0` decode is a pre-declared sensitivity (S-P0ANCHOR).                                                                                                                                                                                                                                                    | Anchoring to `p0` made the reference's timing depend on the output of the system family under test (circular for `p0_rule`). |
| OD-3 | Rater-2 gate: "not assessable" (no rater 2, `subset_below_minimum`, too few events in the subset) gives a **conditional** verdict, annotated; an assessable gate that **fails** makes the class descriptive (section 7).                                                                                                                                                                                                      | Separates lack of a check from a failed check.                                                                               |
| OD-4 | `o2026_13` / `ft_small_v1` (family PGF) is included iff, at the stage-2 freeze, its branch is merged into the system commit and the `ft_small_v1` CT2 `model.bin` sha256 is pinned in `SYSTEM_FREEZE_G.json`. Decided by merge status alone; deadline = the stage-2 freeze. Otherwise PGF is removed and recorded as `not_available` (reason: not merged at the stage-2 freeze).                                              | No annotation count, gate value or other consideration may enter this decision.                                              |
| OD-5 | Until the author confirms at the NOCANDO source page that non-commercial research annotation is covered by its licence (CC BY-NC-SA 3.0 recorded, not verified for this use; `docs/corpora/README.md` records "audit only"), only aggregates and **hashes** are published, in the repository and on OSF; this includes the manifests (uploaded as sha256 only). This is a pending author action, not an open design decision. | Annotations are an adaptation; ShareAlike terms apply.                                                                       |

## 1. What this test can and cannot claim

| Contribution                           | Question                                                                                                                                                              | Reference                                                                                                                                                                                                                 | Claim it can support                                                                                                                                                                              |
| -------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **C1 reference difference**            | How much does the transcript-derived lexical reference (the frame of protocol F) change the F1 of the lexical rule against a human audio reference on the same items? | (i) `R_T`: lexical rule on the corpus transcript, timed by an independent aligner; (ii) `H_lit`: blind human audio reference, repetitions in any state (primary); (iii) `H_disf`: disfluent-state repetitions (secondary) | The **reference difference due to transcript origin, with timing from an independent aligner** (`R_T` vs `H_lit`), and separately the convention effect (`H_lit` vs `H_disf`), **on these items** |
| **C2 ASR erasure validated by humans** | What fraction of human-heard events leaves textual evidence in each ASR's output?                                                                                     | `H_disf` events (`REP`, `FP`; `PRO` descriptive)                                                                                                                                                                          | Survival per ASR against human-heard events, not against a transcript heuristic                                                                                                                   |
| **C3 discourse-marker state**          | Does a system's state for a heard marker agree with the human function label better than chance?                                                                      | `H` marker rows (`marker_token`, `marker_function`)                                                                                                                                                                       | A "better than chance" check of the system's binary state against the human label, under a declared construct overlap (section 12)                                                                |

Not claimable: human-judged disfluency in general (one primary rater, the author), any population beyond 7 speaker labels
of one corpus and one task (picture narration), dialect effects (dialect undocumented), speaker independence from the
spent cell (OD-1), calibration, video or multimodal benefit, false starts / repairs / pauses (not annotated), recognition
of discourse function as such (C3 is a chance-level check), clinical or psychological inference of any kind. The visual
channel is not used. Every verdict is "supported on these recordings", never "demonstrated" in general.

A class whose quality gate (section 7) or evaluability gate (section 8) fails gets **"no confirmatory evidence
available"**, which is neither a success nor a failure.

## 2. Order of operations (binding)

1. **OD-1 closed** by the author (redraw of the manifest only if exclusion is chosen).
2. **Calibration** per `docs/annotation_20261108/README.md` (20 calibration clips, solo re-pass after >= 48 h, stop rule,
   at most one more round); guide frozen by commit. The calibration clips are disjoint windows **of the same 19
   recordings**; their labels and counts are seen by the author before stage 1 and are declared in the registration
   (section 12). They are excluded from every analysis.
3. **Stage 1 freeze.** `PROTOCOL_G.md` written from this draft; `PROTOCOL_G.stage1.sha256.json` (section 14) written and
   committed; OSF pre-annotation registration by the author; `results/protocol_g/OSF_REGISTRATION_STAGE1.json` (URL, DOI
   if any, UTC timestamp, stage-1 commit, sha256 of `PROTOCOL_G.md`) committed. Any value of this draft changed between
   the draft commit and stage 1 is listed in `PROTOCOL_G.md` as "changed after calibration", with the calibration figures
   that had been seen.
4. **Annotation campaign** (pass 1, >= 7 days, pass 2, rater-2 subset, agreement **before** adjudication, adjudication).
   All raw CSVs, the adjudicated CSV and the agreement JSONs are committed under `docs/annotation_20261108/` with `.sha256`
   sidecars.
5. **Stage 2 freeze.** Harness and tests finished (section 15); `REFERENCE_GATES_G.json` computed by the harness's gate
   generator **from the committed annotation files only** (no system output exists); `SYSTEM_FREEZE_G.json`
   (`freeze-system`); OD-4 record; `PROTOCOL_G.stage2.sha256.json`; all committed; the stage-2 hash file uploaded to the
   OSF project as a dated document and its upload timestamp added to `results/protocol_g/OSF_REGISTRATION.json`. The
   harness refuses `run` if either OSF file is missing or if the sha256 of `PROTOCOL_G.md` differs from the registered one.
6. `verify`, then **one** `run`. The cell is spent after it.
7. Conclusions written from the result file by the rules of section 10, committed before any secondary reading.

Nothing in steps 4-7 may consult a system output on these recordings. Stage-2 additions implement the stage-1 text; they
cannot change it.

## 3. Data cell

### 3.1 Items and units

- **Items:** the 150 rows of `docs/annotation_20261108/manifest_nocando_unlabeled.csv` (columns `item_id, file, sha256,
start_s, end_s, speaker_if_in_name`), drawn by `scripts/annotation/sample_items.py` with seed 20261108 from file names
  and durations only. 19 recordings, 7 speaker labels: `cristina-mem` 44 items (3 recordings), `enrique2` 25 (3),
  `maribel` 25 (3), `iris` 24 (3), `gemma` 22 (3), `jessica3-mem` 5 (3), `maria` 5 (1). (Counts change only if OD-1 is
  closed as exclusion.)
- **Unit of C1 = the item** (one 6-10 s window). An item is positive for a class under a reference if that reference has
  at least one event of the class in it (section 4). The 20 calibration items (`manifest_nocando_calibration.csv`) are
  excluded from everything.
- **Unit of C2 = the human event** (one row of the adjudicated CSV). **Unit of C3 = the human marker span** (one
  adjudicated row with `class=discourse_marker`).
- Excluded by name: the 14 recordings of the spent cell (stem match), as in the annotation protocol.

### 3.2 Audio

Same recipe as protocol F: mp3 -> channel mean -> 16 kHz -> PCM_16 WAV bytes in memory; each mp3 sha256 is checked against
the manifest. **Each system, and the aligner, analyses the WHOLE recording once** (the service condition of use; a
6-10 s clip is below the 13 s service minimum). Events are then mapped to items (section 6).

### 3.3 Transcripts (for `R_T` only)

A file listing (names only, no content) shows one `transcripciones/<stem>_ts.txt` for each of the 19 recordings and no
`alineamientos/<stem>.TextGrid` for any of them. The transcripts are **never opened by a person** and are read by the
harness only inside the single run (raters never see them; `PROTOCOL_ANNOTATION.md` section 5). Their sha256 is computed
from bytes by a script at stage 1 (digests only, no content displayed) and frozen. The earlier audit published segment
counts of `hm|eh|em|mm` over all 40 transcripts, including these 19 (`CONTAMINATION_AUDIT.md` row 19): the order of
magnitude of transcribed filled pauses is known, the positives are not.

## 4. References

### 4.1 Human reference `H` (adjudicated CSV, kit v1.1 schema)

`class` in `filled_pause` (`FP`), `lexical_filler` (`LF`), `repetition` (`REP`), `prolongation` (`PRO`), `inaudible`,
`discourse_marker`, `none`; `state` in `disfluent`, `functional`, `uncertain`, `none` (marker rows carry `state=none` and
the two marker columns). Times are clip-relative and are converted to recording time by adding the item's `start_s`. An
item with `inaudible` overlapping >= 50 % of its duration is excluded from every class (count reported).

- **`H_lit` (primary for C1, repetition):** item positive iff it has >= 1 `REP` event in **any** state (`disfluent`,
  `functional` or `uncertain`). It is the human construct closest to `R_T`, whose literal rule counts every immediate
  repeat regardless of function (the DIHANA / transcript convention, in which "sí, sí" is a repetition).
- **`H_disf` (secondary for C1; primary for C2 and the PGA criteria):** item positive for class c iff it has >= 1 event of
  c with state `disfluent`. An item whose only events of c are `uncertain` is **not scorable** for c under `H_disf` (count
  reported).
- **Availability of the "sí, sí" distinction.** The guide yields a **state** distinction for repetitions (`functional`
  covers emphatic and polarity repeats), so the convention contrast `H_lit` vs `H_disf` is available. The guide records
  **no word content** for `REP`, so the polarity subset is **`not_available`** from the human reference; plan A's R4
  effect (`results/plan_a_20261108/RESULTS_A.md`, reading 2) can be confronted only through the state contrast.

### 4.2 Marker function (C3)

Field contract of kit v1.1 (`markers.py`, not renamed here): rows with `class=discourse_marker`, `state=none`, column
`marker_token` in the closed list `este`, `o_sea`, `bueno`, `pues`, `a_ver`, `tipo` (`eh`, `em`/`mm` are **not** markers,
they stay `FP`), and `marker_function` in {`functional`, `disfluent`, `uncertain`}. Decision order of the guide
(`DECISION_ORDER`): `functional` if the token has an observable syntactic role; otherwise `disfluent` only if it has no
role AND is deletable AND at least one adjacency cue is present (`silent_or_filled_pause`, `prolongation`,
`restart_or_repair`); otherwise `uncertain`. The marker layer is separate from `LF` (an `este` can carry both rows).

### 4.3 Transcript-derived reference `R_T`

**Tokens.** Built **exactly as in protocol F** from the transcript text: the vendored CHM150 rule
(`mrcd.context_eval.transcript_reference`; lower case, accents removed, non-`[a-z0-9n]` to space; non-lexical filled pauses
`^(e+h*|eh+|ehm+|em+|mm+|hm+|ah+|aa+h*)$` separated, bare `e` excluded before `i`/`hi`; `_`, `#` and bracket marks not
interpreted). `rep_literal` events = every immediate exact repetition of 1-4 contiguous lexical tokens (event = the second
copy); `fill_ref` events = every filled-pause token. `este` is not a filler.

**Timing (primary, frozen choice): CTC forced alignment with a model that no tested system uses.**

- Model: `jonatasgrosman/wav2vec2-large-xlsr-53-spanish`, revision `96d7e9b4e4a78af515a3c6d3cee7c0826045d276`,
  `pytorch_model.bin` sha256 `41c110e55d2eac8c79486ad87dbe8f9527ed034fe087a6adf03c891eeba914c1` (the checkpoint already
  pinned in `results/asr_survival_ctc_20261107/`; Apache-2.0 per its model card, to be re-read by the author at stage 1).
  No tested system (MRCD with Whisper small, CrisperWhisper `cw2`, `ft_small_v1`) uses this or any wav2vec2 checkpoint.
  Its `language_model/` folder is not used.
- Emissions: `transformers` `Wav2Vec2ForCTC`, float32, `torch.inference_mode`, log-softmax over the checkpoint's
  character vocabulary, computed over consecutive 20 s windows with 2 s of left and right context, keeping only the
  central frames (no frame dropped or duplicated). GPU; CPU float32 allowed only as a device fallback (same semantics,
  recorded).
- Alignment: the standard CTC Viterbi trellis (blank interleaved, repeats allowed only through a blank) of the normalised
  transcript tokens joined by the word delimiter `|`, implemented in the harness in numpy (no new alignment dependency).
  Characters absent from the checkpoint vocabulary are dropped from the target (count reported); a token left with no
  character takes times linearly interpolated between its nearest aligned neighbours (count reported).
- Token time = first to last frame of its characters times the frame stride of the checkpoint (20 ms). An `R_T` event
  belongs to the item that contains its midpoint.
- **Aligner-confidence gate (per recording, the only exclusion on `R_T` timing):** token confidence = mean posterior
  probability of its characters along the Viterbi path. If fewer than 70 % of the recording's lexical tokens have
  confidence >= 0.50, the recording's items are excluded from every contrast that uses `R_T` (all systems and all human
  references equally; count reported). Both values are judgement values frozen here. No system output enters this gate.
  Sensitivity S-NOGATE drops it.
- If the aligner fails on a recording, `R_T` for that recording is `not_available` and its items leave every `R_T`
  contrast; if more than 1 of 19 recordings fail, `R_T` is not interpretable and every `R_T` contrast is "not evaluable".
- The pinned identity (model id, revision, weights hash) is frozen here; stage 2 records only provenance (`torch`,
  `transformers` versions, device). The harness refuses to align if the weights hash differs.

**Sensitivity S-P0ANCHOR (pre-declared, descriptive):** the earlier method, transcript tokens aligned by unit-cost
Levenshtein (substitution before deletion before insertion) to the `p0` word sequence, matched tokens taking the `p0` word
times, unmatched ones interpolated; its own 70 % exact-match recording gate applies inside this sensitivity only. It is
declared circular for `p0_rule` and is reported only to show how much the timing method matters.

The `R_T` counts and the aligner gate are computed by the harness **before** any scoring, inside the single run, and are
written to the result before any metric.

## 5. Systems (frozen at the system commit; nothing is fitted, selected or thresholded on this cell)

One decode per (recording, ASR profile), shared by every system that uses that profile (`CachingASR`), so systems on the
same ASR differ only in their decision.

| id               | Decision                                                                                                                                                                                     | ASR profile           | Role                                                    |
| ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------- | ------------------------------------------------------- |
| `default`        | `MrcdAnalyzer.analyze_bytes` with decision profile **`o2026_11`** (the resolved value of `DEFAULT_DECISION_PROFILE` at the time of this draft; run by name), options as protocol F section 2 | `p0`                  | primary                                                 |
| `o2026_12`       | opt-in profile (single P1 pass, lexical rule + demotion layer, lexical-filler subclass)                                                                                                      | `p1_disfluent_prompt` | primary                                                 |
| `o2026_13`       | opt-in profile (FT ASR + primary lexical rule, no demotion)                                                                                                                                  | `ft_small_v1`         | **candidate**, family PGF only, OD-4                    |
| `V0`             | decision profile `v0`                                                                                                                                                                        | `p0`                  | primary                                                 |
| `p0_rule`        | protocol F `rule_events` on the `p0` words                                                                                                                                                   | `p0`                  | primary                                                 |
| `p1_rule`        | same rule on the P1 words                                                                                                                                                                    | `p1_disfluent_prompt` | primary (descriptive contrasts)                         |
| `cw2_rule`       | same rule on `cw2` words, tags mapped by `FP_TAG_NORM`, MRCD decision discarded (annex F section 2, CUDA float16, no CPU fallback)                                                           | `cw2`                 | **secondary**, family PGC only (non-commercial licence) |
| `always_present` | every item positive                                                                                                                                                                          | -                     | floor, repetition and filled pause                      |

- **Resolved default.** The `default` system is the profile named `o2026_11`. If `DEFAULT_DECISION_PROFILE` at the system
  commit resolves to another name, `run` refuses (a change of the tested default is a protocol change, not a pin).
- **Lexical-filler rule parameters (part of the tested definition, frozen):** `mrcd.service.contextual`
  `EVIDENCE_GAP_S = 0.15`, `EVIDENCE_DUR_S = 0.35`, `EVIDENCE_PUNCT = {",", "...", "…"}`, `LEXICAL_FILLERS` as at the
  system commit (its sorted list is written to `SYSTEM_FREEZE_G.json`). A different value of the three constants makes
  `run` refuse.
- `ft_small_v1` is evaluated here only because NOCANDO is out of domain: the model was trained on DIHANA folds 1-4
  speakers and **must never be run on DIHANA training speakers**; this cell has none.
- Engine settings as protocol F: packaged checkpoint, Whisper `small` CTranslate2 int8 on CPU for `p0`/`p1` (the two
  profiles share one `model.bin` and differ by prompt), `OMP_NUM_THREADS=4`, `HF_HUB_OFFLINE=1`.
- For filled pauses, `default`, `o2026_12`, `o2026_13` and `V0` contribute their `filler_word` events of subclass
  `filled_pause` (or without subclass); `lexical_filler` events are never pooled with filled pauses (plan A reading 7).
- Repetition for `default` equals `V0` if the default keeps the engine decision; the result records
  `default_events_equal_v0_events`.

## 6. Mapping and matching

- **Event to item (C1):** a system event of class c belongs to the item whose window contains the event midpoint (events
  shorter than 20 ms widened to 20 ms, as in protocol F). Events in no item are dropped and counted. Sensitivity
  S-OVERLAP: any overlap >= 1 ms.
- **Failed recordings (C1):** a recording on which a system fails (service error) is **excluded** from every contrast in
  which that system takes part, for every member of the contrast (pairing is kept); it is listed and never scored as
  all-zero. If more than 1 of 19 recordings fails for a system, that system is `interpretable: false` and its contrasts
  are "not evaluable" (Holm members still counted).
- **Survival (C2):** for a human event [s, e] (recording time), the ASR words considered are those overlapping
  [s - 0.25, e + 0.25] (**±250 ms**, the annotation protocol's primary tolerance; ±100 ms and ±500 ms as sensitivity
  S-TOL). A `REP` event survives iff those words, after non-lexical fillers are removed, contain an immediate exact
  repetition of 1-4 tokens (the protocol F rule). An `FP` event survives iff at least one of those words matches the
  non-lexical filled-pause pattern (bare `e` exception; `cw2` tags mapped by `FP_TAG_NORM`; other tags dropped).
  Survival is **pooled over events** (survived events / events), with recordings as the resampling unit. `PRO`: standard
  orthographic ASR output has no lengthening mark, so **text-evidence survival of prolongation is `not_available` by
  construction**; reported descriptively is only the share of `PRO` events with >= 1 ASR word in the window. A failed
  decode makes that ASR arm `not_available` for C2 (annex F section 7 rule); it is never reported as 0 or over the
  recordings that did run.
- **Event-level detection (descriptive only):** system events vs `H_disf` events, greedy one-to-one by temporal IoU >= 0.10
  (`agreement.match_events`), per class; precision/recall/F1 reported, no test.
- **Marker state (C3):** a human marker span is "transcribed" by an ASR iff an ASR word (or two consecutive words for
  `o_sea`, `a_ver`, joined with `_` after normalisation) whose normalised form equals `marker_token` overlaps the span
  ±250 ms. The system state is the output of `mrcd.service.contextual.classify_lexical_fillers` on that ASR's words at the
  system commit: `disfluent` or `uncertain`. A span with no such word is **`not_transcribed`**. The rule never emits
  `functional`, so system recall of human-`functional` as such is `not_available`; agreement is measured on the binary
  system state (1 iff `disfluent`). Primary system for C3: `o2026_12` (the only profile that emits `lexical_filler`); the
  same rule on `p0`, `cw2` and `ft` words is descriptive.

## 7. Reference quality gates (computed BEFORE any system run, from the committed annotation files only)

All values are **pre-adjudication** outputs of kit v1.1 `scripts/annotation/agreement.py` (`--mode intra` for pass 1 vs
pass 2; `--mode subset` for rater 2 vs pass 1 and vs pass 2, never vs the adjudicated file). Where a gate needs a per-class
statistic that `agreement.py` does not print, the gate generator computes it with `agreement.match_events` and
`agreement.cohen_kappa` imported unchanged. Thresholds are pre-declared working values (0.60 as in the annotation
protocol's stop rule; 0.40 for harder judgements), not literature values. Point estimates decide; intervals are reported.

| Gate                                     | Statistic                                                                                                                                        | Applies to                                          | Threshold                                                                     |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------- | ----------------------------------------------------------------------------- |
| QG-1 presence (intra)                    | per-class item-level Cohen κ for "item contains >= 1 event of class c" (pass 1 vs pass 2) **and** per-class Dice (`per_class[c].dice`, verbatim) | FP, REP, LF, PRO                                    | κ >= 0.60 and Dice >= 0.50; assessable only if c has >= 5 events in each pass |
| QG-2 state (intra)                       | κ on `state` of matched `REP` events; κ on `marker_function` of matched marker spans (`compute_markers`, `kappa` verbatim)                       | REP (convention contrast PGS4/PGS5), markers (PG3)  | κ >= 0.40                                                                     |
| QG-3 rater 2 (subset)                    | item-level presence κ of rater 2 vs pass 1 **and** vs pass 2 (both must pass); for markers, `compute_markers` in subset mode vs each pass        | each class with >= 10 events in the subset; markers | κ >= 0.40                                                                     |
| QG-4 boundaries (intra, descriptive)     | `boundary_agreement` (±250 ms) and `mean_iou_matched`                                                                                            | all classes                                         | reported, no gate                                                             |
| QG-5 marker detection (PG3 precondition) | `function_kappa_descriptive_only` of `compute_markers`, verbatim (true iff `detection_f1` < `MIN_DETECTION_F1` = 0.5)                            | markers                                             | must be false                                                                 |

Consequences (never a silent drop):

- **QG-1 fails** for c (assessable and below threshold) -> class c is **descriptive**: metrics reported under "reference
  below quality gate", its Holm members are "not evaluable" (family size unchanged), no verdict. **QG-1 not assessable**
  (fewer than 5 events of c in a pass) -> c is descriptive (such a class also cannot meet the floors of section 8).
- **QG-1 is not the calibration stop rule.** The stop rule (`PROTOCOL_ANNOTATION.md` section 4) is applied to the 20
  calibration clips and only decides whether the guide is revised; QG-1 is applied to the 150 cell items after the main
  passes. Passing one implies nothing about the other; both are reported.
- **QG-2 fails** for REP -> PGS4 and PGS5 are "not evaluable" (descriptive); for markers -> PG3 is "not evaluable".
  For markers, `compute_markers` returns `kappa = None` with `not_available_reason` when there are fewer than
  `MIN_MARKER_SPANS` = 30 **matched spans between the two human sources**; that counts as a failed QG-2. QG-2 for markers is
  read only if QG-5 passes on the intra comparison.
- **QG-5:** `function_kappa_descriptive_only = true` on the intra comparison -> PG3 "not evaluable" (descriptive).
- **QG-3 reads `subset_below_minimum` from `agreement.py` verbatim** (true iff fewer than `MIN_SUBSET_ITEMS` = 40 common
  items). Two outcomes are kept apart:
  - **Not assessable** - no rater-2 file, `subset_below_minimum = true`, fewer than 10 events of c in the subset, or (for
    markers) `kappa = None`: the verdict of every hypothesis on c is still computed and reported as **conditional**, with
    the annotation "inter-rater gate not assessable: <reason>".
  - **Failed** - assessable and κ < 0.40 against either pass, or (markers, assessable) `function_kappa_descriptive_only =
true` in subset mode: class c (or PG3) is **descriptive**, its Holm members "not evaluable".
- An undefined statistic on the intra comparison (zero variance) is `not_available` and counts as a failed gate.
- Krippendorff α (nominal) is reported next to every κ; the gates use κ.

## 8. Evaluability floors (from the reference only, before any scoring)

| Use                         | Minimum                                                                                                                                                                                                           |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| C1, PG1 / PGS6 / PGC1 (REP) | under `H_lit`: >= 20 positive and >= 20 negative scorable items, positives in >= 4 recordings and >= 3 speakers; and `R_T` (after the aligner gate, computed in the run before any metric): >= 20 positive items  |
| C1, PGS4 / PGS5 (REP)       | the same floor under `H_disf` on the decomposition set (section 9)                                                                                                                                                |
| C2, type c (REP, FP)        | >= 20 `H_disf` events of c, in >= 4 recordings and >= 3 speakers                                                                                                                                                  |
| C3, PG3                     | reference floor: >= 30 non-uncertain adjudicated marker spans, in >= 4 recordings and >= 3 speakers; in-run floor: >= 20 of them transcribed by the `o2026_12` ASR (else "not evaluable", with the erasure count) |
| PRO                         | descriptive in any case (no text survival, no transcript reference); counts only if < 20 events                                                                                                                   |

**Two different 30-span minima.** `MIN_MARKER_SPANS` = 30 (kit v1.1) is the minimum of **matched spans between two human
sources** for a reliability κ (QG-2, QG-3). The PG3 reference floor of 30 is the number of **adjudicated non-uncertain
human spans** available to compare with the system. One does not imply the other.

**Why 20.** The floor equals the annotation protocol's "fewer than 20 reference events -> counts only". It is lower than
protocol F's 30 **not** because a small cell needs less evidence (it needs more), but because a 150-item cell (about 20
min of audio) cannot reach 30 for most classes; the price is wide intervals and a large minimum detectable effect (section
10.2). Reaching the floor makes a test computable, not powered. Below the floor the result is `not_evaluable (n = ...)`,
never a number and never 0. Sampling is uniform in time, so rare classes may fail this gate; that is a declared possible
outcome.

## 9. Statistics (decided before any output)

- Cluster = **recording** (19). Bootstrap: resample recordings with replacement, **seed 13**, 95 % percentile CI with 2 000
  resamples, tests with 5 000 resamples; same code as protocol F (`boot_ci`, `paired_delta`, `holm` of
  `scripts/confirmatoria/run_confirmatoria.py`, imported unchanged). Recordings of one speaker are dependent (7 speakers,
  `cristina-mem` 29 % of the items): intervals are anti-conservative, hence the leave-one-speaker-out requirement of
  section 10.
- p two-sided = `min(1, 2*min(P(d<=0), P(d>=0)))`, p one-sided = `P(d<=0)` (or `P(d>=t)` against a threshold t), +1
  correction. A resample in which a statistic is undefined counts against the hypothesis and the number of such resamples
  is reported (PG3 adds the 10 % rule of section 10.1).
- **Scorable item sets, identical for both references of a contrast.** `S_lit` = items scorable under `R_T` (aligner gate
  passed, recording not failed) and under `H_lit` (not inaudible-excluded). `S_dec` = `S_lit` minus items not scorable under
  `H_disf` (uncertain-only `REP` items). PG1, PGS6 and PGC1 use `S_lit`; PGS4 and PGS5 use `S_dec`. For every contrast the
  prevalence (positive items / items) under **each** reference on its set is reported.
- **Decomposition (exact on `S_dec`, descriptive):** F1(`R_T`) - F1(`H_disf`) = [F1(`R_T`) - F1(`H_lit`)] (transcript-origin
  effect) + [F1(`H_lit`) - F1(`H_disf`)] (convention effect), all F1 of `p0_rule` on `S_dec`. The origin term on `S_dec`
  is reported next to PG1 (on `S_lit`); they differ only by the uncertain-only items.
- All contrasts are paired: the same items (C1), the same human events (C2) or the same marker spans (C3) in every
  resample.
- F1 with no predicted and no reference positive in a resample: undefined (rule above). F1 of a system with zero
  predictions on the full cell: reported as 0 with `tp=fp=0` (a measured outcome, not an absence).
- Leave-one-speaker-out (LOSO): for each speaker label, the point estimate of the hypothesis statistic recomputed without
  that speaker's recordings; an undefined LOSO estimate counts as not keeping the predicted direction.

## 10. Hypotheses, multiplicity, success and failure

A hypothesis is **supported on these recordings** iff (i) `p_Holm < 0.05` within its family, (ii) the 95 % interval
excludes the null value (0, or the threshold) on the predicted side, (iii) the sign is the predicted one, and (iv) the LOSO
point estimate keeps the predicted direction for at least 6 of the 7 speakers (at least 5 of 6 if OD-1 excludes
`maria`). A significant result in the opposite sign is **contradicted**. Otherwise **not demonstrated** (a result, not an
inconclusive run). A member that cannot be computed (gate, floor) is "not evaluable" and still counts in its family. A
verdict under a not-assessable QG-3 is reported as conditional (section 7). Wording is always "supported on these
recordings", never "demonstrated".

### 10.1 Primary family (Holm, fixed size 3)

| id  | Contribution         | Contrast                                                                                                                                             | Test                                                            | Predicted                                                                                                                                       |
| --- | -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| PG1 | C1 transcript origin | Δ_origin = F1(`p0_rule` vs `R_T`) - F1(`p0_rule` vs `H_lit`), repetition, on `S_lit`                                                                 | two-sided                                                       | Δ_origin > 0 (the transcript-derived reference, timed by an independent aligner, gives the rule a higher F1 than the human any-state reference) |
| PG2 | C2 erasure           | pooled survival of `H_disf` `REP` events in `p0` text, against t = 0.80                                                                              | one-sided, H0: survival >= 0.80                                 | survival < 0.80                                                                                                                                 |
| PG3 | C3 marker state      | Cohen κ between `o2026_12` binary marker state and human `marker_function` (disfluent = 1, functional = 0) on transcribed, non-uncertain human spans | one-sided, H0: κ <= 0; criterion: **lower 95 % bound of κ > 0** | κ > 0 - explicitly a **"better than chance" check**; no κ range is predicted                                                                    |

**PG1 claim.** A supported PG1 states a **reference difference due to transcript origin, with timing from an independent
aligner**, for the lexical rule on `p0`, on these items. It does not by itself say which reference is right.

**PG2, threshold and power.** t = 0.80 (judgement value) separates "most human-heard repetitions leave a text trace" from
substantial erasure; the transcript-heuristic survival of the same `p0` ASR on DIHANA was 0.491
(`results/asr_survival_models_20261104/README_RESULTADOS.md`, small P0) and literal-repetition survival 14/30 on Plan B
val; neither is a human-validated figure, which is what PG2 adds. **Minimum detectable effect (analytic, normal
approximation to the one-sided binomial test, power 0.80, written now without data):** with n events and design effect
DEFF for recording clustering, the smallest true survival detectable against 0.80 is about

| n events | α = 0.05, DEFF 1 / 1.5 / 2 | α = 0.0167 (Holm first step), DEFF 1 / 1.5 / 2 |
| -------- | -------------------------- | ---------------------------------------------- |
| 20       | 0.56 / 0.51 / 0.46         | 0.52 / 0.45 / 0.40                             |
| 30       | 0.61 / 0.56 / 0.52         | 0.57 / 0.52 / 0.47                             |
| 50       | 0.65 / 0.62 / 0.59         | 0.62 / 0.58 / 0.55                             |
| 80       | 0.68 / 0.66 / 0.63         | 0.66 / 0.63 / 0.60                             |

So only substantial erasure (true survival around 0.6 or lower) is detectable at the expected event counts. At stage 2 a
**simulation** is added (descriptive, changes no rule, threshold, floor or family): using only the committed per-recording
`H_disf` `REP` event counts and synthetic survival outcomes (beta-binomial per recording, intra-recording correlation
ρ in {0, 0.1, 0.3}, true survival on a grid 0.40-0.80), seed 20261108, 1 000 simulated cells, each passed through the same
recording bootstrap and Holm rule; the output is the simulated power curve. No ASR output is used or simulated from data.

**PG3 rules.**

- Human spans: adjudicated marker rows with `marker_token` in the closed list; `uncertain` excluded (the uncertain rate is
  reported). Spans **`not_transcribed`** by the `o2026_12` ASR are excluded from κ and reported separately as an **erasure
  count** (and share).
- κ undefined on the full-cell point estimate (a constant margin) -> PG3 **not evaluable**, κ `not_available` with the
  reason. If more than 10 % of the bootstrap resamples are degenerate (κ undefined), PG3 is **not evaluable**; otherwise
  degenerate resamples count against the hypothesis (section 9).
- Companions, always reported: the full 2x2 table, precision and recall of the system's `disfluent` against human
  `disfluent`, PABAK (= 2·p_o - 1), and the erasure count.
- Construct overlap (section 12) is declared: a supported PG3 says the system's state agrees with the human label better
  than chance, not that the system recognises discourse function.

### 10.2 Secondary family (exploratory; Holm within the family, fixed size 6; no global FWER; never alters 10.1)

| id   | Contrast                                                                                                                    | Test                   | Predicted                                                                                                                           |
| ---- | --------------------------------------------------------------------------------------------------------------------------- | ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| PGS1 | pooled survival of `H_disf` `FP` events in `p0` text, t = 0.80                                                              | one-sided, H0: >= 0.80 | < 0.80                                                                                                                              |
| PGS2 | `REP` survival, `p1_disfluent_prompt` - `p0`                                                                                | two-sided              | > 0 (prompt helps repetitions; DIHANA +0.23 heuristic)                                                                              |
| PGS3 | `FP` survival, `p1_disfluent_prompt` - `p0`                                                                                 | two-sided              | < 0 (prompt trades away filled pauses; DIHANA -0.24 heuristic)                                                                      |
| PGS4 | convention effect: F1(`p0_rule` vs `H_lit`) - F1(`p0_rule` vs `H_disf`), repetition, on `S_dec`                             | two-sided              | > 0 (counting functional repeats as positives favours the rule)                                                                     |
| PGS5 | `H_disf` arm (total): F1(`p0_rule` vs `R_T`) - F1(`p0_rule` vs `H_disf`), repetition, on `S_dec`                            | two-sided              | > 0                                                                                                                                 |
| PGS6 | ranking DiD = [F1_R_T(`p0_rule`) - F1_R_T(`default`)] - [F1_H_lit(`p0_rule`) - F1_H_lit(`default`)], repetition, on `S_lit` | two-sided              | > 0 (protocol F H2's advantage of the rule over the default shrinks under the human reference); the `H_disf` version is descriptive |

### 10.3 Secondary arm cw2 (exploratory; Holm, fixed size 3; non-commercial licence; reported separately, as annex F)

| id   | Contrast                                                                                              | Predicted |
| ---- | ----------------------------------------------------------------------------------------------------- | --------- |
| PGC1 | Δ_origin,cw2 = F1(`cw2_rule` vs `R_T`) - F1(`cw2_rule` vs `H_lit`), repetition, on `S_lit`, two-sided | > 0       |
| PGC2 | `REP` survival, `cw2` - `p0`, two-sided                                                               | > 0       |
| PGC3 | `FP` survival, `cw2` - `p0`, two-sided                                                                | > 0       |

### 10.4 Candidate arm FT (exploratory; Holm, fixed size 3; only if OD-4 is met at the stage-2 freeze, otherwise removed and `not_available`)

| id   | Contrast                                                                                                                                                               | Predicted |
| ---- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------- |
| PGF1 | `REP` survival, `ft_small_v1` - `p0`, two-sided                                                                                                                        | > 0       |
| PGF2 | `FP` survival, `ft_small_v1` - `p0`, two-sided                                                                                                                         | > 0       |
| PGF3 | **ASR-plus-decision contrast** (the two systems differ in ASR and in decision; no attribution to either): repetition F1 vs `H_disf`, `o2026_13` - `default`, two-sided | > 0       |

### 10.5 Absolute criteria (CI-based, outside every family; judgement floors, the same values as protocol F)

- **PGA1** repetition, `default` vs `H_disf`: lower 95 % bound of F1 >= 0.30 (protocol F A1 floor).
- **PGA2** filled pause, `default` vs `H_disf`: lower 95 % bound of recall >= 0.40 (protocol F A2 floor). Precision and F1
  are reported too: the human listened to the whole clip, so an unmarked item is a verified negative within the
  reference's reliability.
- **PGA3** `default` repetition F1 vs `H_disf` minus `always_present` F1 vs `H_disf`: lower bound > 0.

### 10.6 Descriptive (no p-value, no verdict)

Per system and reference: TP/FP/FN, precision, recall, F1 with intervals, for REP and FP, under `R_T`, `H_lit`, `H_disf`,
with prevalence per reference. The decomposition of section 9. **Ranking change:** systems ranked by F1 under `R_T` and
under `H_lit` (and `H_disf`); Kendall τ_b between rankings with a bootstrap interval; every pairwise order flip listed with
its paired interval. Survival per ASR and type with intervals (including `PRO` word presence). C3: human uncertain rate
among marker spans, per-token counts, transcription coverage and erasure count per ASR, the 2x2 table of `o2026_12`, the
same for the rule on `p0`/`cw2`/`ft` words, and the QG-5 marker detection figures. Event-level detection F1. Intra- and
inter-rater κ/α per class from section 7, repeated in the result. Events without item, per system. Aligner gate counts and
untimed-token counts.

### 10.7 What counts as failure

(a) PG2 contradicted or PGS1 contradicted (`p0` keeps >= 0.80 with significance): the erasure claim is not validated by
humans and is reported as such. (b) PG1 not supported: no evidence, on this cell, that the transcript-derived reference
changes the rule's F1; protocol F's circularity caveat stays a caveat, not a measured effect. (c) PGA1, PGA2 or PGA3 not
met. (d) PG3 not supported: the system's marker state is not shown to agree with the human label better than chance.
(e) More than 1 of 19 recordings failing for a primary system makes that system not interpretable. Nothing is rerun,
re-thresholded or re-framed after the fact.

### 10.8 Expected outcome - uncalibrated guesses written before any output

These are the author-side guesses of an agent, informed by protocol F and plan A results on sibling recordings and DIHANA;
they are **not calibrated predictions**, carry no evidential weight and change no rule. They are written so that the result
can contradict them.

- PG1: Δ_origin roughly in [+0.05, +0.30]. PGS6: DiD roughly in [0.00, +0.20], "not demonstrated" likely for lack of power.
- PG2: `p0` REP survival roughly in [0.30, 0.65]; PGS1: `p0` FP survival roughly in [0.30, 0.65].
- PG3: no κ range is guessed (construct overlap, section 12). Human uncertain rate among marker spans roughly in
  [0.15, 0.45].
- PGS2 in the predicted direction plausible; PGS3 direction as predicted, possibly not demonstrated. PGS4 > 0 only if the
  reference contains functional repeats (count unknown).
- `cw2` REP survival >= 0.85, FP >= 0.75. `ft_small_v1` out of domain: REP survival roughly in [0.50, 0.85].
- `default` repetition F1 vs `H_disf` roughly in [0.15, 0.40]; PGA1 likely not met.
- Several classes may fail the evaluability floors (about 20-35 positive REP items guessed; PRO few).

## 11. Sensitivity analyses (pre-declared, descriptive, outside every family; never change a verdict)

S-MARIA (exclude `maria`, OD-1), S-MARIA-ENRIQUE (also `enrique2`), LOSO tables for `default`, `p0_rule` and every primary
contrast (`cristina-mem` carries 29 % of the items), S-UNC-POS / S-UNC-NEG (uncertain-only items counted as positive /
negative under `H_disf` instead of excluded), S-TOL (±100 / ±500 ms), S-OVERLAP (any-overlap mapping), S-NOGATE (no aligner
gate), S-P0ANCHOR (`p0`-word anchoring of `R_T`, section 4.3, declared circular for `p0_rule`), S-ALLSTATES (C2 survival
over events of any state), S-RATER (C1 and C2 recomputed with pass 1 alone and pass 2 alone instead of the adjudicated
file).

## 12. Threats to validity (declared before measuring)

- **Single primary rater = the author** (self-annotation, declared deviation from `PREREGISTRO.md` section 7). The author
  wrote the guide and knows the engine's heuristics. Blinding to system output reduces expectation bias, does not remove
  it. Intra-rater agreement measures consistency, not shared construct; the rater-2 subset checks this only partially.
  Every figure carries "blind single-rater reference (author), intra-rater κ = ..., inter-rater κ on n = ... items = ...".
- **The author knows protocol F and plan A results** (on the 14 sibling recordings and on DIHANA), including which
  conventions favour the rule and how the profiles behave. This knowledge can shape labelling choices in borderline cases
  even when blind to system output on these recordings. Declared; it is a reason for the rater-2 subset and for reporting
  S-RATER, not something they remove.
- **Construct overlap in C3.** The guide's `disfluent` requires an adjacency cue (silent or filled pause, prolongation,
  restart); the engine's lexical-filler rule calls `disfluent` on an adjacent gap of at least 0.15 s, a duration of at least
  0.35 s or a comma. Both look at pause adjacency, so agreement above chance may come from the shared cue rather than from
  function recognition. PG3 is therefore a "better than chance" check, no κ range is predicted, and no claim of function
  recognition is made.
- **Calibration overlap.** The 20 (or 40) calibration clips are disjoint windows of the **same 19 recordings**. Their labels
  are seen before stage 1; they reveal something about the prevalence of each class in this cell. The values of this draft
  were written before calibration; any value changed between this draft and stage 1 is listed as "changed after
  calibration" with the calibration figures seen.
- **Post-annotation discretion at stage 2.** The harness is written (or finished) after the annotation counts are known.
  The rules it implements are frozen at stage 1, it is tested on synthetic data only, and any deviation from the stage-1
  text is a declared annex; building as much of the harness as possible before pass 1 is recommended.
- **One corpus, one task** (picture narration), **7 speaker labels**, unequal items per speaker, **dialect undocumented**.
- **Speaker overlap with the spent cell** (`maria`; `enrique2` unknown) and the same corpus, task and recording setup.
  Systems `o2026_12`, `o2026_13` and `cw2_rule` were developed or chosen **after** protocol F's result on the 14 sibling
  recordings was known; their choice is informed by the same corpus, though not by these recordings.
- **`R_T` timing comes from a model outside the tested systems** (section 4.3). Its errors (read-speech training, character
  vocabulary, transcript-audio mismatch) move events between items; the aligner gate uses only the aligner's own
  confidence and applies to every system and reference equally; S-NOGATE and S-P0ANCHOR show the dependence. The
  transcript was not designed for disfluency annotation (protocol F section 7).
- **Unit mismatch:** human `REP` spans cover first to last copy, rule events cover the second copy; C1 is item-level to
  avoid this, event-level figures are descriptive.
- **Sampling uniform in time:** rare classes may be unevaluable; item edges cut events (`borde` note).
- **Recording-level clustering** with 19 recordings of 7 speakers: anti-conservative intervals, mitigated (not removed) by
  the LOSO requirement.
- **Small n:** with about 20-40 events per class, intervals will be wide and the minimum detectable effect large (section
  10.1); "not demonstrated" does not mean "no effect".
- **Determinism:** `environment_is_pinned = False`. Library versions, OS, CPU/GPU, threads and driver are recorded, not
  pinned; bit-reproducibility of ASR text (which depends on `OMP_NUM_THREADS` and machine load) and of GPU emissions is not
  claimed. Unloaded machine. `cw2` needs CUDA (no CPU fallback); a failure makes that arm `not_available`.
- **Licences:** NOCANDO CC BY-NC-SA 3.0 recorded but not verified at the source for annotation use (OD-5); `cw2` weights
  and outputs are non-commercial research only and its transcripts are never stored or published; the aligner checkpoint
  is Apache-2.0 per its model card (to be re-read at stage 1).

## 13. Outputs and licence

`results/protocol_g/RESULT_G.json` (aggregates, counts, gates, hashes, environment, system commit; **no corpus text, no
transcript, no per-unit ASR text**), `RUN_G.lock.json`, local-only per-item predictions (`local/`, git-ignored). Only
aggregates and, once OD-5 is resolved, annotation ids/labels are shared. NOCANDO: Brunetti, Bott, Costa, Vallduví (2011),
CC BY-NC-SA 3.0. CrisperWhisper 2.0: Nyra Health Non-Commercial Research License.

## 14. Locks, hashes, two-stage freeze, single shot, annexes

- **Stage 1 - `PROTOCOL_G.stage1.sha256.json`** (before pass 1): sha256 of `PROTOCOL_G.md` (CRLF normalised to LF), of the
  frozen `GUIA_ANOTACION.md`, `PROTOCOL_ANNOTATION.md`, `ANNEX_20261108_SELF_ANNOTATION.md`, both manifests, kit v1.1
  `scripts/annotation/agreement.py` and `markers.py`, the 19 mp3 (must equal the manifest), the 19 transcript files (bytes
  only) and the aligner weights hash of section 4.3. The OSF pre-annotation registration cites the stage-1 commit and the
  sha256 of `PROTOCOL_G.md`; `OSF_REGISTRATION_STAGE1.json` records it.
- **Stage 2 - `PROTOCOL_G.stage2.sha256.json`** (after the annotations are committed, before any system or aligner run):
  sha256 of every annotation CSV and sidecar, the agreement JSONs, `REFERENCE_GATES_G.json`, the harness and its helper
  modules and tests, `SYSTEM_FREEZE_G.json` and the OD-4 record; it repeats the stage-1 hashes, which must be unchanged.
  Uploaded to the OSF project as a dated document; `OSF_REGISTRATION.json` records the upload time.
- **`SYSTEM_FREEZE_G.json`** (`freeze-system`, stage 2): system commit (>= 12 hex); sha256 of every tracked file of
  `mrcd/`, `models/`, `pyproject.toml`; resolved default decision profile (must be `o2026_11`) and the other profile names;
  lexical-filler constants (must equal section 5) and the sorted `LEXICAL_FILLERS`; ASR identities and weight hashes -
  Whisper small CT2 int8 `model.bin` shared by `p0`/`p1` (expected sha256
  `3e305921506d8872816023e4c273e75d2419fb89b24da97b4fe7bce14170d671`, the value recorded in earlier MRCD service artefacts),
  `cw2` revision `f4334f6e8193f2691212d49b20fa12d370e13896` with weights manifest sha256
  `b22cf7a5208e73405514f32b9e7686019364771d51a8a8683f105596d88e7ac9` (annex F), `ft_small_v1` `model.bin` sha256 if OD-4 is
  met; library versions (expected as in protocol F: `faster-whisper` 1.2.1, `ctranslate2` 4.8.2; plus `torch`,
  `transformers`, `numpy`, `scipy`, `soundfile`, `librosa`); `environment_is_pinned: false`. A difference in the profile
  name, the lexical-filler constants or a weights hash makes `run` refuse; a different library version is recorded as a
  declared provenance difference. `run` also refuses if the tree differs or the freeze commit is not an ancestor of HEAD.
- **Leakage check** at the system commit (expected: no hit). Search terms: the 19 file stems of the manifest (whole-token
  match) and the 19 transcript sha256 digests. **Allow-list:** `docs/annotation_20261108/`, `results/protocol_g*/`,
  `scripts/protocol_g/`, `tests/test_protocol_g*`. Any hit in a tracked file outside the allow-list, or in any file
  (tracked or ignored, caches included) under `results/` outside the allow-list, fails the check and `run` refuses. The
  corpus name is not a search term (it legitimately appears in audit, plan and protocol-F code).
- **Single shot:** `run` creates `RUN_G.lock.json` with O_EXCL before loading anything and refuses to start again. A
  finished run (`RESULT_G.json` exists) can never be repeated. A crash (lock `started`, no result) can be continued only
  with `--continue-after-crash ANNEX_<date>.md`, an annex that states the crash and changes nothing else; the continuation
  repeats the run from the start with identical inputs. The `cw2` arm runs in the same process after the primary systems;
  its failure follows annex F section 7 and does not affect the primary result.
- **Annex rules.** Between stage 1 and stage 2, only the three stage-2 additions are allowed without an annex. Any other
  change after pass 1 has started is a dated annex with its own hash, **declared post-annotation**, listing exactly which
  annotation counts, agreement figures or gate values the author had seen when writing it; it may add, never alter, a
  family, a gate or a threshold, and it is uploaded to OSF. After stage 2, an annex can never be written after any output
  of the affected arm exists, except to declare a crash or a deviation that already happened (stated as such, with no
  change of analysis).

## 15. To build (declared; nothing here has been built or run)

Preferably before pass 1 (it reduces post-annotation discretion); at the latest at stage 2:

- `scripts/protocol_g/run_protocol_g.py` (`freeze-system`, `verify`, `run`, `gates`) importing the protocol F harness and kit
  v1.1 `scripts/annotation/agreement.py` unchanged; CTC emissions and numpy Viterbi aligner (section 4.3); `S_lit`/`S_dec`
  construction; C2 survival; C3 matching with `not_transcribed`; PG3 degenerate-resample rule; LOSO; families and
  sensitivities; leakage check with the allow-list.
- Tests on synthetic data with a stub analyser and synthetic emissions (as protocol F section 10): reference parity with
  `transcript_reference`, Viterbi alignment on synthetic emission matrices, aligner gate, decomposition identity on
  `S_dec`, gates (including verbatim reads of `subset_below_minimum` and `function_kappa_descriptive_only`), lock,
  freeze/verify, refusals (profile name, constants, hashes), `not_available` paths, Holm family sizes fixed, primary result
  byte-identical with and without the cw2 arm.
- `REFERENCE_GATES_G.json` generator (reads annotation files only) and the stage-2 power simulation (section 10.1).

## 16. Language and attribution

Outputs and comments in English; the annotator guide is in Spanish. No personal paths; the corpus root is an environment
variable. Never "no system exists"; never inference of psychological states; absence is `not_available`, never 0; no
uncalibrated figure is presented as a result (every threshold here is a declared judgement value).

## Revision log

Revision 2 (2026-11-08) vs commit `25ce1f1` (revision 1), after a fresh-context review ("merge after fixes"):

- **Freeze timing (H1).** Single freeze after annotation replaced by a two-stage freeze: stage 1 (everything that defines
  the test, OSF pre-annotation registration) before pass 1; stage 2 limited to harness code, `SYSTEM_FREEZE_G.json` pins and
  the OD-4 record; other post-annotation changes are declared annexes stating what was seen. Header, section 2, section 14
  and the OSF checklist rewritten. OD-1 is the only open decision (default keep + sensitivity, closed before pass 1);
  OD-2 to OD-5 closed as rules.
- **`R_T` timing (H2).** Primary timing = pinned CTC forced aligner (`jonatasgrosman/wav2vec2-large-xlsr-53-spanish`, revision
  and weights hash of `results/asr_survival_ctc_20261107/`), independent of every tested system; `p0`-word anchoring moved
  to sensitivity S-P0ANCHOR; the system-output-dependent anchoring gate replaced by an aligner-confidence gate. PG1 claim
  reworded to "reference difference (transcript origin), with timing from an independent aligner".
- **Reference of PG1 (H3).** PG1 against `H_lit`; `H_disf` arm moved to the secondary family (PGS5); explicit decomposition
  origin + convention on one item set `S_dec`; item sets identical for both references of each contrast; prevalence
  reported per reference.
- **Marker hypothesis (H4).** `not_transcribed` spans excluded from κ and reported as an erasure count; undefined κ ->
  not evaluable; > 10 % degenerate resamples -> not evaluable; companions 2x2, precision/recall of `disfluent`, PABAK;
  criterion "lower 95 % bound > 0", labelled a better-than-chance check; construct overlap declared; no κ range predicted.
- **Clustering (M1).** "Supported" also requires the LOSO point estimate in the predicted direction for >= 6 of 7 speakers;
  wording "supported on these recordings".
- **Families and power (M2).** DiD ranking (old G2) moved to the secondary family (PGS6, now against `H_lit`, `H_disf`
  version descriptive, for consistency with PG1); primary family size 3. Analytic minimum-detectable-effect table for PG2
  and a stage-2 simulation without real outputs. Floor rationale corrected (a small cell needs more evidence, not less).
- **Rater-2 gate (M3).** "Not assessable" (conditional verdict, annotated) separated from "failed" (descriptive); QG-3 reads
  `subset_below_minimum` verbatim.
- **Leakage check (M4).** Restricted to the 19 stems and 19 transcript sha256 digests with an explicit path allow-list;
  the corpus name is no longer a search term.
- **Pins (M5).** Resolved default profile `o2026_11` (refusal on mismatch), Whisper small `model.bin` hash, `faster-whisper` /
  `ctranslate2` versions, lexical-filler constants; `environment_is_pinned = False` stated; OD-4 decided by merge status
  alone at the stage-2 deadline; PGF3 labelled an ASR-plus-decision contrast.
- **Names (M6).** Hypotheses renamed G1-G4 -> PG1 (old G1, new reference), PGS6 (old G2), PG2 (old G3), PG3 (old G4);
  GS -> PGS, GC -> PGC, GF -> PGF, GA -> PGA, also in the OSF checklist.
- **Low items.** QG-5 reads `function_kappa_descriptive_only`; QG-2 cites the 30-matched-span minimum; the two 30-span minima
  distinguished; QG-1 >= 5-events clause; calibration overlap stated; QG-1 distinguished from the calibration stop rule;
  failed recordings in C1 excluded instead of all-zero; expected outcomes labelled uncalibrated guesses; threat that the
  author knows plan A / protocol F results; secondary families labelled exploratory with no global FWER; pooled event
  survival with recordings resampled; kit v1.1 class list (`discourse_marker`, `state=none`) adopted; OSF manifests uploaded
  only as hashes until OD-5 is resolved.
