# Annex 2026-11-08: declared deviation from PREREGISTRO §7 (self-annotation)

Status: DRAFT drafted 2026-11-07; revised 2026-11-08 (pre-annotation, still unsigned: no annotation had started and the annex had not been committed/signed; see §5). This file must be committed, and its commit hash recorded in the README, **before any annotation pass starts**. Any later change is a new dated annex, never an edit.

## 1. Deviation

`PREREGISTRO.md` §7 requires two external annotators and states that the author does not annotate. That condition **will not be met** in this campaign.

- **Cause:** there is no budget and no external annotator available.
- **What changes:** the thesis author is the primary (and only first-line) rater, under the controls below.
- **What does not change:** the sampling design, the three-state scheme, the class set, the guide (`GUIA_ANOTACION.md`), and the rule that agreement is computed before adjudication.
- **Scope note:** §7 of the preregistration concerns validating the LLM judge on a sample stratified by judge label. This campaign is different: a time-stratified sample of unlabeled audio meant to become a new reference cell for the engine. This annex does not amend the §7 judge-validation design; if the same data is to serve that design, it needs its own annex.

## 2. Compensating controls

**(a) Blind to the system.** The author never sees system output (MRCD, Whisper, Speechmatics, any detector, any transcript) for these clips. No system touches the audio before the annotation CSVs are committed and hashed. The existing record that no system has run on NOCANDO audio is `results/confirmatoria_20261107/CONTAMINATION_AUDIT.md` (row 19, "No system was ever run on NOCANDO audio"); that audit is a record, not a guarantee, and nothing is re-verified here.

**(b) Blind to provenance.** Clips are presented under random blind ids with no file name, speaker or offset, in an order re-randomised per pass (`scripts/annotation/blind_session.py`, one seed per pass). The id-to-clip key is a local file outside the repository.

**(c) Two passes, intra-rater reliability.** The author annotates all items twice, at least 7 days apart, with fresh blind ids and a fresh order. Intra-rater Cohen κ and Krippendorff α (nominal), per-class agreement, temporal IoU and boundary agreement are computed with `agreement.py --mode intra` **before** the author adjudicates his own disagreements, and reported as is.

**(d) Second rater on a subset.** A native Spanish speaker (unpaid; e.g. the advisor or a peer, never someone who has seen system output) annotates a random subset of at least 40 items, drawn with a seed declared beforehand and independent of any label. Inter-rater κ/α on that subset is computed with `--mode subset` and reported as is, including if low. A subset below 40 items is flagged by the script and must be reported as such.

**(e) Hash before any system run.** The raw pass-1 CSV, pass-2 CSV and rater-2 CSV are committed with their sha256 (`<file>.sha256`) **before** any system runs on the audio. `unblind.py` refuses to unblind a CSV without a matching hash sidecar.

**(f) What this reference can and cannot support.**

- It **is** a blind single-rater reference with intra-rater reliability and a partial inter-rater check on a subset.
- It **is not** a two-external-rater gold standard. Every table, figure and sentence that uses it must say "blind single-rater reference (author), intra-rater κ = …, inter-rater κ on n = … items = …".
- Intra-rater agreement measures the consistency of one person, not shared understanding of the construct; it can be high for a rater who is consistently idiosyncratic. The subset check addresses this only partially.
- The author wrote the guide and knows the engine's heuristics and the hypotheses. This is an unmeasured threat to validity (expectation bias toward the system's behaviour) and must appear in the limitations. Blinding to system output reduces it; it does not remove it.
- Absence is never zero: a class with too few reference events is reported as "not evaluable (n = …)", never as a number. Pre-declared floor: fewer than 20 reference events in a class means descriptive counts only.

## 3. Pre-declared analysis choices (fixed now)

- Primary agreement statistic: κ on the joint label `class:state` over aligned event units; class-only κ is secondary.
- Event alignment: temporal IoU >= 0.10, greedy one-to-one per item (`agreement.py`).
- Boundary tolerance reported at ±250 ms (primary), with ±100 ms and ±500 ms as sensitivity.
- Calibration stop rule: `PROTOCOL_ANNOTATION.md` §4.
- Adjudication of the author's own disagreements: the written rules in `PROTOCOL_ANNOTATION.md` §7, applied mechanically, never by consulting system output.

## 4. Sign-off

Author: ______________________ Date of commit: __________ Commit hash recorded in README: ______________

## 5. Revision 2026-11-08: discourse-marker function layer

Made before the annex was signed and before any annotation started, so it is part of the pre-registered design and not a post-hoc change.

- **Added:** in the same passes, a human label `marker_function` in `functional, disfluent, uncertain` (plus `marker_token`) for every occurrence of the closed list `este, o_sea, bueno, pues, a_ver, tipo` (`eh`, `em` and `mm` stay `filled_pause`). Definitions use observable cues only (`GUIA_ANOTACION.md` v1.1, §3).
- **Further revisions, same date (2026-11-08), still before sign-off and before any annotation:** (1) exhaustive decision order `functional` > `disfluent` (only with no syntactic role, deletable and at least one adjacency cue) > `uncertain`; (2) the marker layer is declared separate from and stricter than `LF`, with expected disagreement between layers, both analysed separately; (3) `eh` and `em_mm` removed from the closed list; (4) marker agreement now also reports `match_rate` and detection F1 next to κ/α, with κ/α descriptive only if detection F1 < 0.5; (5) the subset-coverage flag (fewer than 40 items) also applies to the marker layer.
- **Unchanged:** classes, existing columns, sampling, controls (a)-(f), event-level agreement. Marker rows are excluded from the event statistics.
- **Reliability on the new field:** intra-rater κ and α on matched marker spans, rater-2 subset agreement, proportion of `uncertain`; not available (with reason) below a pre-declared minimum of 30 matched spans.
- **Cost:** about +2 h of author time (README time budget).
- **Limit:** the same single-rater caveats of §2(f) apply; the author also knows the hypothesis that functional and disfluent markers differ, an unmeasured expectation-bias threat to be listed in the limitations.
