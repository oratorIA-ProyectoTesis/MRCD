# Confirmatory evaluation of the MRCD service default - protocol (2026-11-07)

**Status.** Written and frozen BEFORE any MRCD output exists for the cells below. The sha256 of this file, of the harness
and of the data bytes is in `PROTOCOL.sha256.json`; the harness refuses to run if any of them differs. A deviation must be a
dated annex (`ANNEX_<date>.md`), never an edit of this file. Authored by a software agent; self-review only; no independent
reviewer and no advisor ratification. The evaluation described here has **not been run**.

**Decision D-1 (given).** No new human annotation and no own recordings as accuracy evidence: only existing labels of
available corpora. `CONTAMINATION_AUDIT.md` (same directory) is the evidence for which cells are still unread.

## 1. What this test can and cannot claim

| Class | Cell | Reference | Status of the evidence | Claim it can support |
|---|---|---|---|---|
| repetition (literal) | NOCANDO, 14 timed recordings / 5 speakers | transcript-derived (`rep_literal`, same frozen rule as CHM150) | **confirmatory** (cell untouched; reference circular for lexical comparators) | A figure for *literal repetition detection against a transcript-derived reference* in Spanish narrative speech of 5 speakers; nothing about human-judged repetition disfluency |
| filler_word | NOCANDO, same recordings | transcript tokens `hm`, `eh`, `em`, `mm`, `ah`, `e` (not the conjunction) | **confirmatory, recall only** (counts of such tokens were published by an earlier audit; no system output was ever compared) | Sensitivity to *transcribed* filled pauses. No precision, no F1: an untranscribed filled pause is not a verified negative |
| prolongation | **no untouched audio cell exists** | - | **second look**, not confirmatory: AMERESCO segments outside the spent 98-segment pool (same 4 conversations, same speakers) | A recall figure with its contamination declared. Never called confirmatory or "first look" |
| revision, block, rhetorical_pause, neutral_pause | none | no compatible reference | out of the evaluated scope (`INFORME_OE4_REDUCIDO.md`, section 1) | nothing |

Not claimable by anything here: performance on human-annotated disfluency, performance of the Peruvian or any speaker
population beyond 5 NOCANDO speakers, calibration, video/multimodal benefit, cross-corpus generalisation of prolongation.
DIHANA fold 5, CHM150 repetition, EPIC (all 39 speakers used), the text corpora test buckets (no audio) are spent or
unusable (audit rows 3, 6, 7, 11-13, 16). A figure on them would be a replication, not a confirmation.

If the gate of section 6 fails for a class the verdict for that class is **"no confirmatory evidence available"**, which is
neither a success nor a failure of the engine.

## 2. Systems (all frozen; nothing is fitted, selected or thresholded on these cells)

The system commit is a parameter (`--system-commit`, >= 12 hex chars) and is recorded in every output. It is frozen by
`freeze-system` (section 9) AFTER the other improvements are merged and BEFORE the run; the freeze stores the sha256 of
every tracked file of `mrcd/`, `models/` and `pyproject.toml`. The run refuses if the tree differs or the freeze commit is
not an ancestor of HEAD.

* **default** = `MrcdAnalyzer.analyze_bytes` with `AnalysisOptions(decision_profile=<service default at the system commit>,
  asr_profile="p0", head=None, text_route=None, decision=False, output_scope="evaluated", include_text=False)`. The profile
  name is read from `mrcd.service.decision_profile.DEFAULT_DECISION_PROFILE` and written to the freeze and the result. Only
  `in_scope` events of the three evaluated classes are used. Engine: packaged checkpoint, Whisper `small`, CPU int8,
  `OMP_NUM_THREADS=4`, `HF_HUB_OFFLINE=1`.
* **v0** = same, `decision_profile="v0"` (the engine's historical window decision). For `repetition` the current default
  keeps the engine decision, so default = v0 there; the result records `default_events_equal_v0_events`.
* **rule** = lexical comparator on the system's own ASR words (the words of the single decode shared by both profiles):
  repetition = immediate exact repetition of 1-4 tokens after removing non-lexical filled pauses (event = the second
  occurrence); filler = tokens matching the non-lexical filled-pause pattern (conjunction `e` excluded). Same functions as
  the reference, applied to the ASR text instead of the transcription.
* **always_present** (repetition only) = every unit positive. A trivial floor, not a competitor.
* One ASR decode per recording is shared by default, v0 and rule (`CachingASR`), so the profiles differ only in their
  decision, not in ASR noise.

## 3. Data cells (exact)

Root: `$MRCD_CORPORA_ROOT` (no path is built in). File bytes are hashed in `PROTOCOL.sha256.json["data"]`.

**Confirmatory cell: NOCANDO Spanish, 14 aligned recordings** (`02_NOCANDO_ES/alineamientos/*.TextGrid` with the same-stem
`audio/*.mp3`): `berta_es_{1,2,3}`, `carmen_es_{1,2,3}`, `enrique_es_{1,2,3}`, `maria_es_{2,3}`, `nuria_es_{1,2,3}` - 5 speakers
(`berta, carmen, enrique, maria, nuria`), 3 488 s of audio in total. The 19 other mp3 files and the 26 transcripts without
alignment are NOT used (no timing, so no unit-level mapping). Speaker = file stem without `_es_<n>`.

* **Unit** = a non-empty interval of the FIRST interval tier of the TextGrid (named `transcript`, `transcription`,
  `Transcript` or `Tras` according to the file inventory; chosen by position, not by name).
* **Reference** (computed by the harness from the interval text only, never from system output):
  tokens = lower-case, accents removed, non-`[a-z0-9n]` replaced by space; non-lexical filled pauses
  (`^(e+h*|eh+|ehm+|em+|mm+|hm+|ah+|aa+h*)$`, bare `e` excluded before words starting `i`/`hi`) are separated from the lexical tokens.
  `rep_literal` = 1 iff the lexical tokens contain an immediate exact repetition of 1-4 contiguous tokens; `fill_ref` = 1 iff at least one
  filled-pause token. This is the CHM150 rule (`mrcd.context_eval.transcript_reference`), vendored in the harness and
  checked against it by `tests/test_confirmatoria_harness.py`. `_`, `#` and bracket marks of the NOCANDO transcription
  have no documented meaning and are not interpreted (they only disappear in normalisation). `este` is not a filler.
* **Audio**: mp3 -> channel mean -> 16 kHz (`librosa.resample`) -> PCM_16 WAV bytes in memory (the recipe of
  `scripts/baselines/ameresco_prepare.py`). The WHOLE recording is analysed once (all are >= 13 s, the service minimum).
* **Mapping**: each event goes to the unit with the largest temporal overlap (events shorter than 20 ms are widened to
  20 ms); events overlapping no unit are dropped and counted. A unit is predicted positive for a class if it received >= 1
  event of that class. A recording that fails (service error) keeps all-zero predictions (counted as no detection, reported).

**Second-look cell (NOT confirmatory): AMERESCO, 4 conversations** (`12_AMERESCO/<id>/*.mp3` with
`source_annotation_index.csv`). Segments with no automatic exclusion reason whose duration is OUTSIDE 3-8 s (the 98 pool
segments are excluded). Reference = `<alargamiento/>` presence (`source_alargamiento_present`). Whole conversation analysed
once (610-1767 s; `max_duration_s=3600` of the analyser). Contamination: same recordings and speakers as the spent pool;
segment-level outputs were never scored; negatives are unmarked, not verified. Reported: recall of marked segments with an
exact Clopper-Pearson interval (segments treated as independent: optimistic, declared) and the flag rate in unmarked
segments (explicitly not a false-positive rate). No test, no Holm membership, no success criterion.

Declared and not run: EPIC unused sentences (74 filled-pause sentences of already-seen speakers; needs the video
inventory) and CHM150 `e`/`em` (reference weak, utterances < 13 s). They would be second looks too; they are not part of
this run.

## 4. Statistics (decided before any output)

* Cluster = **recording** (14); NOCANDO has only 5 speakers, so a speaker bootstrap would be degenerate. Recordings of one
  speaker are dependent, so the intervals are anti-conservative; this is declared, and the leave-one-speaker-out F1 of the
  default repetition detector is reported as a robustness view. Bootstrap: resample recordings with replacement,
  seed 13; 95 % percentile CI with 2 000 resamples; tests with 5 000 resamples.
* Repetition: precision, recall, F1 for default, v0, rule, always_present over all units. Filler: recall over reference-positive
  units (default, v0, rule).
* Tests (unit-paired, cluster bootstrap): p two-sided = `min(1, 2*min(P(d<=0), P(d>=0)))` with +1 correction; p one-sided =
  `P(d<=0)` with +1 correction.

## 5. Hypotheses, multiplicity, success and failure

**Holm family, fixed size 4** (a test that cannot be computed still counts as a member):

| id | Class | Contrast | Sides | Reading of "supported" |
|---|---|---|---|---|
| H1 | repetition | F1(default) - F1(always_present) | one-sided, > 0 | non-trivial: beats a unit-everywhere detector |
| H2 | repetition | F1(default) - F1(rule) | two-sided | the direction found is reported (`direction` field) |
| H3 | filler | recall(default) - recall(v0) | two-sided | the product profile changed sensitivity versus the historical decision |
| H4 | filler | recall(default) - recall(rule) | two-sided | MRCD acoustics versus a one-line lexicon on the same ASR text |

A hypothesis is **supported** iff `p_Holm < 0.05`, the 95 % interval of the paired difference excludes 0 and (H1) the sign is
positive; otherwise **not demonstrated**. "Not demonstrated" is a result, not an inconclusive run.

**Absolute criteria** (CI-based, outside the family; the floors are judgement values set here, before any NOCANDO output;
they are NOT the OE4 targets 0.80/0.65/0.60, which the earlier reports show unreachable and which would make every outcome
a foregone failure):

* **A1** repetition (default): lower 95 % bound of F1 >= **0.30**. Rationale: the point estimates of the same detector on
  other corpora were 0.35 (DIHANA fold 5, [0.21, 0.48], `INFORME_OE4_REDUCIDO.md:86`) and 0.38 (CHM150, [0.263, 0.484],
  `README_CHM150.md`); a lower bound below 0.30 means the detector is not shown to be usable even at its historical level.
* **A2** filler (default): lower 95 % bound of recall >= **0.40**. Rationale: V0 recall 0.44 on DIHANA fold 5 (64/145,
  `INFORME_OE4_REDUCIDO.md:77`) and a published ASR cap near 0.5 for vocalic fillers in Whisper text.

**What counts as failure.** (a) H1 not supported: the detector is not shown to beat a trivial one. (b) A1 not met: the
repetition figure is not shown to reach its historical level. (c) A2 not met. (d) H2 or H4 supported with `default_worse`:
a one-line lexicon on the same ASR text is better - reported as such, not softened. (e) Any recording with a service error:
listed with its code; a failure rate above 10 % of the recordings makes the comparison not interpretable (`interpretable: false` in the result). Nothing is rerun, re-thresholded or re-framed after the fact.

**Expected outcome, written before the run (so it can be falsified):** default repetition F1 between 0.25 and 0.45; rule >=
default on repetition (H2 supported with `default_worse`, biased in favour of the rule by circularity); filler recall of the
default between 0.30 and 0.55; H1 supported; A1 likely not met. If the run contradicts this, that is information, not an error.

## 6. Evaluability gate (computed from the reference only, before any scoring)

A class is evaluated only if the reference has **>= 30 positive units, in >= 3 recordings and >= 3 speakers**. If not, no metric
or test of that class is produced, its Holm members are "not evaluable", and the class verdict is "no confirmatory evidence
available". The thresholds were fixed from the published order of magnitude of the filled-pause counts and the CHM150
repetition rate (2.9 % of utterances); the actual NOCANDO repetition count is unknown to the author.

## 7. Threats declared before measuring

* Five speakers, one elicitation task (picture narration), one variety, one channel: no generalisation claim.
* The reference is the lexical rule applied to a transcription not designed for disfluency annotation: emphatic repeats and
  lists count as positives, cut-word restarts and unrecorded fillers count as negatives or are missed. It favours
  `rule` (H2, H4) and says nothing about human-judged disfluency.
* Alignment provenance (manual or automatic) of the TextGrids is not documented locally; boundary error moves events between
  adjacent intervals.
* Recording-level clustering understates dependence between recordings of the same speaker.
* The engine analyses whole 2-6 minute narrations, unlike the DIHANA-turn / CHM150 unit recipe of the earlier evidence; this is the
  service condition of use, and also a difference with the figures that justify A1/A2.
* Unit counts depend on the transcription, e.g. a 6 s interval and a 1 s interval weigh the same.
* Determinism: ASR text depends on `OMP_NUM_THREADS` (set to 4 before numpy) and on machine load; the run must happen on an unloaded
  machine (section 9). Cross-environment determinism is not established; the environment is recorded.
* A crash is possible on the 30-minute AMERESCO conversation (memory); see the crash rule in section 9.

## 8. Outputs

`CONFIRMATORY_RESULT.json` (aggregates, counts, hashes, environment, system commit; no corpus text), `RUN.lock.json`,
`local/nocando_unit_predictions.csv` (unit ids and 0/1; local only, `results/` is git-ignored: CC BY-NC-SA 3.0 corpus, only
aggregates are shared). Per-recording documents are not saved.

## 9. Procedure and exact commands

Frozen now (this commit): protocol, harness, tests, data fingerprints. Pending (orchestrator, after the improvements are merged):

```powershell
# 0. leakage check at the system commit: no code outside the audit may mention NOCANDO (expected: no output)
git grep -il nocando <SYSTEM_SHA> -- mrcd scripts ':!scripts/audit' ':!scripts/confirmatoria'
# 1. freeze the system (HEAD must be <SYSTEM_SHA>; the tracked tree of mrcd/ models/ pyproject.toml must be clean)
cd <repo with the merged system>
$env:OMP_NUM_THREADS = "4"; $env:MRCD_CORPORA_ROOT = "<root of the corpus repository>"
.\.venv\Scripts\python.exe scripts\confirmatoria\run_confirmatoria.py freeze-system --system-commit <SYSTEM_SHA>
git add -f results\confirmatoria_20261107\SYSTEM_FREEZE.json
git commit -m "docs(eval): freeze the system for the confirmatory run"
# 2. dry check of every hash (reads the data bytes, runs nothing)
.\.venv\Scripts\python.exe scripts\confirmatoria\run_confirmatoria.py verify --system-commit <SYSTEM_SHA>
# 3. THE run (once; about 2 h on CPU; unloaded machine; no other Python/ASR/Docker job; do not touch the machine)
.\.venv\Scripts\python.exe scripts\confirmatoria\run_confirmatoria.py run --system-commit <SYSTEM_SHA>
```

`run` writes `RUN.lock.json` (O_EXCL) before it loads anything; afterwards it refuses to start again. If the process dies
(lock status `started`, no result) the only allowed continuation is `run ... --continue-after-crash ANNEX_<date>.md`, with a
dated annex that states the crash and changes nothing else; there is no per-recording cache, so a continuation repeats the
run from the start with identical inputs. A finished run (`CONFIRMATORY_RESULT.json` exists) can never be repeated.
Post-run edits to the system make the result about a different system and are forbidden by the hash check.

## 10. Development checks already done (declared)

* Unit tests: `tests/test_confirmatoria_harness.py` (22): reference and statistics parity with the repo modules, TextGrid
  parser (long and short format, on synthetic text), encoding detection, gates, lock, freeze/verify cycle in a temporary git repository.
* **Dry run on SYNTHETIC data with a STUB system** (generated TextGrids and noise audio, a stub analyser that goes through the
  real `asr_wrapper` path), written to a scratch directory outside the repository; its numbers have no scientific content.
* The real service (`MrcdAnalyzer`, profiles `o2026_11` and `v0`, shared ASR via `CachingASR`) was run once on the
  repository's synthetic speech sample `site/sample_es.wav` to check that the shared-decode path works (22.7 s; both profiles
  returned events). No corpus cell was run.
* Audio decoding was checked on one AMERESCO conversation (decode only: 652 s -> 16 kHz mono in 2.6 s), not on NOCANDO.
* What the author inspected in the data cells: file names, durations and sample rates of the NOCANDO and AMERESCO audio; the
  row counts of `source_annotation_index.csv` by recording and duration (labels column not opened); the first 60 bytes of the
  14 NOCANDO TextGrids (encoding headers) and, to validate the parser, the harness parser was run in-process on the 14
  TextGrids printing ONLY the number of non-empty first-tier intervals and the end time of the last interval (1 531 units;
  last interval ends equal the audio durations within 0.1 s, so the alignment spans each recording; the TextGrids are
  UTF-16 with BOM from `berta_es_2` on and Latin-1 for `berta_es_1`, handled by `decode_textgrid`). No interval text was
  printed or read, no reference was computed on NOCANDO, no NOCANDO audio sample was decoded, and no AMERESCO label was opened.

## 11. Language and attribution

Outputs and comments in English. NOCANDO: Brunetti, Bott, Costa, Vallduvi (2011), CC BY-NC-SA 3.0; AMERESCO: CC BY-NC-SA 4.0.
Only aggregates are versioned.
