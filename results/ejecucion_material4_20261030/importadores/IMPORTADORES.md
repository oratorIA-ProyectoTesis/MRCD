# Corpus importers (DIME, ESLORA, DIHANA)

Package `mrcd/corpora/` (new), script `scripts/corpora/import_all.py`, tests `tests/test_corpora_*.py`. Import/verification infrastructure only: no labels are created, no evaluation is run.

## Commands
```
python -I scripts/corpora/import_all.py          # writes this folder (summary, utt74_example.json, eslora_per_recording.csv, data/)
OMP_NUM_THREADS=4 python -m pytest tests/test_corpora_*.py -q     # 127 passed
```
`data/` holds derived text (DIME/ESLORA views) of corpora with unresolved licences: not committed.

## Schema (`mrcd.corpora.base.Example`, frozen)
corpus, recording_id, speaker_id (None unless the source has real ids; DIHANA only), group_id, source_files (path, sha256, member), original_text, original_annotations (tuple of `Annotation`), temporal_unit (`seconds` | `untimed`), start_s/end_s (seconds), source_label, proposed_mrcd_mapping + mapping_justification, coverage {class: pos|neg|unknown} for all 7 classes, negative_basis, provenance (human|automatic|heuristic|unknown), input_view, audio_duration_s, extra.
`Annotation`: source_kind, source_label, provenance, start_s/end_s, text, locator, proposed_class, mapping_strength (exact|approximate|none), justification.

Enforced at construction (errors, not warnings): start<end, inside audio duration (tol 2 ms), no same-class overlaps, untimed examples carry no times, `pos` iff a mapped annotation exists, `neg` only with a written declaration (`negative_basis`), forbidden mappings, `<tags>`/DIME markers/`habN` in `input_view`, WAV rate/channels (16 kHz mono).
ms->s: `ms_to_s(value, MillisecondsPerFrame)` rounded to 1 us; PHN header is read, never assumed.

## Decisions
- **dataset.py contract untouched.** `manifest_human_annotation()` returns `coverage: "complete"` only if every class is pos/neg AND provenance is human; otherwise `"partial"`, which `build_recording_windows` rejects (tested). utt74 therefore cannot generate negative windows.
- **DIME dialogues (20)**: unlabeled material, all classes `unknown`; TXT cues (`<sil>`, `<ruido>`, hyphen fragments...) kept only as untimed annotations without mapping. `speaker_id=None`; `s:/u:` are turn tags, not identities. XLS are never read; `refuse_xls_as_labels` raises, and `dime_xls_*` sources cannot carry a mapping.
- **DIME TXT input_view**: removes `<markers>`, `+`, `s:/u:`, bracket characters; splits `pue-puede` into `pue puede`. Keeps `eh/mm` and literal repeats. Cannot restore what the TXT omits (lengthening, blocks, durations, the 267 ms silence of utt74).
- **utt74**: group `d12` read from `xcorr_samples.csv` (best ncc >= 0.9 and margin >= 0.3; otherwise unresolved -> error). `modification repair` -> `revision` (exact, justified); event 0.537998-1.812573 s; reparandum `un` 0.538-0.6415, alteration `la estufa un` 0.909-1.8126, editing silence 0.2676 s kept as an unmapped annotation. Repetition is not emitted. Provenance `human` rests on the paper (tiers hand-tagged), not on per-file metadata. Other repair types are not mapped.
- **ESLORA**: one example per recording, untimed. `alargamiento` -> prolongation, `approximate`, text-level only; `palabra_cortada`, `pausa`, `pausa_larga`, `silencio`, `vacilacion` kept as annotations with NO class (forbidden-mapping table: fragment -> revision/block; pauses -> block/rhetorical/neutral). Coverage: prolongation `pos` when a tag exists (130/130 recordings), everything else `unknown`. Hab labels are per-recording: `speaker_id=None`. ZIP read in memory with name/size validation (no extraction). License: strict reading (attribution + non-commercial).
- **DIHANA**: `scripts/baselines/dihana_prepare.py` runs `main()` on import, so it is not importable; its rules are reproduced and a parity test compares input text and the 4 presence flags with `turns.csv` (60 XMLs). The 4 families get `neg` under the adapter's convention (manual documents negative coverage); block/pauses `unknown`. `r:repeticion` is `approximate` (it also files cut-word restarts and corrected mispronunciations, see REPETICION_REVISION). `<f/>` is an empty element: the filler word is not in the text and is not reconstructed; `palabra="..."` attributes are never exposed. group = speaker.
- **Leakage**: `Example.model_features()` returns only `{"input_text"}` and runs `assert_features_clean` (target-like keys such as label/coverage/file/group/split; recording/group/file names and response tokens in values).
- **Groups** (`groups.py`): `group_for_name` maps AVI/WAV/TXT/XLS/PHN/cache/segment names to dialogue or recording; utt* via the audited table; conflicting parents -> error. `find_group_crossings`/`assert_no_group_crossing`/`find_duplicate_sources`.

## Run results (this machine)
DIME: 20 dialogues, 3283 utterances, 241.618 min, all unknown; d10 header declares 96 utterances, 100 parsed (matches audit). ESLORA: 130 recordings; tags alargamiento 32,812, palabra_cortada 7,689, pausa 87,972, pausa_larga 26,126, silencio 9,793, vacilacion 782 (identical to audit); 71,657 non-empty turns and 811,575 whitespace words in input views (audit: 74,078 turns / 812,893 words use a different counting; empty-after-stripping turns are dropped here). DIHANA: 6279 user turns, 0 leaks. utt74: revision=pos, six classes unknown, manifest coverage `partial`.

## What utt74 checks and what it does not
Checks: PHN parsing, ms->s, tier bounds vs the WAV header, reparandum/alteration vs word tier, mapping, coverage, group. Does not: any detector performance, any generalisation (n=1, a segment of d12, not independent of it), the negative status of other classes, the label provenance per file.

## Limits
Undefined DIME markers (`[ ]`, `+`, `<vocal>`) are cues only. ESLORA has no audio/times. DIHANA negatives inherit the adapter convention. Full suite (`OMP_NUM_THREADS=4`): 429 passed, 2 skipped, 2 failed, both in `mrcd/live` territory owned by the other agent (`test_live_engine_parity.py::test_live_demo_rows_equal_offline_probabilities_with_the_offline_centre`, `test_live_stream_partitions.py::test_quick_stop_start_leaks_no_stale_rows_or_words`); not touched.


## Addendum 2026-10-31
* The test count above ("429 passed, 2 failed") is stale. Final suite after the round-3 corrections: see `logs/pytest_final.log` (OMP=4) and `logs/pytest_final_slow.log` (MRCD_SLOW=1).
* Forbidden-mapping guard hardened (REVISION3 section 1): labels are canonicalised (`canon_label`: lower case, no accents, only letters and digits), so `palabra cortada`, `palabra-cortada`, `PALABRA CORTADA`, `.sil`, `<sil>`, `Pausa Larga`, `silence`, `sp`, `fragmento`, `pausa_corta` are refused; `Example.source_label` / `proposed_mrcd_mapping` are validated (known class, not forbidden, backed by an annotation); any source kind containing `xls` is refused whatever its case or suffix. Not done: non-string/recursive scan and case-folded identifiers in `assert_features_clean`, ZIP duplicate names / size cap, `ms_to_s(nan)` (low severity, listed in REVISION3).
