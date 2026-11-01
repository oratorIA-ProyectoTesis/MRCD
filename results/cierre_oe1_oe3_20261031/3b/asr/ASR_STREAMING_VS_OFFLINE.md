# ASR streaming vs offline against HUMAN references (Task 3.2)

Self-review by the executing agent; no independent review. EXPLORATORY. V0, P0 and live_cfg_v2 remain the defaults; P1 is experimental; scores are uncalibrated. DIHANA fold 5 was never used (folds 1-4 only, nothing was tuned on them in this task).
Raw data: `pred/<condition>/<unit>.json` (word, start, end, prob per unit), `asr_analysis.json`, `class_descriptive.json`, `units.json`, `config.json` (seeds, versions, sha256 of units/model/code). Code: `scripts/cierre3b/{prepare_units,asr_compare,asr_analyze,class_descriptive}.py`.

## 0. What the "WER 0.76" was
`PARIDAD_LIVE_V2.md` section 5: pooled word disagreement between two Whisper outputs (offline vs final streamed text) on 8 pilot clips. Neither side is a human transcript, so it measured disagreement, not accuracy. Here the same quantity is re-measured on 67 units and is labelled DISAGREEMENT; accuracy is measured separately against human references.

## 1. Units and references (legitimate human text only)
* **AMERESCO** (75 clips, 4 recordings): audio = the pilot/baseline clip wavs (+-5 s context); reference = the human ELAN (Val.Es.Co.) transcription of every tier whose annotation midpoint lies inside the clip, in time order. Verified human: taken from the corpus `.eaf` files, not from any ASR output. Clips are chosen greedily so that no two clips of a recording share audio. Cleaning (frozen): self-closing tags, `<ininteligible/>`, overlap brackets `[ ]` and `(( ))` removed keeping content, `<tag>inner</tag>` keeps `inner`, cut words ending in `-` dropped. Caveats: overlapping speakers and unintelligible stretches make this reference noisy (only 9 clips with a single tier and no overlap/uncertain/unintelligible marks = subset `ameresco_clean`); boundary words of the clip may be cut; only **4 groups**, so group-bootstrap intervals are not informative.
* **DIHANA folds 1-4** (60 user turns >= 10 s, 15 per fold, 39 speakers, deterministic hash order): reference = `mrcd.corpora.dihana.dihana_input_view` (turn text without response marks; asserted equal to the stored turn text). Caveats: `<f/>` fillers are empty elements, so filler words are NOT in the reference (hypothesis fillers count as insertions; the secondary `nofill` variant drops pure vocalic fillers from the hypothesis); numbers are spelled out in the reference while Whisper may write digits (counted as errors, equally in every condition).
* One unit with an empty reference was dropped (134 evaluated of 135).
* Frozen normalization: NFKC, lowercase, `_` to space, every non-alphanumeric character removed, whitespace collapsed; accents kept (accent-folded and `nofill` columns are in `asr_analysis.json`). Disfluencies are not touched.
* Conditions: offline p0 and p1 (engine path: `transcribe_chunked`, 8 s chunks / 2 s overlap, 134 units); streaming p0 and p1 with a 2 s tick (live path simulated: first tick at 6 s, then every 2 s plus a final tick, ring = last 13 s, `StreamingASR.merge_recent`) on a 68-unit subset (every second unit, 38 AMERESCO + 30 DIHANA; cost budget); streaming p0 with a 0.5 s tick on 8 units (the live tick; 0.5 s tick for p1 was not run: "not measurable" for cost reasons, command ready below). OMP=4 before numpy, CPU int8, same model for all.
* Intervals: percentile 95 % bootstrap resampling **groups** (speaker for DIHANA, recording for AMERESCO), 2000 resamples, seed 20261031.

## 2. WER against human references (pooled, primary normalization)

| condition | units | groups | all (95 % CI) | AMERESCO | AMERESCO clean | DIHANA f1-4 |
|---|---|---|---|---|---|---|
| offline p0 | 134 | 42 | 0.583 (0.299-0.734) | 0.794 (4 groups, CI 0.665-1.083: not informative) | 0.717 (3 groups) | 0.154 (0.131-0.176) |
| offline p1 | 134 | 42 | 0.529 (0.238-0.689) | 0.720 | 0.733 | 0.139 (0.118-0.156) |
| streaming p0, tick 2 s | 67 | 24 | 0.396 (0.208-0.505) | 0.510 | 0.469 | 0.168 (0.139-0.191) |
| streaming p1, tick 2 s | 67 | 24 | 0.401 (0.233-0.509) | 0.508 | 0.452 | 0.190 (0.130-0.269) |
| streaming p0, chunk-aligned (opt-in, section 5) | 67 | 24 | 0.552 | 0.737 | 0.847 | 0.184 |
| streaming p0, tick 0.5 s | 8 | 7 | 0.583 | 0.782 | 0.678 (1 unit) | 0.252 |

The overall number is dominated by AMERESCO (noisy reference, repetition loops, see section 4); the DIHANA column (clean orthographic transcripts, 38 speakers) is the most trustworthy accuracy figure: about 0.14-0.19 for every condition. Absolute WER here is not comparable with published Whisper figures (spontaneous speech, disfluencies kept, no number conversion).

### Paired differences (same units; change in pooled WER; group bootstrap)

| comparison | all | AMERESCO (4 groups) | DIHANA f1-4 |
|---|---|---|---|
| streaming p0 (2 s) minus offline p0 | -0.156 (-0.237 to +0.013) | -0.227 (-0.339 to -0.029) | -0.017 (-0.043 to +0.007) |
| streaming p1 (2 s) minus offline p1 | -0.117 (-0.250 to +0.062) | -0.189 (-0.445 to +0.012) | +0.028 (-0.019 to +0.097) |
| offline p1 minus offline p0 | -0.054 (-0.100 to -0.017) | -0.074 (-0.152 to -0.029) | -0.015 (-0.031 to -0.002) |
| streaming p1 minus streaming p0 (2 s) | +0.006 (-0.012 to +0.043) | -0.003 | +0.022 (-0.022 to +0.090) |
| streaming p0 0.5 s tick minus offline p0 (8 units) | +0.128 (+0.034 to +0.206) | +0.169 | +0.059 |
| streaming p0 0.5 s tick minus 2 s tick (8 units) | +0.020 (+0.004 to +0.043) | +0.009 | +0.030 |

Reading: against the human references, **streaming was not less accurate than offline**; on AMERESCO it was better, on DIHANA the differences are within noise. P1 lowers offline WER slightly (also on DIHANA, but P1 was designed on DIHANA folds 1-3, so that column is not independent evidence; AMERESCO has 4 groups); P1 does not help streaming. A 0.5 s tick was not better than a 2 s tick (8 units only).

## 3. Streaming vs offline as DISAGREEMENT (not accuracy)
Pooled word disagreement streaming(2 s, p0) vs offline(p0): 0.403 (95 % CI 0.195-0.522, 67 units, median unit 0.15); p1: 0.367 (0.180-0.473); 0.5 s tick p0 (8 units): 0.573. Of the 1088 disagreeing tokens (p0, 2 s): 510 deletions, 424 substitutions, 154 insertions relative to the offline text. Word start times of matched words (1766): median difference 0.00 s, |delta| p90 0.14 s, 11 % differ by more than the 0.12 s merge tolerance.

## 4. Where the loss comes from
1. **ASR content errors (floor)**: offline p0 vs human: S 1259, D 662, I 1103 over the pooled references; the insertions are not ordinary mistakes (next point).
2. **Repetition loops in the OFFLINE engine path (concrete defect found, pre-existing)**: 8 of the 75 AMERESCO units contain Whisper hallucination loops in the offline p0 decode (e.g. "No, no, no, ..." about 110 times, `[BLANK _AUDIO]`), accounting for about 805 of the 829 excess adjacent duplicate tokens. The offline outputs of all 75 AMERESCO units reproduce the frozen `asr_cache` word for word (75/75), so this is deterministic existing behaviour of the 8 s chunked path with temperature 0 and no fallback, not a run artefact. `detect_asr_anomalies` flags 7 offline units (3 streaming). The streaming single-pass 13 s decode has loops too (the first inspected unit showed "y que se entramo perfecto" repeated), but fewer: 49 duplicate tokens vs 362 for the offline path on the same 67 units. That is why streaming scores a lower WER on AMERESCO. Changing the engine's decode (e.g. temperature fallback) would need a new experimental profile; not done here, not measured, command pattern ready (`asr_compare.py` with a new profile name).
3. **Chunking / timestamps**: the hypothesis "errors concentrate in the first 1.5 s after each ring start" was tested and is NOT supported: 15.9 % of disagreeing tokens fall in those zones, which cover 14.4 % of the timeline. Timestamps differ by more than 0.12 s for 11 % of matched words (section 3).
4. **MRCD classification (descriptive only; no human window labels exist, so no accuracy)**: with the same acoustic frames and only the ASR words changed, 40.0 % of 1456 window argmaxes (8 checkpoint classes) change between offline p0 and streaming p0 (2 s), 31.3 % between offline p0 and offline p1, 48.0 % between offline p0 and streaming p1; 0 % for the chunk-aligned mode (identical words). Class counts per condition are in `class_descriptive.json`. This shows the classifier output is sensitive to the ASR path; it says nothing about which one is right.

## 5. Change implemented (opt-in, nothing changes by default)
`StreamingASR.step_chunk_aligned` + `ChunkAlignedState` (`mrcd/live/streaming_asr.py`): decodes the same 8 s / 2 s chunk grid as `transcribe_chunked`, freezes complete chunks and re-decodes at most two partial chunks per tick; on the last tick the result equals the offline words exactly. Tests in `tests/test_asr_chunking.py`: equality with offline for several lengths and ticks (including a real adjacent repetition), a fake-ASR reproduction of the single-pass loop defect (the default `merge_recent` shows the loop, the new mode does not), and lost-chunk bookkeeping under consumer lag. The live driver still uses `merge_recent`; profiles are untouched.
Before/after on the same 67 units (p0): disagreement with offline 0.403 to 0.000, but WER vs human 0.396 to 0.552 (worse, because it inherits the offline loops); p1: 0.401 to 0.518. So the mode removes the streaming/offline inconsistency but does NOT improve accuracy; adopting it is a design decision (consistency with the engine path vs the loop-prone chunked decode), not a defect fix. Cost: RTF 1.17 vs 1.29 for p0 at a 2 s tick.

## 6. Not measurable / ready experiments
* Streaming p1 at a 0.5 s tick, and the full 135-unit streaming set: not run (cost). Command: `python scripts/cierre3b/asr_compare.py --out <dir> --conds str_p1_t0.5` (add `--subset-ids` / omit it for all units), then `asr_analyze.py`.
* WER of any condition on app audio (browser capture with AEC/NS/AGC), on non-Spanish or telephone audio, and a loop-safe decode profile: no reference/profile available here.
* AMERESCO conclusions rest on 4 recordings; DIHANA conclusions on read-style travel-dialogue turns (task-oriented, young adult speakers).
* Timing (RTF): offline p0 0.58, p1 0.70; streaming 2 s tick p0 1.29, p1 0.96; 0.5 s tick 3.3 (8 units). Machine idle (background CPU 6-11 % at the start of each run), but see `config.json`.
