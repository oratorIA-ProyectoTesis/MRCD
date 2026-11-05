# ASR-survival benchmark (Spanish disfluencies, DIHANA frame)

A small, reusable evaluation frame: **how much of the human-annotated disfluency evidence of a turn is still
present in the transcript an ASR system produces?** Spanish version: [LEEME.md](LEEME.md).

Status: exploratory, not a calibrated or validated measure of ASR quality. Survival is a transcript heuristic, not a
human judgement (the manual-validation kit has not been run). One corpus, telephone speech, peninsular Spanish.

## What is measured

For each frame turn, "survival" is decided by `mrcd.context_eval.survival.turn_survives` (the exact code behind
paper section 4.8 and `results/metrics.json`):

| Metric | Population | A turn counts as surviving when |
|---|---|---|
| `survival_rep` | turns with a literal repetition (A-literal subtype), n = 214 | the annotated span is recovered in order with >= 80 % of its tokens **and** an adjacent duplicate remains inside the aligned ASR region |
| `survival_rev` | turns with a revision (substitution/insertion/deletion), n = 161 | reparandum, correction and span are each recovered at >= 80 % |
| `survival_filler` | turns with an annotated non-lexical filled pause, n = 308 | the ASR text contains a non-lexical filler token (`eh`, `mm`, `em`, `ah`...; `este` is never a filler) |

Each is a ratio of sums with a **speaker-clustered bootstrap** (122 speakers, seed 20261101, 5000 resamples, 95 %
percentile CI). The interval is read with fixed tiers: upper < 0.50 "majority erased"; upper < 0.75 "erasure
persists"; lower >= 0.75 "substantially attenuated"; otherwise "inconclusive". Absence is never reported as zero:
an empty population is `unavailable`.

## The frame

`frame/turn_ids.csv`: 1,025 DIHANA user turns, folds 1-4 (214 literal-repetition, 161 revision, 308 filler-positive;
679 sampled negatives, 122 speakers). Columns:

| Column | Meaning |
|---|---|
| `turn_id` | DIHANA turn identifier (`<dialogue>_<turn>`), also the name of your ASR output file |
| `speaker_id` | DIHANA speaker code; the bootstrap cluster |
| `fold` | speaker-disjoint fold (1-4) |
| `kind` | `pos` (selected for a repetition/revision) or `neg` (sampled negative) |
| `sampling_weight` | inverse sampling weight of the frame design |
| `rep_literal` | 1/0, or empty when the repetition sub-type is indeterminate |
| `revision` | 1/0 |
| `filler_present` | 1 if the turn has an annotated non-lexical filled pause |

There is no transcript text and no audio. The reference marks are rebuilt from your own DIHANA copy, and
`survival_bench.py` aborts if any rebuilt label disagrees with the CSV.

## Get DIHANA

DIHANA (Spanish task-oriented dialogues), Zenodo record 5572470, DOI [10.5281/zenodo.5572470](https://doi.org/10.5281/zenodo.5572470) (Benedí, Lleida, Varona et al.; verified 2026-11-04 via the Zenodo API),
licence **CC BY 4.0** (attribution required; cite the corpus authors as named on the record). The record is
listed in [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md). Extract it so that `data/*/*/*.xml` exists under
the root you pass as `--dihana-root`. Verify licence and DOI on the record page before redistributing anything.

## Run

```bash
pip install -e ".[train]" faster-whisper soundfile     # from the repository root, Python >= 3.11
cd benchmarks/asr_survival/scripts

# 1) Decode (any ASR works; this is the reference decoder). Audio: 16 kHz mono WAV named <turn_id>.wav, supplied by you.
python decode_faster_whisper.py --wav-dir WAVS --model large-v3 --variant P0 --out-dir out/lv3_P0 \
       --device cuda --compute-type float16        # variants: P0, P1, C1, p_none

# 2) Score
python survival_bench.py --dihana-root /path/to/dihana --asr-dir out/lv3_P0 --out out/lv3_P0.json
```

Use another ASR system by writing one `<turn_id>.json` per frame turn: either `{"words": [{"text": "..."}, ...]}`
(`--kind words`) or `{"text": "..."}` with CrisperWhisper-style bracket tags (`--kind cw`; tags are stripped and
filled-pause tags counted as fillers, PROTOCOL 4.3). Output: `survival_rep`, `survival_rev`, `survival_filler`, each
with `k`, `n`, `value`, `ci`, `reading`.

Prompts (exact strings, `P0`, `P1`, `C1`; `p_none` = no prompt): [prompts/](prompts/).

## Expected outputs and reference results

`results/metrics.json` holds aggregates only (no text, no per-turn data) for the systems of the frozen protocol:
Whisper small (CPU int8) and large-v3 (GPU float16) under P0/P1/C1/no-prompt, plus a verbatim-tag model. For
example `small_P0` gives survival_rep 105/214 = 0.491 (CI 0.426-0.550), survival_rev 53/161 = 0.329,
survival_filler 158/308 = 0.513. `survival_bench.py` reproduces these three values from the cached small-P0
transcripts (verified before release). Decoding on other hardware is not bit-identical to the reported runs.

`results/PROTOCOL.md` is the frozen protocol; its sha256 (`b83fb8fc...357c`) is recorded in
`results/PROTOCOL.sha256.json` and inside `metrics.json`, and checked by the tests. It is published byte for byte,
so it still names the authors' original local environment paths; they are historical and not needed to run
anything here.

Not included: the full paired-contrast/Holm analysis script of the paper (its numbers are in `metrics.json`), the
transcripts of any system, and the CrisperWhisper decoder and outputs (its weights and outputs are under a
non-commercial research licence; run it yourself under those terms).

## Cite

Cite the software (see `CITATION.cff` at the repository root, version 0.3.0) **and** DIHANA. Once the Zenodo archive
exists, cite its DOI instead.

## Licence

Code: MIT (repository `LICENSE`). `frame/turn_ids.csv` is derived from DIHANA (CC BY 4.0) annotations, so using it
requires attributing DIHANA. `results/` contains aggregates computed from model outputs; no third-party text.

## Archive on Zenodo (maintainers; three steps)

The authors must enable the integration themselves; nothing here does it.

1. Sign in at <https://zenodo.org> with GitHub, open *Account > GitHub*, and switch the repository
   `oratorIA-ProyectoTesis/MRCD` on (an organization owner may need to approve Zenodo's access).
2. Check `CITATION.cff` and `.zenodo.json` (title, authors, version, licence), then create a GitHub **release** with
   a tag (for example `v0.3.0`); only releases are archived.
3. Zenodo archives the release and mints a DOI. Add the DOI to `CITATION.cff` (`doi:`) and to the README badge, and
   use the "all versions" concept DOI in papers.
