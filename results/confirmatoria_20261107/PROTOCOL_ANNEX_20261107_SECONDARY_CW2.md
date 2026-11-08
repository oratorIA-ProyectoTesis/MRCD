# Annex 2026-11-07 to PROTOCOL_CONFIRMATORIO.md: SECONDARY arm cw2 (CrisperWhisper 2.0 + lexical rule)

**Status.** Written and frozen BEFORE any MRCD output exists on NOCANDO and before anybody opens NOCANDO audio, transcripts or
labels for this arm. Its sha256 and that of the files it adds are in `PROTOCOL_ANNEX_SECONDARY_CW2.sha256.json`; the secondary
runner refuses to run if any of them differs. This annex **adds** a secondary arm. It changes nothing in
`PROTOCOL_CONFIRMATORIO.md`, in the primary harness (`scripts/confirmatoria/run_confirmatoria.py`, unchanged byte for byte,
hash in `PROTOCOL.sha256.json`), in the primary systems (default, v0, rule, always_present), in the cells, in the Holm family
of size 4 (H1-H4), in the absolute criteria A1/A2, or in the failure rules. Authored by a software agent; self-review only; no
independent reviewer and no advisor ratification. The arm described here has **not been run**.

## 1. Purpose and origin

The exploratory benchmark `results/benchmark_empirico_20261107` (DIHANA folds 1-4 and CHM150, development data; fold 5 never
read; protocol frozen before any competitor ran; the author of MRCD is the evaluator) found that CrisperWhisper 2.0 large + the
same one-line lexical rule beat the MRCD service default on all three of its tables: DIHANA repetition F1 0.815 versus 0.516,
CHM150 repetition F1 0.818 versus 0.331, DIHANA filled-pause F1 0.894 versus 0.743 (`README_RESULTADOS.md`, `metrics.json`;
those were different frames, units and a different reference than this run). The arm was therefore **chosen on development
data** and is declared here, before the confirmatory run, so that the confirmatory cell can test whether the observation holds
on data that played no part in choosing it. It is a **secondary** arm: it is reported separately and **never** replaces or
alters the primary result or its Holm family. The decision of the user was to keep the primary system F unchanged.

## 2. The exact secondary system (frozen)

* **Name.** `cw2_rule`.
* **ASR.** `mrcd.asr_profiles` profile `cw2` (backend `crisperwhisper`, no prompt, Spanish, verbatim mode, word timestamps).
  Filled-pause tags (`[UH]`, ...) become the lexicon word and any other tag is dropped, exactly as declared in
  `mrcd/verbatim.py` (`FP_TAG_NORM`) at the system commit; the long-form chunking is the package's own.
* **Model.** `nyralabs/CrisperWhisper2.0_large`, Hugging Face revision `f4334f6e8193f2691212d49b20fa12d370e13896` (the value of
  `CW2_MODEL_REVISION` in `mrcd/asr_profiles.py`; equal to `refs/main` of the local cache on 2026-11-07).
* **Weights hash.** `model.safetensors` sha256 `f097f853b6992e66fcae802a15770a8242670351198d68dad577db3209fe3fcc`. Hash of the
  whole local snapshot (14 files, symlinks resolved; sha256 over the sorted lines `<file name>\t<file sha256>\n`):
  `b22cf7a5208e73405514f32b9e7686019364771d51a8a8683f105596d88e7ac9`. Per-file hashes were computed locally without running any
  corpus (config.json `efd226b2...`, tokenizer.json `567da8c8...`, vocab.json `8f680bba...`; the runner recomputes all 14 at run
  time and treats a mismatch as "not available", section 7). The local files were read from
  `~/.cache/huggingface/hub/models--nyralabs--CrisperWhisper2.0_large/snapshots/f4334f6e...` with `local_files_only=True`.
* **Device.** CUDA, float16 (`mrcd.verbatim.resolve_device`; no CPU fallback in this arm: a CPU float32 decode would be a
  different system). The GPU name, torch version, CUDA runtime and driver-visible device are recorded in the result at run time
  (`environment.gpu_name`). The development machine has an `NVIDIA GeForce RTX 5060 Laptop GPU`, 8 GB; the run records whatever
  GPU is actually used.
* **Decision rule on the words.** The **same functions** as the primary `rule` (`run_confirmatoria.rule_events`): repetition =
  immediate exact repetition of 1-4 tokens after removing non-lexical filled pauses (event = the second occurrence); filler =
  tokens matching the non-lexical filled-pause pattern (bare `e` excluded). It is applied to the cw2 words instead of the p0
  words. The MRCD classifier output produced while the engine runs is **discarded** (a valid decision profile, `v0`, is passed
  only because the analyser requires one); nothing of MRCD's decision enters this arm.
* **Audio, units, references, mapping.** Identical to the primary: same 14 recordings, same TextGrid units, same
  transcript-derived references (`rep_literal`, `fill_ref`), same whole-recording mp3 -> 16 kHz mono decode, same max-overlap
  event-to-unit mapping (20 ms minimum), same unit-level prediction rule.
* **Statistics.** Same as the primary and the same code: cluster = recording (14), bootstrap seed 13, 2 000 resamples for the
  95 % percentile intervals and 5 000 for the tests, `p two-sided = min(1, 2*min(P(d<=0), P(d>=0)))` with +1 correction,
  alpha 0.05. (The task note "speaker bootstrap" corresponds to the primary protocol's declared recording-level clustering,
  section 4 there; NOCANDO has 5 speakers, so a speaker bootstrap would be degenerate. This arm does not change that.)
  The evaluability gate of section 6 of the protocol applies unchanged and per class.

## 3. Secondary family (fixed size 2) and descriptive row

Both contrasts are unit-paired cluster bootstraps, **two-sided**, against the **primary `default`** predictions read from the
primary unit file (ids and 0/1 only):

| id | Class | Contrast |
|---|---|---|
| S1 | repetition | F1(cw2_rule) - F1(primary default), over all units |
| S2 | filler_word | recall(cw2_rule) - recall(primary default), over reference-positive units |

* **Holm within {S1, S2} only**, family size 2 fixed (a test that cannot be computed still counts), alpha 0.05, and the same
  support rule as the primary: a contrast is **supported** iff `p_Holm < 0.05` and the 95 % interval of the paired difference
  excludes 0; the direction is reported (`cw2_better` / `cw2_worse`). Otherwise **not demonstrated**.
* S1 and S2 are **not** members of the primary family H1-H4 and do not enter its adjustment; the primary p-values are not
  recomputed, re-adjusted or re-read after the secondary result.
* **Descriptive-only row** (no p-value, no verdict, outside any family): cw2_rule versus the **p0 rule** (the same rule on the
  p0 ASR words), difference and interval for repetition F1 and filler recall. It says how much of any gain comes from the ASR
  with the rule held fixed.
* Also reported (descriptive): per-system tp/fp/fn, precision, recall, F1 with intervals, dropped events without unit.

## 4. Written-before expectation (so it can be falsified)

On the development data cw2_rule beat the default by large margins, so the expectation is S1 and S2 both positive. It may fail
here: the reference is the lexical rule applied to a transcription, which favours any system that writes a verbatim text and
applies that rule (circularity, protocol section 7), and the NOCANDO units come from 2-6 minute narrations decoded by the
package's long-form strategy, unlike the 1-30 s turns of the development data. A negative or null result is information, not an
error. Nothing here is a claim about human-judged disfluency or about generalisation beyond 5 speakers.

## 5. What this arm does not do

No tuning, selection or thresholding on NOCANDO. No second ASR profile, no model variant (`CrisperWhisper2.0_turbo`, v1,
OpenOPAF and commercial ASR are not part of this arm), no prompt, no post-hoc change of the rule, the tag mapping, the device or
the precision. No new human annotation (decision D-1). No second look on AMERESCO for this arm. No prolongation (no system of
this arm produces it).

## 6. Order of execution and independence from the primary

1. Freeze the system as the protocol says (`freeze-system`), then run `verify`, then the **primary** `run` (section 9 of the
   protocol), all at the same `--system-commit`.
2. Compute and **write the primary conclusions** (verdicts of H1-H4, A1, A2, failure criteria) in
   `results/confirmatoria_20261107/PRIMARY_CONCLUSIONS.md` and **commit** it. The secondary runner refuses to start unless this
   file exists and is committed, and unless `CONFIRMATORY_RESULT.json` exists, is valid and matches the hash in a `completed`
   `RUN.lock.json`. The primary conclusions cannot be changed by the secondary result.
3. Only then, under the **same frozen system commit** (the runner calls the primary `verify_all`: protocol hash, primary harness
   hash, system tree hash, data hashes, ancestry and committed state) and the annex hash check:

```powershell
cd C:\zArnes-tesis\mrcd
$env:OMP_NUM_THREADS = "4"; $env:MRCD_CORPORA_ROOT = "<root of the corpus repository>"
.\.venv\Scripts\python.exe scripts\confirmatoria\run_secondary_cw2.py verify --system-commit <SYSTEM_SHA>
.\.venv\Scripts\python.exe scripts\confirmatoria\run_secondary_cw2.py run --system-commit <SYSTEM_SHA>
```

   The secondary `run` is a separate process with its own lock `RUN_SECONDARY_CW2.lock.json` (O_EXCL): it runs **once**.
   The environment needs the optional `crisperwhisper` package and a CUDA build of torch (see `THIRD_PARTY_NOTICES.md`); the
   current `.venv` has neither, so this may need a prepared environment. Preparing it is allowed only without changing `mrcd/`,
   `models/` or `pyproject.toml` (the system tree hash is checked). The same Python environment rule applies: unloaded machine.

**Why a new file instead of editing the primary harness.** `PROTOCOL.sha256.json` freezes the hash of
`run_confirmatoria.py`; editing it would make the primary `run` refuse. The secondary arm therefore lives in
`scripts/confirmatoria/run_secondary_cw2.py`, which imports the primary harness unchanged and reuses its reference, statistics,
rule, mapping and verification functions. `tests/test_confirmatoria_secondary_cw2.py` asserts that the primary harness still
matches its frozen hash. This annex's manifest freezes the two files added.

## 7. Failure handling

* If CUDA is unavailable, `torch` or the `crisperwhisper` package cannot be imported, the pinned snapshot is missing, the local
  weights hash differs from section 2, the model fails to load, any recording raises or returns an analysis error, or the
  process crashes (e.g. out of memory on a 2-6 minute narration), the arm is recorded as
  `{"status": "not_available", "reason": "<the cause>"}` in the secondary result and **no metric is produced**. It is never
  reported as 0 (rule FR-025 of the thesis) and never as a partial result over the recordings that did run.
* **No retry** with changed settings (no CPU fallback, no other precision, no other batch or chunk size, no other revision).
  The lock makes a second attempt impossible; a hardware or environment fix could only be made through a new dated annex that
  states the cause and changes nothing else, and only if no secondary result with metrics was ever produced.
* A failure of the secondary arm has no effect on the primary result, its conclusions or its Holm family.

## 8. Outputs

`CONFIRMATORY_RESULT_SECONDARY_CW2.json` (aggregates, counts, hashes, device and GPU name, system commit, annex hash,
environment; **no corpus text, no transcript, no per-unit file**) and `RUN_SECONDARY_CW2.lock.json`. Nothing is written to the
primary outputs. Only aggregates may be versioned or shared.

## 9. Licence

CrisperWhisper 2.0 code is MIT, but the **weights and the outputs** are under the Nyra Health Non-Commercial Research License:
non-commercial research use only, not usable in a commercial product, and its transcripts must not be published (see
`CW2_LICENSE_NOTE` in `mrcd/asr_profiles.py`). This arm is a research evaluation for the thesis; the arm and its numbers are not
a statement about any commercial use. NOCANDO is CC BY-NC-SA 3.0 (Brunetti, Bott, Costa, Vallduvi, 2011).

## 10. Development checks already done (declared)

* `tests/test_confirmatoria_secondary_cw2.py` (synthetic fixtures only, stub analyser, no GPU, no model, no corpus): refusal
  before the primary is complete, refusal on an annex hash mismatch, the `not_available` path (preflight, crash, recording
  error), Holm family = {S1, S2} only, an underpowered class, and byte-identity of every primary file with and without the
  secondary arm; plus a check that the primary harness still equals its frozen hash.
* The pinned weights were hashed locally (section 2). No NOCANDO audio, transcript, label or corpus file was opened, listed,
  decoded or indexed for this annex, and neither `run_confirmatoria.py run` nor `freeze-system` was executed.
* The `.venv` has no `crisperwhisper` package and a CPU build of torch: the arm itself was not exercised even on synthetic audio.

## 11. Language and attribution

Outputs and comments in English. NOCANDO: CC BY-NC-SA 3.0. CrisperWhisper 2.0: Nyra Health Non-Commercial Research License.
