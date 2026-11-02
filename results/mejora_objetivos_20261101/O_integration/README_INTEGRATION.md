# O integration: optimised pipelines as the service default (2026-11-02)

**Status: implemented and tested; the underlying result stays EXPLORATORY.** Self-review by the authors only; nothing here is independently confirmed. The decision to make this the product default was taken by the orchestrator with the user's authorisation to close the product.

## 1. What changed

| Area | Change |
|---|---|
| Service | New versioned **decision profile**. `o2026_11` is the **service default**; `v0` is the explicit historical profile. `mrcd/service/decision_profile.py`. |
| `filler_word`, `prolongation` | Come from the O-campaign pipelines. Campaign F1 (filler 0.697 vs V0 0.567, lexical rule 0.661; prolongation 0.619 vs V0 0.458) was measured on **DIHANA turns and does NOT apply to the service**, which uses pause-delimited segments (>= 0.6 s): unmeasured domain shift; do not quote in product material. |
| `repetition` and every other class | **Unchanged** (engine decision). The campaign pipeline does not beat the one-line rule (-0.002) or S3 `rep_ctx_p1`, and `rep_ctx_p1` is an opt-in `text_route`, not the product path, so there was nothing to keep or replace. The repetition pickle (HGB) is not used by the product. |
| Library | `DisfluencyEngine.analyze` is untouched and always returns V0 events (history and experiments stay reproducible). Only `DisfluencyEngine.unit_features(clip, words)` was added (opt-in, additive). |
| Artifacts | Product reads **plain JSON** (`mrcd/service/profiles/o2026_11/{filler_word,prolongation}.json`), exported from the campaign pickles by `scripts/o_integration/export_json_pipelines.py` (pickle read only there, after the campaign manifest sha256 matched). Both exported pipelines are logistic regressions, so the export is exact: mean/sd/coef/intercept/thresholds. `mrcd/optimize/safe_pipeline.py`. |
| Integrity | sha256 of each JSON is **pinned in source** (`PINNED_SHA256`), compared before parsing, parsed from the same bytes. A changed or missing file raises `ProfileArtifactError`; `preload` fails at start-up; a request gives `status: failed`, `analysis_failed`, message pointing to `decision_profile=v0`. No silent fallback. `.gitattributes` keeps these files byte-exact (`-text`). |
| Contract | **Revision 1.4**, additive: `request.decision_profile`, `provenance.decision_profile` (`name`, `status`, per-class `route`, `artifact_sha256`, `source_pickle_sha256`, `pipelines_verified`, `calibration_claim: none`), score kind `unit_pipeline_model_score`. Documents of revisions 1.0-1.3 still validate (test). Docs: `docs/contracts/APP_INTEGRATION_CONTRACT.md` B.3c; fixtures `docs/contracts/examples/service_analysis_v1_4_{default_profile,v0}.json`. |
| Query / CLI | `POST /v1/jobs?decision_profile=o2026_11\|v0` (default o2026_11; unknown value -> 400 `invalid_request`); `python -m mrcd.service analyze --decision-profile v0`. Sessions (`/v1/sessions`) take the same query. |
| Architecture docs | C4 level 3 gets the `decision_profile` component; `verify_architecture.py` rule "optimize is imported by nothing" became "only `service/decision_profile.py`, through `safe_pipeline`"; two stale app checks updated (see section 6). |

## 2. How units are built (the main domain shift)

The pipelines were trained on **DIHANA turns** (one clip per turn). A service recording is a monologue, so the profile splits the ASR words into **segments at pauses >= 0.6 s (max 30 words)** (same convention as the S3 text route), cuts each segment as a standalone clip (+0.15 s margin each side), re-times its words, and calls `unit_features`: zero-pad to one 3 s window if shorter, window probabilities and pooled vector of the base checkpoint, pause statistics. Fidelity check against the stored campaign features on 12 random DIHANA units (folds 1-4): max abs diff probs 4.8e-7, pooled 9.5e-7, pause 0.0; decisions identical 12/12 for both classes (`scripts/o_integration/verify_features_from_audio.py`).

Consequences, all unevaluated outside DIHANA: events are **segment-level** (seen in the fixture: 5-11 s intervals, coarser than V0's 3 s windows); a segment can fire both classes; turn length distribution differs from pause-split segments.

## 3. Activate / revert

- Default (nothing to do): `o2026_11`. The user's running service (127.0.0.1:8765) must be **restarted** to load it; I did not touch it.
- Per request: `?decision_profile=v0`. Per CLI call: `--decision-profile v0`.
- Service-wide revert: set `DEFAULT_DECISION_PROFILE = "v0"` in `mrcd/service/decision_profile.py` (one constant; the server, CLI and `AnalysisOptions` all read it).
- Library / experiments: nothing to do, they never used the profile.
- The app sends no `decision_profile`, so it follows the service default; its `output_scope` stays `evaluated`.

## 4. Tests and evidence

- Full suite, literal command `OMP_NUM_THREADS=4 .venv\Scripts\python.exe -m pytest -q -rs`: **1065 passed, 20 skipped** (`pytest_full.log`; skips are the pre-existing `MRCD_SLOW` ones). New: `tests/test_decision_profile.py` (15 tests).
- **Regression on saved units:** 40 DIHANA units (folds 1-4, seed 13, golden `tests/data/o2026_11_regression.json` computed from the campaign pickles): the JSON pipelines reproduce decisions and three-state exactly, scores within 1e-9; the JSON export equals the pickle pipelines (decisions, states, scores 1e-10).
- **Tamper:** altered file (valid JSON, different threshold) refused; missing file refused; failed document (not a fallback) when it happens at request time; product modules contain no pickle import.
- **v0 bit-for-bit:** against a worktree of the previous commit (`da3bdaa`), same clip (20 s excerpt), same flags: the documents are identical after removing `generated_at` and the additive rev 1.4 fields, 3 of 3 pairs (with and without `--decision --include-text`). Note: ASR int8 output on this machine alternates between two numeric attractors under CPU load (old code too: 0.822061 vs 0.826882); pairs were compared within the same attractor. Real-engine test `test_real_engine_service_events_equal_the_direct_engine_path` (MRCD_SLOW) now proves v0 == direct engine events and that the default profile leaves repetition untouched (that test was already failing before this work, comparing out-of-scope classes the service hides since rev 1.2; fixed to the in-scope classes).
- **Incremental == offline with the new profile (real Whisper, full HTTP stack, `scripts.s8_incremental.equivalence_real --quick`):** **44/44 passed** (`equivalence_o2026_11.json`, `equivalence_run.log`): 22 cases x {p0, p1_disfluent_prompt}, `include_text=true&decision=true`, durations 13-60 s, partitions, out-of-order, duplicates, gap/resend, cold vs warm. Documents equal `==` excluding `generated_at`. The ASR cache covers only ASR; unit features are recomputed from the audio at close, so the profile is incremental-safe. The 599.9 s case was not run (`--quick`).
- Manual check on test port 8768 (stopped afterwards): default and `v0` documents valid; default shows `unit_pipeline_model_score` events plus engine repetition; wall time 8.0 s for both on the 20 s excerpt (no measurable overhead there; 16 s on the 45 s synthetic clip).

## 5. Limits (read before quoting any number)

1. **Exploratory.** DIHANA folds 1-4, nested CV by speaker. Every warning/provenance block says so (`decision_profile_exploratory`, `status: exploratory`).
2. **Unstable selection:** every fold chose a different configuration in filler (modal config 1/5); only the model family is stable. The packaged configuration is the one refit on all four folds.
3. **V0 is a weak baseline.** Against the best fixed comparators the gain is modest: filler +0.036 vs the lexical rule (0.661), prolongation +0.027 vs the S3 stack (0.592); the +0.13/+0.16 headline is against V0 only.
4. **No external gain demonstrated:** EPIC filler delta +0.006 [-0.050, 0.068]; AMERESCO prolongation recall 0.95 with precision 0.37, CI uninformative. Plateau: more data of the same kind is unlikely to help.
5. **Repetition has no improvement** over the one-line rule on P1 text (-0.002) nor over S3 `rep_ctx_p1`; it keeps the engine decision.
6. **Scores are not calibrated probabilities** (`calibration_claim: none`, `score.calibrated: false`; the campaign chose no calibrator for these two classes, so the three-state `estimate` is the raw score). The band `[lo, hi)` is from the inner CV.
7. Turn -> pause-segment shift (section 2), coarse intervals, and pipelines trained on **P0 text** only (warning `decision_profile_asr_mismatch` for other ASR profiles).
8. Prolongation means observable lengthening; its disfluent function is not demonstrated (existing rule, unchanged).
9. Latency/cost of unit features was measured on two clips only.

## 6. Architecture verification (`docs/architecture/verify_architecture.py`: 206 checks, 0 failed)

The two failures caused by this work are fixed by design (the rule changed because the service now adopts `optimize`). The other two came from the app side and were stale, not regressions: the app repo (committed `960c351 chore(config): local free adapters and incremental MRCD as defaults`) now defaults `mrcd_mode` to `incremental`, and the events section decides scope from the service's `in_scope` flag (the app requests `output_scope: "evaluated"` explicitly), not from a hard-coded class list. Both behaviours are consistent with the service, so only the checks and the two diagram sentences were updated; the app was not touched.

## 7. Files

`mrcd/optimize/safe_pipeline.py`, `mrcd/service/decision_profile.py`, `mrcd/service/profiles/o2026_11/*.json`, `mrcd/engine.py` (`unit_features`), `mrcd/service/{analyzer,contract,server,cli}.py`, `mrcd/service/schemas/mrcd.analysis.v1.schema.json`, `pyproject.toml` (package data), `.gitattributes`, `scripts/o_integration/*`, `tests/test_decision_profile.py`, `tests/data/o2026_11_regression.json`, `docs/contracts/*`, `docs/architecture/*`.
