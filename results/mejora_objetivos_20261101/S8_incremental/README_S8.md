# S8: B incremental (2026-11-01)

Self-review by the authors. **No independent review.** Advisor ratification of the privacy change and the ethics review of the new consent text are PENDING. Design: `docs/contracts/DESIGN_B_INCREMENTAL.md`.

## 1. Result

Every equivalence case that was executed passes (27 with a fake content-sensitive ASR through the real HTTP surface; the real-Whisper coverage with p0/p1 CPU int8 in the full service is in the table, with the gaps listed in section 9). The incremental path is therefore enabled by flag in the running stack, but **offline stays the default** in both repos. The acceptance criterion was reducing the wait without altering results: the post-close latency drops 3x to 7x and every document is identical to offline.

## 2. Equivalence cases (27) and privacy (6) / progress (5)

Fake = `tests/test_incremental_equivalence.py`, `tests/test_incremental_privacy_progress.py` (62 tests in the session). Real = `scripts/s8_incremental/equivalence_real.py` (44/44 pass, `equivalence_real.json`), `equivalence_real_long.json` (599.9 s), `equivalence_real_options.json` (head vc, text route), and `tests/test_incremental_real_whisper.py` (8 pass, MRCD_SLOW).

| # | Case | Fake | Real |
|---|---|---|---|
| 1 | 13.0 s | pass | pass (p0, p1) |
| 2 | 13.6 s | pass | pass (p0, p1) |
| 3 | 14.0 s (6+8) | pass | pass |
| 4 | 30 s, 48 s | pass | pass |
| 5 | 31 s, 32 s | pass | pass |
| 6 | 37.0 s, 36.999 s | pass | pass |
| 7 | 45.466 s | pass | pass |
| 8 | 60 s, 599.9 s | pass | pass (599.9 s p0 only) |
| 9 | > 10 min, engine level | pass (650 s) | NOT run with real Whisper beyond 599.9 s service level |
| 10 | 7.5 s, 8.0 s | pass | pass (slow test) |
| 11 | Partitions 160 ms, 2 s, random, one block | pass | pass |
| 12 | Out of order | pass | pass |
| 13 | Duplicates, duplicate close | pass | pass |
| 14 | Same seq, other bytes | pass | pass |
| 15 | Gap, resend | pass | pass |
| 16 | Retries, PCM equals buffer | pass | pass |
| 17 | Service restart | pass | pass + E2E fault injection |
| 18 | Cancel | pass | pass |
| 19 | p0 and p1 | pass | pass |
| 20 | noloop | pass | pass (p0_noloop; p1_noloop not run) |
| 21 | With and without warm-up | pass | pass (fresh process, preload only) |
| 22 | 3 sessions + a job | pass | pass |
| 23 | Options | pass | pass (include_out_of_scope, decision, all, head vc, text route) |
| 24 | Errors | pass | pass (audio_too_short); too long: `audio_too_long` at append (fake) |
| 25 | Shifted-plan mutant | pass (cache miss, same document) | not run |
| 26 | Merge mutant | pass (detected) | not run |
| 27 | Process thread state | pass | pass (single torch value 1; p1 after p0 and p0 after p1); load-order effect reproduced (4, 1, 4) |
| P1 | No files written | pass | |
| P2 | No content in logs | pass (both repos) | |
| P3 | Discard on completed, failed, cancelled, expired, shutdown | pass | |
| P4 | Events carry counts only | pass | |
| P5 | No consent, no session | pass (app) | |
| P6 | Idle expiry | pass | |
| M1 | Event order, single `completed` | pass | |
| M2 | Late subscription | pass | |
| M3 | `Last-Event-ID` resume | pass | |
| M4 | Frontend: no `refetchInterval` with the flag, polling intact without it | pass (vitest) | E2E: 2 detail reads after close |
| M5 | SSE cut: one detail read, reopen | pass (vitest) | |

Equal means `==` on the sorted-key JSON; only `generated_at` is excluded.

## 3. Benchmark (real service process, serial, p0, CPU int8, OMP=4)

`benchmark/BENCHMARK.md`, `benchmark/runs.json`. 1 discarded cold round + 2 measured; median (min-max). Background CPU before runs: median 6.7 %, max 24.5 % (docker, idle Postgres and app, the editor). Real-time pace means 2 s batches sent at their wall-clock moment.

| Recording (s) | Offline latency after close (s) | Incremental latency (s) | Offline CPU (s) | Incremental CPU (s) | Equal |
|---|---|---|---|---|---|
| 13.6 | 5.5 | 3.8 | 20.4 | 21.1 | yes |
| 30 | 9.7 | 2.7 | 35.8 | 37.9 | yes |
| 45.5 | 13.9 | 2.9 | 50.9 | 53.9 | yes |
| 60 | 19.9 | 3.6 | 72.6 | 77.9 | yes |
| 120 | 40.5 | 5.9 | 147.9 | 157.2 | yes |

- Total CPU grows about 3 to 7 % (the cache lookups, sha256 and the extra requests); peak working set of the MRCD process is unchanged (about 910 to 1020 MB, dominated by models). With paces of 0 (fast) there is no gain, as expected: latency 5.4 / 9.6 / 13.6 / 19.6 / 39.3 s equals offline.
- Startup: lazy 1.06 s to ready but 11.5 s for the first request (5.8 s the second); preload + warm-up 10.5 s to ready and 5.4 s for the first request. The benefit of preload is about 6 s on the first session.
- Not measured: the app backend's own memory peak (two copies), 2 to 3 simultaneous sessions (queue delay), the 600 s duration in the benchmark (599.9 s equivalence only).

## 4. Suites (literal summaries in `logs/`)

- Engine (OMP=4): `990 passed, 10 skipped` before the last tests; the final incremental suites alone: `62 passed, 7 skipped` (7 = MRCD_SLOW, run separately: `8 passed in 98.42s`). Total after the last commit was not re-run in full.
- App backend: ruff clean, mypy `Success: no issues found in 347 source files`, import-linter `18 kept, 0 broken`, pytest `1491 passed, 92 skipped` (same 92 Postgres-only skips as S7).
- App frontend: `tsc -b` clean, `eslint .` clean, vitest `68 files, 681 tests passed`.

## 5. Real E2E (docker compose + MRCD + Chromium with fake microphone and `e2e_synthetic_es.wav`)

| Run | Mode | Latency from "Cerrar sesion" | Detail reads after close | SSE opens | Outcome |
|---|---|---|---|---|---|
| `e2e_offline` | offline (default), under heavy background load | 32 s | 14 (polling) | 0 | completed, 9 events |
| `e2e_incremental` | incremental + push, same load | 8.7 s | 2 | 1 | completed |
| `e2e_incremental_verify` | verify | 35.9 s (runs both) | 2 | 1 | `verify = {equal: true, differing_keys: []}`; stored result is the offline one |
| `e2e_incremental_fault` | MRCD killed 15 s into the recording and restarted | 25.7 s | 2 | 1 | completed via fallback (`binding.mode offline`, `fallback_reason mrcd_unreachable`) |
| `e2e_final_incremental` | final code, idle machine | 5.3 s | 2 | 1 | completed |

Event counts differ between separate sessions (9, 10, 11) because each browser session captures slightly different audio; the equality claim rests on `incremental_verify` (same session, both ways) and the controlled equivalence tests, not on comparing different sessions.

## 6. Findings during the work

1. The OpenMP/torch thread count is per thread. An offline reference computed on the main thread differs in `provenance.environment.cpu_threads.torch` from one on the engine worker thread; the service runs both modes on the same worker, so the service is unaffected, but my first harness produced 8 false failures. Fixed in the harness (reference runs on the worker).
2. Real bug found and fixed (commit `fix(service): never expire a closing session...`): the idle sweeper expired a closing session whose analysis ran longer than 120 s (599.9 s case). Regression test added and verified to fail without the fix.
3. The venv `python.exe` is a launcher stub; the benchmark first measured the stub (0 CPU). Fixed to measure the listening process; first-pass log kept in `logs/`.
4. Design deviations: no `Last-Event-ID` between browser and app (state-first stream is enough); the browser event announces only the terminal status and the page reads the document through the authorised route.

## 7. Commits (local, not pushed)

Engine repo (`<repo>`, main): `fa76637` per-chunk cache and wrapper, `e9b746e` sessions, SSE and contract 1.3, `5713453` serve flags, `d553730` test suites, then the sweeper fix, the real-Whisper tests and this report (see `git log`). The docs (`PRIVACY_DECISION.md` section 8, `APP_INTEGRATION_CONTRACT.md` addendum, `DESIGN_B_INCREMENTAL.md` status) and the S8 scripts are in the final docs commit. `comandos-ejecutar/COMANDOS.MD` (outside any repo) has the new addendum.

App repo (`feat/mrcd-integration`): `a4e1aa0` client, `4b7e3b5` uploader, verify mode, hub and SSE endpoint, `298064a` consent text version, `349f690` report subscription, then the typing fix. `frontend/.gitignore` (foreign change) was never committed.

## 8. Activate, deactivate, revert

- Service: `serve --incremental --device cpu --preload p0 --warmup` (refuses to start without the three). Without `--incremental`, `/v1/sessions` answers `incremental_disabled` and the app falls back.
- App: `ORATORIA_MRCD_MODE=offline | incremental | incremental_verify` (default offline) and `ORATORIA_MRCD_PROGRESS_PUSH=true|false`; `docker compose -f docker-compose.local.yml up -d backend`.
- Revert: set `ORATORIA_MRCD_MODE=offline` and `ORATORIA_MRCD_PROGRESS_PUSH=false`; or `git revert` the commits above (all additive; offline code path untouched).

## 9. Pending

1. Independent review of both diffs (none done).
2. Advisor ratification of the two-copy, session-long exposure; ethics approval of consent text `2026-11-01-draft-2-incremental` (all accounts must consent again).
3. Cases 9 (real, > 10 min engine level), 20 (p1_noloop), 25 and 26 (real) not run; 2 to 3 simultaneous sessions and app backend memory not benchmarked.
4. Decide whether incremental becomes the default (criteria in the design, section 5; `incremental_verify` field evidence still to accumulate).
5. No TLS or authentication between app and MRCD; microphone test by a human still pending.

## 10. Addendum 2026-11-01 (corrections after the independent review `revision/REVISION_S8.md`; self-review, no independent re-review)

Test count correction: the incremental suites ran **63** tests (not 62) when the review was written; after this addendum they are **81 passed** (equivalence + privacy/progress + hardening, fake ASR) plus **10 passed** with real Whisper (`MRCD_SLOW=1`, 8 old + 2 new).

| Finding | Fix | Evidence |
|---|---|---|
| H1 (terminal SSE event carried the document) | Terminal `completed`/`failed` carries only state and counts (`status`, `result_available`, `session_id`); the app client reads the document with `GET /v1/sessions/{id}`. New per-session `subscription_token` (256 bit, returned once by `POST /v1/sessions`, required in `X-Session-Token` or Bearer on audio/close/state/events/delete; wrong or missing token = the same 404 as an unknown id; constant-time compare). P4 now covers the terminal event and scans the raw stream for transcript words. | `test_p4_*`, `test_m2_*`, `test_every_session_endpoint_requires_the_subscription_token`, `test_a_token_is_bound_*`, `test_delete_is_idempotent_*`, `test_the_failed_terminal_event_*`; app `test_client_sessions.py` |
| H8 (`*_noloop` equivalence not demonstrated) | Provoked REAL loops with Whisper (pieces of the repo clip `c60s_RkP96R.wav` tiled to 8 s; no third-party text). Result: `p0_noloop` early == offline bit for bit; **`p1_noloop` differed**: one chunk was replaced by the sampled rung 2 (T=0.4, `reproducible:false`) in the early decode and not in the offline one. So `*_noloop` profiles are now **refused** in incremental sessions (`422 profile_not_supported_incremental`, the app falls back to offline). The remaining claim is pinned by a slow test: any difference must be explained by a flagged sampled-rung replacement. Logic of the ladder also covered with a fake ASR (rung 1 identical, rung 2 flagged). | `test_real_whisper_loop_redecode_divergence_is_always_a_flagged_sampled_rung[p0/p1_noloop]`, `test_loop_redecode_rung1_*`, `test_profiles_with_the_sampled_loop_redecode_cannot_open_*`; `equivalence_real.py` case 20 now asserts the refusal |
| H2 (zeroing) | Zeroed now: session PCM, canonical WAV built at close (kept as a `bytearray`), request-body accumulator, per-chunk raw bytes and float32 copy. NOT zeroable, documented in `PRIVACY_DECISION.md` 8 and 8.1: immutable `bytes` handed over by the HTTP framework, `BytesIO`/float32 arrays inside the unchanged offline analysis, Whisper/CTranslate2 internals, allocator residue. | `test_controlled_transient_copies_are_overwritten_with_zeros` |
| H3 (no global byte cap) | `max_total_audio_bytes` (128 MiB default) over all live sessions (PCM growth + WAV reserved at close); exceeding it = `429 session_limit` (retryable), released on any end. | `test_global_audio_budget_*`, `test_budget_includes_the_canonical_wav_*` |
| H4 (cancel while closing) | The analysis checks the session state before each real chunk decode and stops there. Limit, documented: the decode in flight and the post-ASR stages cannot be interrupted; their result is discarded. | `test_cancel_during_closing_frees_the_worker_at_the_next_chunk_boundary` |
| H5 (priorities) | A close (priority 0) never waits behind queued chunks, only for the one running task (documented in `JobRunner`). A queued `POST /v1/jobs` (the app's offline fallback) is promoted to the front class after 20 s. | `test_a_close_never_waits_*`, `test_a_queued_job_is_promoted_*`, `test_session_tasks_use_the_documented_priorities` |
| H6 (frontend push) | Same 15 min cap in push mode shows the "stalled" notice; after 5 reconnections it makes one last detail read, stops reopening and shows the notice, with no reload. | vitest `use-session-report.test.tsx` (2 new) |
| H7 (idle stream) | A stream for a session with no analysis document and no progress closes with `unavailable: no_analysis` after 120 s; a pending analysis is never cut. | `test_a_stream_for_a_session_without_analysis_closes_by_timeout` |
| H9 (verify) | If the offline run fails (exception or failed document) after a good incremental result, that result is stored (`binding.mode = "incremental"`) with `binding.verify = {equal: null, differing_keys: [], offline_status, offline_code}`; equality is never claimed. | two new tests in `test_incremental.py` |
| M3 (cache key without profile) | Namespace is `"<asr_profile>\|<asr config key>"`. | `test_cache_namespace_includes_the_requested_profile_name` |
| A5 (progress not sanitised) | Test that hostile progress data (text, words, result, bool counters, long stage) is reduced to counters and a 40-char stage. | `test_forwarded_progress_is_sanitised_*` |
| H11 (hangs without diagnosis) | `pytest-timeout` installed in the venv and listed in the `dev` extra; the incremental suites carry `pytest.mark.timeout` (120/180 s fast, 1800 s slow), so a stuck test fails with a stack dump. | e.g. mutant F8 now ends in `Timeout` with dumps instead of hanging |
| README count | 62 -> 63 at review time, 81 now (see above). | |

Mutants checked in throw-away clones outside both repos (`%TEMP%\s8fix_mut_engine`, `%TEMP%\s8fix_mut_app`; the clones could not be deleted by the tooling and remain in `%TEMP%`): engine F1 (terminal event with document), F2/F2b (token unchecked), F3a-c (body / chunk / WAV not zeroed), F4/F4b (budget not enforced / not released), F5 (no safe point), F6a/F6b (finalize low priority / jobs never promoted), F7 (namespace without profile), F8 (cache key without content hash), F9 (noloop accepted): **all killed** (F6a and F8 first survived; the two tests `test_session_tasks_use_the_documented_priorities` and `test_the_cache_never_answers_for_different_audio_*` were added and they now die). App A5, H9, H9b, H7, H7b, H1c-e, H6, H6b, H6c: **all killed**.

Re-run (final code): engine incremental suites `81 passed, 10 skipped` (the 10 = MRCD_SLOW) and, with `MRCD_SLOW=1 OMP_NUM_THREADS=4`, `test_incremental_real_whisper.py` `10 passed in 389.89s`; app backend full suite `1497 passed, 92 skipped`; app frontend `68 files, 683 tests passed`, `tsc --noEmit` and the touched ESLint scope clean, `ruff` and `mypy` clean for `platform/mrcd`. Short real E2E (separate ports: MRCD 8766, compose project `s8fix` on 5273/8100/5444, mode incremental + push, fresh DB, then torn down with `down -v`): `e2e_review_fix_incremental/` = completed, `binding.mode incremental`, no fallback, 6.3 s after "Cerrar sesion", 2 detail reads, 1 SSE open, 10 events (machine shared with a long campaign, so the time is not evidence).

Files added/changed for this addendum: engine commits `fix(service): per-session token, ...` and `test(s8): token-aware scripts, ...`, plus the docs commit; app commits `fix(backend): send the MRCD session token ...`, `fix(backend): keep the incremental result when verify's offline run fails ...`, `fix(report): stalled notice ...`. Not done: `equivalence_real.py` (44 cases, 20 min) was not re-executed in full after the token change (its harness was adapted and the noloop case rewritten); the 10 slow pytest cases were. Offline remains the default in both repos.

## Addendum 2026-11-02: full real equivalence re-run after the token change

`scripts/s8_incremental/equivalence_real.py` was executed again in full on the final code: `equivalence_real_rerun_after_token.json` and `equivalence_real_rerun_after_token.log` record 45/45 passed (the 44 cases of `equivalence_real.json` plus the single-torch-thread check). This supersedes the sentence above saying the 44 cases were not re-executed after the token change; the 10 slow pytest cases and the other results are unchanged. Offline remains the default.

### 10.1 Follow-up (revision 2 residuals and full re-run)

- Full real-Whisper battery re-run after the token change, in-process (no port), p0 and p1, CPU, same worker thread: **45/45 passed** (`equivalence_real_rerun_after_token.log`, sha256 `332094d0ff0378fae768d9cb5d769ad7537c6409fb8b558ab9dce209d3c00307`; `equivalence_real_rerun_after_token.json`, sha256 `73a65cde99361c0f6e28320d9cc12dddbd1731c5bf5aa949bb8e98d4e393f209`).
- R1: `DELETE` answers the same idempotent 204 for an unknown id and a wrong/missing token and acts only with the token. R2: the WAV reservation is made after the sha check (a refused close leaves `bytes_in_use` unchanged; test `test_a_refused_close_leaves_no_byte_reservation`). R3: app tests for token dropped after the result and for `profile_not_supported_incremental` being fatal. Mutants E19, A6, A7 now die.
- Re-run after R1/R2: incremental suites `82 passed`; slow real-Whisper `10 passed in 365.92s`.
