# S7 - App integration (privacy option B) and real end-to-end run (2026-11-01)

**Self-review by the authors; no independent review yet.** Exploratory integration evidence, not a validation of the engine. Defaults of the engine are unchanged (V0, p0 CPU int8, 3-class scope). No change was made to the MRCD engine code in this task.

## 1. What was done

- Applied the S3 patch (rev 1.2, `app_patch_package/0001-feat-report-mrcd-disfluency-section.patch`) to the app on a NEW local branch `feat/mrcd-integration` created from `refactor/f0-guardrails` @ `f46a2b8` with `git switch -c` (no stash; the user's own change to `frontend/.gitignore` stayed unstaged and is in no commit). `git apply --3way` applied it clean (15 files).
- Privacy option B in the app backend: `/ws/session` keeps the session PCM16 16 kHz mono only in memory (`backend/src/oratoria/platform/mrcd/audio_buffer.py`, cap `ORATORIA_MRCD_MAX_AUDIO_SECONDS`, default 600 s, clear `error` message `limite_audio_analisis_excedido` when passed), hands it to the MRCD service at close (`client.py`, `analysis.py`; URL `ORATORIA_MRCD_SERVICE_URL`, empty = feature off and nothing buffered), stores ONLY the JSON linked to the session (`practice_session.mrcd_analysis` JSONB, Alembic `a7c3e5f9b1d2`, returned as `mrcd_analysis` by `GET /api/v1/sessions/{id}`) and erases the buffer explicitly on success, MRCD failure, unexpected error, limit exceeded, abrupt disconnection, store failure and shutdown. Result document: ASR profile, scope, `calibrated:false`, provenance, binding with `audio_sha256`, or `unavailable {code, reason}` (never a 0).
- Frontend: `ReportPage` renders `SessionDisfluencySection` from the session report (default classes `filler_word`/`prolongation`/`repetition`, timestamps, "Score (softmax, no calibrado)", provenance, warnings; states pending/unavailable/not requested; polls while `pending`). The patch's manual WAV upload is now a development tool behind `VITE_MRCD_DEV_UPLOAD=true` (off by default).
- Contract docs in the engine repo: `docs/contracts/PRIVACY_DECISION.md` section 6 (decision of 2026-11-01, option B, advisor ratification PENDING) and `APP_INTEGRATION_CONTRACT.md` B.12 (implemented flow).
- Finding about the patch: it broke an existing app invariant test (`tests/invariantes/test_fronteras_del_vocabulario.py` found the quoted literal `"disponible"` in the UI); fixed in the app commit by rendering it as JSX text.

## 2. Commits (all local, nothing pushed)

| Repo   | Branch                  | Commit                                                            | Message                                                                                                             |
| ------ | ----------------------- | ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| app    | `feat/mrcd-integration` | `226427c`                                                         | feat(report): add MRCD disfluency section and service client (frontend)                                             |
| app    | same                    | `7353b34`                                                         | feat(backend): analyze session audio with MRCD from an in-memory buffer (privacy option B)                          |
| app    | same                    | `1351710`                                                         | feat(report): read session disfluencies from the stored report; manual WAV upload is dev-only                       |
| app    | same                    | `0918597`                                                         | fix(backend): treat capture-time jumps as audio gaps; expose MRCD service URL in the local compose (off by default) |
| engine | `main`                  | see `git log` (docs + S7 evidence commit that contains this file) | docs(contracts)/results for S7                                                                                      |

Engine HEAD before this task: `d030e71`. App base: `f46a2b8`.

## 3. Versions and commands (from `comandos-ejecutar/COMANDOS.MD`)

node 22.17.0, Docker 29.1.3 (Docker Desktop started for the run), Python 3.12 venv of the app backend (host python 3.13.5 with fastapi 0.141.1 / pydantic 2.13.4 for the tests), Playwright (global `@playwright/cli`) driving the cached Chromium 149.0.7827.55 (chromium-1228, no download).

```
# MRCD service (engine repo), from COMANDOS.MD addendum
cd <repo> ; $env:OMP_NUM_THREADS="4"; $env:HF_HUB_OFFLINE="1"
.\.venv\Scripts\python.exe -m mrcd.service serve --port 8765        # 127.0.0.1 only, no auth
# App (real backend + Postgres + built frontend), from COMANDOS.MD
cd ...\ORATORIA-PROYECTO
$env:ORATORIA_MRCD_SERVICE_URL="http://host.docker.internal:8765"   # opt-in; compose default is empty
docker compose -f docker-compose.local.yml up --build -d
# E2E (engine repo)
node scripts/mejora_s7/e2e_session.mjs <playwright-dir> <chrome.exe> results/.../inputs/e2e_synthetic_es.wav <outDir> http://localhost:5173 47 normal
# Shutdown (COMANDOS.MD)
docker compose -f docker-compose.local.yml down                     # MRCD service: process killed
```

App-side ASR in the compose was the default `simulado` adapter (no Speechmatics credentials): it only feeds the app's own live feedback. The MRCD analysis is real (its own Whisper `small` p0 + V0 head).

## 4. Test results (literal summaries; logs in `logs/`)

- App backend (`logs/backend_checks.txt`): ruff format/check clean, mypy `Success: no issues found in 361 source files`, import-linter `Contracts: 18 kept, 0 broken`, pytest `1413 passed, 92 skipped` (the 92 skips are the existing suites that need a live Postgres; the new persistence path was exercised by the E2E below, not by a unit test).
- New backend tests: buffer (limit, gaps, capture-time jump, discard, odd length), analysis (success, pending->completed, MRCD errors -> `unavailable` with reason, failed analysis, hash mismatch, limit exceeded, gap/empty without calling MRCD, store failure, shutdown cancellation, no write-mode `open`, no new files in cwd/tmp, no audio in logs), client (real HTTP against a fake loopback server: success, 429 `queue_full`, unreachable, timeout, wrong schema), WebSocket integration (explicit close, abrupt disconnection, MRCD error, limit exceeded with the client error message, sequence gap, feature off, `GET /sessions/{id}` passthrough).
- App frontend (`logs/frontend_checks.txt`): `tsc -b` exit 0, `eslint .` clean, `vitest run` `65 files, 664 tests passed` (new: default 3 classes with timestamps and "Score (softmax, no calibrado)", no `%`, no psychological words, `No disponible` with reason and code, pending status, not requested, contract mismatch, dev upload hidden unless flag).

## 5. End-to-end evidence (nothing of the app simulated; ASR of the app = `simulado`)

Input: `inputs/e2e_synthetic_es.wav`, 45.5 s synthetic Spanish (Windows SAPI voice "Helena", text written by the team with fillers and repetitions; no third-party content), injected as microphone with `--use-fake-device-for-media-stream --use-file-for-fake-audio-capture=<wav>%noloop`.

| Run                           | Path                                                                                                                   | Result                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `run1` (canonical)            | register in the UI -> configure -> start -> record 47 s -> "Finalizar grabacion" -> "Cerrar sesion" -> `/reports/<id>` | WS: 294 binary frames = 47.2 s of audio, `session_closed` received; report first shows the section (poll `pending`), `completed` 16 s later: 9 in-scope events (4 `repetition`, 4 `filler_word`, 1 `prolongation`), 19 out-of-scope events hidden and excluded from totals, 11.5 events/min with "Audio analizado: 47.0 s", Video "No disponible: service v1 receives audio only...", contract revision 1.2, `asr_profile p0`; `audio_sha256` in the binding equals the service's hash (`7b874bb2...`); no `%` in the section; label "Score (softmax, no calibrado)" present |
| `run_abrupt`                  | page closed mid-recording (no `close_session`)                                                                         | analysis still ran: `completed`, 20.0 s buffered, 3 events, verified through `GET /api/v1/sessions/{id}`                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `run_short`                   | 9 s recording                                                                                                          | service answered `audio_too_short`; UI: "No disponible: Audio of 9.0 s; the engine needs at least 13 s." + code; no zero                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `run_mrcd_down`               | MRCD process killed, 16 s recording                                                                                    | session closed normally; UI: "No disponible: MRCD service is unreachable" + `mrcd_unreachable`                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `run1_first_pass_pre_rebuild` | same as run1 before the capture-time-jump fix                                                                          | kept as evidence of the first pass (same hash, 9 events)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |

Privacy checks (`logs/privacy_checks_final.txt`, after the rebuilt stack ran run1 and run_mrcd_down): `docker diff` of the backend container is empty (no file written to its layer), zero matches for RIFF/WAVE/pcm16/data:audio in the backend log, the only MRCD log line is `MRCD unavailable for session ...: mrcd_unreachable`, and the DB holds the JSON only (`pg_column_size` 0.9-3.8 kB per session; no audio column exists). Screenshots (`run*/0*.png`) and `session_detail.json` contain only synthetic audio results; no transcript text is requested (`include_text=false`).

## 6. Limitations

- The real microphone has NOT been tried by the team yet; the browser here used a fake capture device (echo cancellation/noise suppression/AGC settings of the app apply to a file, not to a room). The effect of browser DSP on the checkpoint features is still unmeasured.
- The synthetic TTS speech is not representative of spontaneous disfluent speech; event counts here say nothing about accuracy (no human reference). Scores are raw softmax values, not calibrated.
- The app's live mode still runs ORATORIA-ENGINE v0 / the `simulado` adapter; the MRCD analysis happens only after the session closes.
- One WAV per session: the concatenated audio is analysed only if contiguous (sequence numbers consecutive and capture time within 1.5 s); a pause or lost chunks make the whole session `unavailable / audio_gap` (no splitting into runs yet).
- No authentication or TLS between the app and MRCD (loopback / internal network only); no cap on concurrent buffered sessions (up to 19.2 MB each at the default cap); the immutable `bytes` copy of the WAV cannot be zeroed and waits for the garbage collector; a memory dump of the backend during the analysis would expose the audio.
- The registration consent text does not mention the in-memory audio analysis; ethics approval and legal basis are open (see `PRIVACY_DECISION.md`). Advisor ratification of option B is pending.
- Postgres volume `oratoria_local_pg` was kept after `down`; it contains the E2E accounts (`e2e<timestamp>@example.com`, synthetic) and sessions. The app Alembic migration `a7c3e5f9b1d2` is applied to that local DB only.
- Postgres-backed repository methods are covered by the E2E, not by an automated test (those suites skip without a database).
- Other Docker containers on the machine (`supabase_*`) are not part of this work and were not touched.

## 7. How to revert

Nothing was pushed or merged. The user's branch `refactor/f0-guardrails` is untouched (still `f46a2b8` plus the user's own `frontend/.gitignore` edit). To drop the work: `git switch refactor/f0-guardrails` then `git branch -D feat/mrcd-integration` in the app repo. To undo the local DB column: `alembic downgrade f5b8e2a4c6d1` (drops `mrcd_analysis`) or remove the `oratoria_local_pg` volume. In the engine repo revert the S7 docs commit. To switch the feature off without reverting: leave `ORATORIA_MRCD_SERVICE_URL` empty (default) and `VITE_MRCD_DEV_UPLOAD` unset.

## 8. Self-review declaration

Written and checked by the same author. Pending: independent review of the backend diff (especially the discard paths and the WebSocket handler), a test with a real microphone, advisor ratification of the privacy decision and a consent text that mentions the analysis.


## 9. Addendum 2026-11-01 (corrections after REVISION_S7)

**Self-review by the authors; no independent re-review yet.** App commits (local, nothing pushed, branch `feat/mrcd-integration`): `e9246dc` backend, `c3fd73d` compose opt-in, `21ada69` frontend, `c5fd276` and `aa826b2` test fixes. Engine repo: docs only (`PRIVACY_DECISION.md` section 7, `APP_INTEGRATION_CONTRACT.md` B.12 addendum) plus evidence in `correcciones_20261101/`. The user's `frontend/.gitignore` edit is in no commit.

### 9.1 Changes
- **P1 consent (high).** Separate, UNCHECKED registration checkbox with the reviewer's text, marked in code and UI "Borrador pendiente de revisión ética y del asesor"; per-account decision with date and text version (`mrcd_consent`, migration `b8d4f6a2c0e3`, `GET/PUT /api/v1/users/me/mrcd-consent`); a decision only counts for the current text version; withdraw (optionally deleting stored results) from the new Privacy page (`/privacidad`). Without active consent: nothing is buffered, MRCD is not called, report says "No disponible: sin consentimiento para análisis de audio". Consent is re-checked at hand-off; a store error fails closed. The registration links no longer point to `#`: the policy link goes to a real draft page (`/privacidad-analisis`, based on option B); the Terms of service have no page yet, so the text says "(aún no publicados)" without a link.
- **P4.** Allow-list of the stored JSON (`result_schema.py`): `text`/`transcript` always null, unknown fields dropped (names logged, never content), 2 MiB read cap, 512 KiB / 5000 events stored cap. The four real service fixtures pass unchanged.
- **P2.** Global, per-account, byte and pending-analysis limits (`capacity.py`, env `ORATORIA_MRCD_MAX_*`), clear client `error` and `unavailable` codes, usage metrics at `GET /api/v1/mrcd/usage` (superusers).
- **P3/P5/P6.** Cancellable worker thread on its own executor, bounded stop (5 s, `asyncio.wait`), request timeout 15 s; orphaned `pending` documents expire to "análisis interrumpido" at start-up and every minute; the front stops polling after 15 min and says so; a task cancelled before its first step still releases the buffer.
- **P7/P8.** Discard guarantee documented precisely (class docstring, privacy page, contract); single WAV copy (`b"".join`); `PRIVACY_DECISION.md` contradiction fixed (exact persisted set).
- **S2.** No redirects, `job_id` regex, `asr_profile` validated/encoded, plain `http://` to a non-loopback host refused unless `ORATORIA_MRCD_ALLOW_INSECURE_REMOTE_HTTP=true` (compose passes it through, default false).
- **S6.** The manual WAV upload is tree-shaken: with the flag unset, `dist` contains neither the dev-tool text nor the direct MRCD client (checked with `vite build`, flag off vs on); `frontend/.dockerignore` now excludes `.env*`.

### 9.2 Tests
COMANDOS.MD has no pytest/vitest commands for the app, so the suites use the project tooling (venv pytest, `npx vitest run`, as in the first S7 logs); compose and MRCD service commands are the literal ones from COMANDOS.MD. Logs in `correcciones_20261101/logs/`: backend ruff clean, mypy clean (345 files), import-linter 18 kept, pytest `1460 passed, 92 skipped`; frontend `tsc -b` 0, eslint 0, vitest `68 files, 677 tests passed`. The 92 skips are still the Postgres suites; the new Postgres paths (consent upsert, pending listing, deletion, migration) were exercised only by the E2E below.

### 9.3 Mutants (fresh clone `mut_s7c_clone` in the scratchpad, outside both repos)
The 5 survivors are now killed: B2 (constant-name temp file; test guards `builtins.open`, `io.open`, `os.open` and `tempfile`, not just a listing), B4 (4 variants: transcript copy, allow-list bypass, sanitizer keeping `text`/`transcript`), B10 (white-box handler test), B13 (spy `bytearray` proves zeros before release), F5 (dash, not 0). Extra mutants for the new code: consent ignored / not re-checked / fail-open / stale version, memory cap, orphan expiry, buffer release on early cancel, redirects, `job_id`, http opt-in: all killed. One SURVIVOR: "client ignores the cancellation event while polling" (replaces the interruptible wait by `time.sleep`): near-equivalent, the worker still checks the event before every request, so it stops at the next poll (at most one poll interval, 1 s by default). See `logs/mutants_out.txt` and `logs/mutants_s7b.py`.

### 9.4 Real E2E (docker compose + MRCD service + Chromium fake microphone with `e2e_synthetic_es.wav`; `scripts/mejora_s7/e2e_session_consent.mjs`)
- **With consent** (box ticked in the UI): API reports `active: true`; 293 frames / 47 s; report `completed`, 10 events, `transcript` and `events[].text` null in the stored JSON (3988 bytes), softmax label present, no `%`.
- **Without consent** (box left unticked, default false): `active: false`; report `unavailable / no_consent`, `binding.n_chunks = 0`, text "No disponible: sin consentimiento para análisis de audio" shown; no analysis section data.
- Privacy checks (`logs/privacy_checks.txt`): `docker diff` empty, 0 RIFF/WAVE/pcm16 matches in the backend log, DB holds JSON only. Limit: the development MRCD service does not log requests, so "MRCD never called" for the no-consent run is supported by the document, the integration test and the absence of a job, not by a service-side count.
- Everything was shut down afterwards (`docker compose -f docker-compose.local.yml down`, MRCD process killed; ports 8000/5173/8765 free). The Postgres volume `oratoria_local_pg` still holds the synthetic E2E accounts.

### 9.5 Pending / open
1. Ratification by the ethics committee and the advisor of the consent text, the draft privacy page and option B (all marked DRAFT). 2. Base legal (Ley 29733), Terms of service text, contact channel. 3. Real microphone test (still not done). 4. No auth/TLS between app and MRCD. 5. Postgres paths lack automated tests. 6. Existing accounts have no decision (treated as no consent) and there is no first-session prompt: they must opt in from Privacy. 7. The `C:\tmp_unused` empty directory was created by mistake and could not be removed (blocked by a safety check): delete it manually. 8. Independent re-review of this addendum.
