# S1 - OE3: end-to-end run and optional GPU ASR profile (2026-11-01)

**Self-reviewed by the author; an independent adversarial review was run on 2026-11-01 (`results/mejora_objetivos_20261101/REVISION_S/fix_logs/`), findings applied.** Exploratory. Defaults unchanged: V0, P0 (CPU int8), `live_cfg_v2`. The real app (`ORATORIA-PROYECTO`, HEAD `f46a2b8`, its own dirty `frontend/.gitignore` pre-existing) was not modified. No models or corpora were downloaded (`HF_HUB_OFFLINE=1`). Historical evidence stays on CPU.

## 1. E2E in a temporary clone (`e2e/`)
Commands (clone and `node_modules` were created OUTSIDE both repos, in the session scratchpad):
```
git clone --local <app> app_clone ; git checkout f46a2b8 ; git apply results/cierre_oe1_oe3_20261031/app_patch_package/0001-*.patch   # 15 files, +2473, applied clean
cd app_clone/frontend ; npm ci --prefer-offline --no-audit        # 297 packages, 11 s, from the npm cache (npm_ci.log)
OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 python -m mrcd.service serve --port 8765 --cors-origin http://localhost:5173
VITE_MRCD_SERVICE_URL=http://127.0.0.1:8765 npx vite --port 5173 --host localhost
node scripts/mejora_s1/e2e_run.mjs <playwright 1.61 module> <chromium-1228 chrome.exe> <wav> <outdir>
```
Versions: node 22.17.0, npm 11.4.2, vite 8.2.2, react ^19.2.8, Playwright 1.61.0-alpha (global `@playwright/cli`) driving the cached Chromium 1228 (no download).
Walk: `/reports/<id>` -> panel "Disfluencias por evento (experimental)" -> file picker with the repo WAV `data/annotation_batch_01/clipA_s00.wav` (36 s, 16 kHz mono; screenshots show no transcript) -> POST `/v1/jobs` (202) + 30 GET polls -> table.

| Run | Result |
|---|---|
| `run1` (36 s WAV, default p0 + V0, CPU) | OK in 31.5 s (RTF 0.88 incl. engine load and 1 s polling). 22 events, 36.7 events/min, "Audio analizado: 36.0 s". Each row: class (Spanish + internal key), `a s - b s` decision strip, score with "NO CALIBRADO" (no `%` anywhere), windows. Provenance + modalities: Acustica/Linguistica "disponible", Video "No disponible: service v1 receives audio only...". Console: only vite/React info messages. |
| `run2_synthetic_short` (5 s synthetic tone) | Error path OK: structured `audio_too_short` alert, nothing invented. |

Artifacts: `run*/01-03_*.png`, `network.har`, `checks.json` (service requests + stubbed app requests + console), `service_result.json` (the real `mrcd.analysis/v1` document, rev 1.1), `panel_text.txt`, `service.log`, `vite.log`.

**What is real and what is stubbed.** The MRCD service, the model, the browser and the patched React code are real. The app BACKEND (auth `GET /api/v1/users/me` and `GET /api/v1/sessions/<id>`, 3 requests) is stubbed at the network layer because the real one needs Docker + Postgres (not run). The stub report is empty on purpose; the rest of ReportPage (score 50, empty metric cards) is therefore not evidence about the real app flow.

**Scope decision (user, 2026-11-01) is NOT yet implemented** (S3 adds the service option). What the report shows today: all 7 non-fluent classes as the engine emits them. In `run1`: `filler_word` x5, `prolongation` x4, `rhetorical_pause` x7, `neutral_pause` x5 and `block` x1; `repetition` and `revision` did not occur. `block` and the pauses already carry the badge "SALIDA TECNICA SIN VALIDAR", but they are shown by default and counted in "EVENTOS" and "EVENTOS POR MINUTO" (22 events, of which 13 are out-of-scope classes). S3 must filter them by default.
Finding: the panel footnote says the score is "el maximo de la probabilidad de clase en las ventanas" although the service now names it `max_window_softmax_score` (rev 1.1); the sentence continues "no esta calibrado", but the word "probabilidad" should become "puntaje softmax" in the S3 patch update.

**External dependencies that prevent a fully offline E2E:** (1) `npm ci` needs the npm registry unless the cache is warm (it was); (2) the app backend path needs Docker images (postgres, python base, node/nginx) and, for a real ASR in the app, Speechmatics credentials/network (not exercised here: stubbed); (3) a browser binary (Playwright Chromium cached locally; otherwise a download); (4) Whisper `small` and the checkpoint must already be in the local cache (they were); (5) Playwright itself is a global npm package, not a project dependency. The clone is NOT deleted (see "Pending").

## 2. GPU ASR profile (`gpu/`, code in `mrcd/asr_profiles.py`, `mrcd/extractors/linguistic.py`)
- `p0_cuda`, `p1_cuda` registered (commit `4af76d4`): same decoding parameters/prompts as p0/p1, `device="cuda"`, `compute_type="float16"`, `experimental=True`. New optional fields `device`/`compute_type` on `ASRProfile`; `describe()` adds them only when set. **p0/p1 are bit-identical**: namespace, `asr_cache_key` and `describe()` hash compared before/after the change (pinned in `tests/test_asr_profiles_cuda.py`). A GPU profile demands CUDA: no device -> `ASRDeviceUnavailable` (model never loaded), failed probe -> `ASRDeviceUnavailable`, no silent CPU fallback; conflicting `device=` -> `ValueError`. 8 tests (registered/experimental/defaults, fingerprints, rejection, request cuda+float16, CPU path unchanged).
- Feasibility: the project venv has `ctranslate2 4.8.2` (CUDA build, sees 1 device) but **fails without `cublas64_12.dll`** (`RuntimeError: Library cublas64_12.dll is not found or cannot be loaded`). Fix without touching the project venv: new venv `<repo>/../_venv_gpu` (uv, Python 3.12) with `nvidia-cublas-cu12 12.9.2.10`, `nvidia-cudnn-cu12 9.27.0.42`, `nvidia-cuda-nvrtc-cu12 12.9.86` (~0.7 GB download); the project python runs with those `bin` dirs prepended to `PATH` (`scripts/mejora_s1/run_gpu_bench.sh`). RTX 5060 Laptop (Blackwell, 8 GB, driver 616.92, CUDA UMD 13.4) works; `float16` and `int8_float16` both load. Observed once: the very first float16 call of the probe took 28.8 s (one-time driver kernel JIT/cache for sm_120); later processes show 0.2-0.3 s for the first file. Not repeatable on demand, so it is a caveat, not a figure.
- Side effect to know: with `device="auto"` (the engine default) and those DLLs on `PATH`, `p0` would silently pick the GPU. Without them (default environment) it falls back to CPU int8 as before. The benchmark pins `device="cpu"` explicitly for the baselines.
- Method: 13 local files of 5-60 s (`results/cierre_oe1_oe3_20261031/3b/cost/inputs`, 449 s of audio), offline path `transcribe_chunked(8 s, 2 s)`, one config per process, serial, OMP=4, round 0 (cold) + warm rounds. Load gate: a round is dropped if system CPU > 15 % in the 2 s before it (other processes were active; background 9-24 %). Texts stay in `gpu/_texts` (not committed). `gpu/discarded_gate_failed/` holds the first GPU run (background 16-17 %), discarded and re-run.

| config | load s | cold first pass RTF | warm RTF median/file [min-max] | RSS peak MB | GPU MB (load / peak, delta of device total) | rounds kept |
|---|---|---|---|---|---|---|
| p0 CPU int8 | 1.39 | 0.320 | 0.311 [0.270-0.741] | 921 | n/a | 1 (round 2 had 36 % load) |
| p0_cuda float16 | 2.27 | 0.040 | 0.040 [0.033-0.057] | 921 | 743 / 889 | 4 |
| p1 CPU int8 | 1.76 | 0.424 | 0.476 [0.340-0.731] | 920 | n/a | 2 (run start sample 24 %, per-round samples <= 15 %) |
| p1_cuda float16 | 1.62 | 0.043 | 0.041 [0.036-0.050] | 942 | 742 / 823 | 4 |

Speed-up of the ASR stage about 8x (p0) and 11x (p1) per file; the whole default pipeline (ASR 90 % of 0.5 RTF) would drop to about RTF 0.1 (ASR 0.04 + features 0.04; estimate, not measured end to end). GPU memory is the delta of the device-wide used memory (NVML, WDDM has no per-process figure; other apps use ~2.7-3.5 GB and fluctuate, so +-100 MB). Warm CPU RTF has a content-dependent outlier (see below).

**Text 'desacuerdo' CPU int8 vs GPU float16 (same 13 units; word edit distance / CPU words; NOT a WER against a human reference, says nothing about which text is better):**

| pair | identical units | pooled desacuerdo | median/unit | max/unit | repetition-loop flag units (CPU, GPU) |
|---|---|---|---|---|---|
| p0 | 4 / 13 | 0.070 | 0.015 | 0.694 (`c13s`) | (1, 0) |
| p1 | 5 / 13 | 0.028 | 0.022 | 0.130 | (0, 0) |

The p0 maximum is one unit: the CPU p0 decode of `c13s` yields 85 words flagged as a repeated n-gram loop, the GPU decode 38 words. That is the same file the 3b cost report listed as an undiagnosed CPU outlier (RTF 1.2); the loop is the likely cause (consistent, not proven by a human reference). Without it the p0 disagreement is about 0.02. So GPU float16 text is NOT identical to CPU int8 text: downstream heads were trained on CPU text, hence `experimental`.
Not measured: WER against human references for the GPU text, effect on events/heads, `int8_float16` as a registered profile, video.

## 3. 0.5 s live tick with GPU (`ticks/`)
Route unchanged: `StreamingASR.merge_recent`, 13 s ring re-decoded every tick, warm-up 6 s, defaults untouched. Latency per tick, simulated backlog (a tick starts when audio is there AND the previous tick ended).

| config | tick | units | median s | p95 s | max s | RTF_tick pooled | ticks over budget | max lag s |
|---|---|---|---|---|---|---|---|---|
| GPU p0_cuda | 0.5 s | 11 (634 ticks) | 0.333 | 0.466 | 1.274 | 0.688 | 3.2 % | 1.27 |
| GPU p0_cuda | 2.0 s | 11 (169 ticks) | 0.340 | 0.424 | 1.271 | 0.170 | 0 % | 1.27 (< tick) |
| CPU p0 int8 (3 shortest units) | 0.5 s | 3 (72 ticks) | 2.907 | 3.762 | 5.726 | 5.717 | 100 % | 65.8 |
| CPU p0 int8 (3 units) | 2.0 s | 3 (21 ticks) | 2.577 | 4.509 | 4.618 | 1.373 | 100 % | 9.3 |
| GPU, same 3 units | 0.5 s | 3 | - | 0.349 worst unit | - | 0.565 | - | 0.35 |

**Answer:** with the GPU the existing streaming ASR route reaches real time ON AVERAGE at a 0.5 s tick (RTF_tick 0.69, median latency 0.33 s, p95 0.47 s < 0.5 s) and strictly at a 2 s tick (0 % over budget). At 0.5 s it is NOT strictly sustained: 3.2 % of ticks exceed 0.5 s (spikes to 1.27 s, 8 of 11 units), which produces transient lag of up to 1.3 s that recovers. CPU never keeps up at 0.5 s (RTF_tick 5.7, lag grows without bound) and not at 2 s either on these units (1.37). This is the ASR only: the live acoustic path (0.17-0.20 s per tick, about one core, from 3b) was measured separately; if both ran serially in one thread the sum (about 0.5 s) would not fit, so a real session needs them on separate threads (not tested). Load: background 12-23 % at start of the tick runs (the 15 % gate was not met at the start of some runs; per-tick latencies, not wall totals, are the evidence). No microphone, no end-to-end live session with GPU was run.

## 4. Suite
`OMP_NUM_THREADS=4 .venv\Scripts\python.exe -m pytest -q -rs` -> **844 passed, 10 skipped** in 250 s (`logs/pytest_full.log`; skips are the MRCD_SLOW tests). The `MRCD_SLOW=1` run was not repeated (S1 does not touch those paths).

## Reproduce
`scripts/mejora_s1/run_gpu_bench.sh p0_cpu p1_cpu p0_cuda p1_cuda` (ROUNDS=n env), `gpu_report.py <dir>`, `run_ticks.sh <config> <tick> [limit]`, `tick_report.py <dir>`, `e2e_run.mjs`. Commits: `4af76d4` (profiles + tests), `93dee33` (scripts), plus the one adding the small reports.

## Limitations / pending
- Self-review only. The clone `...\scratchpad\app_clone` (with `node_modules`) was NOT deleted: the sandbox denied every `rm -rf`/`fs.rmSync` on it. Delete it manually (path in the final report); it is outside both repos.
- Machine shared with other processes: background CPU 9-24 %, so some rounds were dropped and the p1 CPU baseline has 2 usable rounds, p0 CPU 1.
- E2E app backend is stubbed; patched panel copy "probabilidad de clase" should be reworded; scope default not implemented (S3).
- GPU profile is experimental and unvalidated against human references; the engine default stays CPU.
