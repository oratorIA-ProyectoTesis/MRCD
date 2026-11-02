# MRCD C4 architecture (local, versioned, regenerable)

Date: 2026-11-01. Scope: the four C4 levels plus two sequence diagrams of the system **as it exists in the inspected trees**, not a target design.

> **Inspected trees.** MRCD: `main` (HEAD printed by `verify_architecture.py`). App `ORATORIA-PROYECTO`: local branch `feat/mrcd-integration` (read-only, not merged). Anything committed later is not reflected; re-run the verifier after any change to `mrcd/` or to that branch.
>
> **What is current.** The app calls the MRCD service over loopback HTTP for accounts with an active, versioned consent. Default mode is `offline` (POST `/v1/jobs`); `incremental` (B incremental, `/v1/sessions`) and `incremental_verify` are opt-in. Tags in node titles: `[PRODUCT]` default path, `[OPT-IN]` only by explicit flag or parameter, `[RESEARCH]` kit or experiment that no product path imports.

## Files

| File                            | What                                                                                                                                                   |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `c4_1_context.{mmd,svg,png}`    | Level 1, system context: people, app, MRCD, cloud ASR, corpora; trust boundaries                                                                       |
| `c4_2_containers.{mmd,svg,png}` | Level 2, containers: browser SPA, backend API, ASR port, embedded ENGINE v0, PostgreSQL, MRCD library, live demo, scripts, service, artifacts, results |
| `c4_3_components.{mmd,svg,png}` | Level 3, components of package `mrcd`; edges are observed imports (offline path purple, live path blue)                                                |
| `c4_4_code.{mmd,svg,png}`       | Level 4, code: central types and entry points; every class/function/method named is verified by AST                                                    |
| `seq_b_original.{mmd,svg,png}`  | Sequence: B original (buffer in app memory, one WAV to `POST /v1/jobs` at close)                                                                        |
| `seq_b_incremental.{mmd,svg,png}` | Sequence: B incremental (open session + token, `PUT audio/{seq}`, close, counts-only SSE, offline fallback)                                          |
| `render_sequences.sh`           | Renders the `seq_*.mmd` to SVG/PNG with the local Mermaid CLI                                                                                          |
| `c4_model.py`                   | **Single data model** of the four diagrams + deterministic renderer (SVG and Mermaid source from the same data)                                        |
| `verify_architecture.py`        | Checks that diagrams do not contradict the code                                                                                                        |
| `render_png.sh`                 | Optional PNG rasterisation with the locally installed Chrome (headless)                                                                                |

## How they were produced (no remote service)

- **C4 SVG** is rendered by `c4_model.py` (standard library only; absolute coordinates in the model; byte-identical on re-run). **C4 `.mmd`** come from the same model.
- **Sequence `.mmd`** are written by `c4_model.py` (`SEQUENCES`); `render_sequences.sh` renders them with `@mermaid-js/mermaid-cli` 11.4.2 and the local Chrome.
- All `.mmd` (C4, sequences, `docs/contracts/disambiguation_states.mmd`) compile locally: `render_mermaid_check.sh` reports 7/7 OK (2026-11-01).
- **PNG** of the C4 diagrams is a convenience raster of the SVG (`render_png.sh`, headless Chrome); not a source.

One-time Mermaid CLI install (outside the repo, no Chromium download):

```bash
TOOLS=~/.cache/mrcd-mmdc; mkdir -p "$TOOLS" && (cd "$TOOLS" && PUPPETEER_SKIP_DOWNLOAD=1 npm i --prefix . @mermaid-js/mermaid-cli@11.4.2)
```

Commands (repo root, Git Bash; Python 3.12.10 from `.venv`):

```bash
.venv/Scripts/python.exe docs/architecture/c4_model.py          # writes the 4 C4 .svg/.mmd and the 2 seq_*.mmd
bash docs/architecture/render_png.sh                            # optional, writes the 4 C4 .png (needs Chrome)
bash docs/architecture/render_sequences.sh                      # writes seq_*.svg and seq_*.png (Mermaid CLI + Chrome)
bash docs/architecture/render_mermaid_check.sh /tmp/mmd_out     # compile-checks every .mmd; exit 0 = all compile
.venv/Scripts/python.exe docs/architecture/verify_architecture.py   # exit 0 = no contradiction found
```

To change a diagram edit `c4_model.py` (nodes, coordinates, edges) and re-run; do not edit the generated files by hand.

## Legend

Blue = exists in the MRCD tree; green = exists in the app repo (read-only inspection); grey = external/data; yellow = person; **dashed orange = PROPOSED, not implemented** (only edges remain proposed after commit `955d27c`). Edge styles: black = request/response; blue = live/streaming path; purple = offline/batch path; grey dotted = file/model read or write; orange dashed = proposed. Red dashed frames = **trust boundaries**; grey dotted frames = grouping only. Colour is never the only cue (line dash and legend text carry the same information).

## What the diagrams state (and what each claim rests on)

| Claim drawn | Evidence (checked by `verify_architecture.py`) |
| --- | --- |
| Service: `/v1/health`, `/v1/schema`, `POST /v1/jobs`, `GET /v1/jobs/{id}` (offline reference and fallback) | `mrcd/service/server.py` |
| Incremental: `POST /v1/sessions` (returns `subscription_token`), `PUT .../audio/{seq}`, `POST .../close`, `GET .../events` (SSE, counts only), `DELETE` | `mrcd/service/incremental/routes.py`, `session.py` |
| Per-session ASR chunk cache, never persisted; close reruns the offline path | `incremental/cache.py`, `cached_asr.py`, `session.py` |
| Profiles: p0 default; p1, `*_cuda`, `*_noloop` opt-in; noloop and cuda refused in incremental | `mrcd/asr_profiles.py`, `SessionRegistry.open` |
| Preload and `--warmup` at service start | `server.prepare`, `incremental/warmup.py`, `cli.py` |
| App: consent versioned, mode default `offline`, profile p0, batch 2000 ms, `X-Session-Token`, SSE with fetch + Bearer, report scoped to 3 classes, HTTP only | app source on `feat/mrcd-integration` |
| `av`, `context_ambiguity`, `recording_check` are imported by no engine, extractor, live or service module; `optimize` is imported only by `service/decision_profile.py` (through `safe_pipeline`, the pickle-free JSON loader); `fusion_text` and `calibration` reach the service only through `s3_features` | AST import scan |
| `decision.py` is not used by the engine; analyze() rejects visual modes | AST scan; guard string in `engine.py` |
| Live rows are windows, not events; no event merge on the live path | `LIVE_CONTRACT.md` |
| Trust boundaries: video never leaves the browser; MRCD process is local; cloud only if a cloud adapter is configured (configuration used: Speechmatics ASR, OpenAI LLM) | app capture sources; `ORATORIA_ASR_ADAPTER` code default `simulado`, local compose default `whisper_http` |

## Verification result (this run)

`verify_architecture.py`: every `modules` path exists; every named symbol exists (AST); every `import` edge has a real import; the negative and package-tag claims above hold; the app-side and service-side facts match the source; no proposed node remains; every library module belongs to a component. Output: 202 checks passed, 0 failed.

## Limits

- The layout is hand-placed: edge crossings are minimised but not optimal; labels sit on the longest segment.
- Components at level 3 group modules by responsibility as written in `results/cierre_oe1_oe3_20261031/inventario_modulos.md`; that grouping is a reading aid, not a package structure.
- Research kits (`annotation`, `corpora`, `context_eval`, `asr_recovery`, `av`, `optimize`, `context_ambiguity`) are shown for completeness; they are not part of the engine path. Some import edges between far-apart level 3 components (service to warm-up and `s3_features`, cache to `VerbatimASR`, `s3_features` to `calibration`, `optimize` to `fusion_text`) are verified but not drawn, to keep the diagram readable.
- Performance, thread model and deployment topology are out of scope.
