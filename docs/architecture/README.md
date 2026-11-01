# MRCD C4 architecture (local, versioned, regenerable)

Date: 2026-10-31. Scope: the four C4 levels of the system **as it exists in the inspected tree**, not a target design.

> **Warning: this is the architecture of the inspected tree.** First inspection: commit `46b4d1dcc398eb95` (`main`). While this folder was being written another task committed `955d27c` (`mrcd/service`, contract v1); the diagrams and the verifier were updated to that tree (HEAD `955d27c673a039541dc400d773b4559e01c6b6cd`, plus an uncommitted edit to `tests/test_service_server.py` by that task). Anything committed later is not reflected. `python docs/architecture/verify_architecture.py` prints the HEAD it ran against; re-run it after any change to `mrcd/`.
>
> The web app (`ORATORIA-PROYECTO`, inspected read-only @ `f46a2b8638b221e44d3cf492cf517f33b68dd0cb`) **embeds ORATORIA-ENGINE v0 and does not call MRCD**. The integration app -> MRCD is drawn dashed orange as PROPOSED because no client for it exists in the app (checked by the verifier).

## Files

| File                            | What                                                                                                                                                   |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `c4_1_context.{mmd,svg,png}`    | Level 1, system context: people, app, MRCD, cloud ASR, corpora; trust boundaries                                                                       |
| `c4_2_containers.{mmd,svg,png}` | Level 2, containers: browser SPA, backend API, ASR port, embedded ENGINE v0, PostgreSQL, MRCD library, live demo, scripts, service, artifacts, results |
| `c4_3_components.{mmd,svg,png}` | Level 3, components of package `mrcd`; edges are observed imports (offline path purple, live path blue)                                                |
| `c4_4_code.{mmd,svg,png}`       | Level 4, code: central types and entry points; every class/function/method named is verified by AST                                                    |
| `c4_model.py`                   | **Single data model** of the four diagrams + deterministic renderer (SVG and Mermaid source from the same data)                                        |
| `verify_architecture.py`        | Checks that diagrams do not contradict the code                                                                                                        |
| `render_png.sh`                 | Optional PNG rasterisation with the locally installed Chrome (headless)                                                                                |

## How they were produced (no remote service, no Figma, no mermaid.live)

Renderer check on this machine (2026-10-31): `mmdc` not installed, `dot` (Graphviz) not installed, `plantuml` not installed (Java 24 exists but no PlantUML jar), `cairosvg` not installed. Node 22 and `npx` exist, but a Mermaid CLI would need to download Chromium, so nothing was installed. Therefore:

- **SVG** is rendered by `c4_model.py` (standard library only; hand-rolled layout with absolute coordinates stored in the model; byte-identical on re-run).
- **Mermaid sources (`.mmd`)** are generated from the same model (flowchart syntax with subgraphs for boundaries). **They were NOT render-tested**: no Mermaid renderer is installed. Treat the SVG as the rendered artifact and the `.mmd` as editable source that has not been compiled.
- **PNG** is a convenience raster of the SVG made with Chrome 156.0.8078.74 headless (`render_png.sh`); it is not part of the source of truth.
- Nothing was installed; no model or corpus was touched.

Commands (repo root, Git Bash; Python 3.12.10 from `.venv`):

```bash
.venv/Scripts/python.exe docs/architecture/c4_model.py          # writes the 4 .svg and 4 .mmd
bash docs/architecture/render_png.sh                            # optional, writes the 4 .png (needs Chrome)
.venv/Scripts/python.exe docs/architecture/verify_architecture.py   # exit 0 = no contradiction found
```

To change a diagram edit `c4_model.py` (nodes, coordinates, edges) and re-run; do not edit the generated files by hand.

## Legend

Blue = exists in the MRCD tree; green = exists in the app repo (read-only inspection); grey = external/data; yellow = person; **dashed orange = PROPOSED, not implemented** (only edges remain proposed after commit `955d27c`). Edge styles: black = request/response; blue = live/streaming path; purple = offline/batch path; grey dotted = file/model read or write; orange dashed = proposed. Red dashed frames = **trust boundaries**; grey dotted frames = grouping only. Colour is never the only cue (line dash and legend text carry the same information).

## What the diagrams state (and what each claim rests on)

| Claim drawn                                                                                           | Evidence                                                                                                                                                                                                                             |
| ----------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Trust boundaries: browser (video never leaves it), app backend, third-party cloud, MRCD local process | app capture sources (`APP_INTEGRATION_CONTRACT.md` Part A); the cloud boundary exists only if a cloud ASR adapter is configured (default `simulado`)                                                                                 |
| Offline path: scripts -> `DisfluencyEngine.analyze_file` -> `AnalysisResult`                          | `mrcd/engine.py`; there is **no CLI around `analyze_file`** besides the new `mrcd.service`                                                                                                                                           |
| Live path: microphone -> `LiveScorer` rows via `scripts/live_demo.py`                                 | `mrcd/live/*`, `scripts/live_demo.py` (port 8001, microphone-bound dashboard; not an API for other systems)                                                                                                                          |
| Live rows are windows, not events; no event merge on the live path                                    | `INFORME_FINAL` section (g), `LIVE_CONTRACT.md`                                                                                                                                                                                      |
| MRCD with video                                                                                       | `DisfluencyEngine.analyze` raises for visual modes; kinesic extractors, face tracking and the sync buffer are reachable only from offline video scripts; only the `audio_text` checkpoint exists locally (verifier checks all three) |
| `decision.py` is not used by the engine                                                               | the verifier finds importers only in research kits (`context_eval`)                                                                                                                                                                  |
| App hosts ORATORIA-ENGINE v0, not MRCD                                                                | app adapters (`oratoria_engine`); no occurrence of `mrcd` in the app source (verifier scans `backend/src`, `frontend/src`, `contracts`)                                                                                              |
| Output/reports                                                                                        | engine returns JSON (`AnalysisResult.to_dict`), live rows and session JSON download; `ReportPage.tsx` has no disfluency section                                                                                                      |

## Verification result (this run)

`verify_architecture.py`: every `modules` path exists; every named symbol exists in its file (AST); every edge marked `import` has at least one real import between the two components (top-level or lazy); negative claims hold; the app has no MRCD client; every library module belongs to a drawn component. See the script output for the count and the HEAD it ran against.

## Limits

- The layout is hand-placed: edge crossings are minimised but not optimal; labels sit on the longest segment.
- Components at level 3 group modules by responsibility as written in `results/cierre_oe1_oe3_20261031/inventario_modulos.md`; that grouping is a reading aid, not a package structure.
- Research kits (`annotation`, `corpora`, `context_eval`, `asr_recovery`) are shown for completeness; they are not part of the engine path.
- Performance, thread model and deployment topology are out of scope.
