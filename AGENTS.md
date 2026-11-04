# AGENTS.md

MRCD detects filler words, prolongations and repetitions in Spanish speech. It runs as a local HTTP service (contract `mrcd.analysis/v1`) and as a Python package (`mrcd/`). Results are exploratory: scores are NOT calibrated probabilities.
The trained checkpoint `ckpt_audio_text.pt` is not distributed (on request, see `models/README.md`). Without it the service starts, `/v1/health` answers, and every job fails with `checkpoint_unavailable`.

## Repo map

| Path | What |
|---|---|
| `mrcd/service/` | CLI (`cli.py`), HTTP server (`server.py`), contract and error codes (`contract.py`), audio checks (`audio_io.py`) |
| `mrcd/service/schemas/mrcd.analysis.v1.schema.json` | JSON Schema, source of truth for the response |
| `INTEGRACION.md` (Spanish), `README.md` | Integration guide, project overview |
| `examples/` | `curl.sh`, `python_client.py`, `analyze.mjs` |
| `scripts/docker_smoke.py` | End-to-end smoke test of a running service |
| `Dockerfile`, `Dockerfile.cuda`, `docker/entrypoint.sh`, `docker-compose.yml` | Images and compose |
| `tests/` | pytest suite (some tests need local data and skip without it) |

## Golden commands

```bash
# Docker CPU (service on 127.0.0.1:8765; PowerShell: use ${PWD})
docker run --rm -p 127.0.0.1:8765:8765 -v "$PWD/models:/models:ro" ghcr.io/oratoria-proyectotesis/mrcd:latest
# Docker GPU (linux/amd64, NVIDIA Container Toolkit)
docker run --rm --gpus all -p 127.0.0.1:8765:8765 -v "$PWD/models:/models:ro" ghcr.io/oratoria-proyectotesis/mrcd:latest-cuda
# Compose
docker compose up --build                          # CPU
docker compose --profile gpu up --build mrcd-gpu   # GPU
# pip from a git tag (Python >= 3.11)
pip install "mrcd[service] @ git+https://github.com/oratorIA-ProyectoTesis/MRCD@<tag>"
python -m mrcd.service serve --ckpt ./models/ckpt_audio_text.pt --device cpu --preload p0 --warmup --incremental
# Dev + tests
pip install -e ".[service,dev]" && python -m pytest tests/test_service_server.py tests/test_service_contract.py -q
# Smoke test a running service (needs `pip install jsonschema`)
python scripts/docker_smoke.py --url http://127.0.0.1:8765 --expect completed   # or checkpoint_unavailable
```

Images: `ghcr.io/oratoria-proyectotesis/mrcd:{<version>,latest,<version>-cuda,latest-cuda}`, published only on `v*` tags. Until then: `docker build -t mrcd .`.

## Contract in brief

- `GET /v1/health`, `GET /v1/schema`, `POST /v1/jobs` (body: raw WAV, `Content-Type: audio/wav`, returns 202 + `job_id`), `GET /v1/jobs/{job_id}` (`queued|running|completed|failed`; `result` holds the analysis).
- Audio: WAV mono, 16 kHz, PCM16, 13 s to 600 s. Anything else fails with a coded error.
- `completed` result: `events[]` (`class`, `start_s`, `end_s`, `score`), `summary`. `failed` result: `error.{code,message,retryable,details}`. Codes: `mrcd/service/contract.py` (`ERROR_CODES`).
- `include_text=false` by default (no transcript unless asked).

## Hard rules

1. Never commit weights (`*.pt`), audio (`*.wav`, `*.mp3`...), corpora, recordings or results. They are git-ignored on purpose.
2. The service has NO authentication: bind `127.0.0.1` only (`-p 127.0.0.1:8765:8765`). Never publish it on `0.0.0.0` or the internet.
3. Never invent API fields, params or error codes. Read `mrcd/service/schemas/mrcd.analysis.v1.schema.json` and `contract.py` first.
4. Never present scores as calibrated or as probabilities. Never present results as validated performance.
5. The video branch never infers psychological states (anxiety, doubt, intent). Video is observable evidence only.
6. Absence is not zero: a measure is `{available, value, unavailable_reason}`; render unavailable as unavailable, never `0`.
7. Only mount a checkpoint obtained from the authors (it is loaded with `torch.load(weights_only=False)`).
8. Conventional commits only (`feat:`, `fix:`, `docs:`...).

## Skills and agents

| Name | Use it when |
|---|---|
| `.claude/skills/mrcd-install` | Install or start MRCD (Docker CPU/GPU, compose, pip), get the checkpoint, verify |
| `.claude/skills/mrcd-integrate` | Call the service from your app: submit WAV, poll, parse, handle errors |
| `.claude/skills/mrcd-troubleshoot` | A job failed, container won't start, slow, port busy, GPU not visible |
| `.claude/skills/mrcd-contribute` | Develop MRCD: venv, tests, commits, CI, release |
| `.claude/agents/mrcd-integration-helper.md` | Integration questions answered from schema/code with `file:line` citations |

Tools without native skill support: open the `SKILL.md` files above as plain docs.
