---
name: mrcd-integrate
description: "Trigger: integrate MRCD, call the API, POST /v1/jobs, consumir, client, fetch, parse result, events, error codes. Submit a WAV, poll the job and parse the result per the JSON Schema."
license: MIT
metadata:
  author: oratorIA-ProyectoTesis
  version: "1.0"
---

## Activation Contract

Use when code must call a running MRCD service. If the service is not running, use `mrcd-install` first.

## Hard Rules

- Read `mrcd/service/schemas/mrcd.analysis.v1.schema.json` (or `GET /v1/schema`) before writing a parser. Never invent fields.
- `score.value` is uncalibrated: never show it as a probability or confidence percentage.
- A measure with `available: false` has `value: null` and an `unavailable_reason`: show "not available", never `0`.
- No transcript unless `include_text=true` is requested.
- Browser on another origin: start the service with `--cors-origin <origin>`; never expose it publicly without your own auth layer.

## Execution Steps

1. Audio must be WAV, mono, 16 kHz, PCM16, 13 s to 600 s (`DisfluencyEngine.MIN_AUDIO_S`; `--max-duration-s`). Convert: `ffmpeg -i in.mp3 -ac 1 -ar 16000 -c:a pcm_s16le out.wav`. Body limit is 64 MiB.
2. `POST /v1/jobs` with `Content-Type: audio/wav`. Optional query: `include_text`, `include_out_of_scope`, `asr_profile`, `decision_profile` (`o2026_11` default, or `v0`). Response 202: `{job_id, status: "queued", ...}`.
3. Poll `GET /v1/jobs/{job_id}` (1 s) until `status` is `completed` or `failed`. Jobs live in memory only (the last 32 are kept).
4. Parse `job.result`: `events[]` (`class`, `start_s`, `end_s`, `score`) and `summary` (e.g. `events_per_minute`). Intervals are decision strides, not human boundaries.
5. Errors surface in two places. Immediate HTTP errors: `invalid_request` 400, `payload_too_large` 413, `queue_full` 429 (retryable), `job_not_found` 404. Analysis errors arrive as `status: "failed"` with `result.error.{code,message,retryable,details}` (audio and checkpoint problems land here). Retry only when `retryable` is true (`analysis_failed`, `queue_full`), with backoff; every other code needs a fix. Full list: `ERROR_CODES` in `mrcd/service/contract.py`.
6. Snippets: `assets/client.py` (stdlib, with retry), `assets/client.ts` (fetch), `assets/curl.sh`. Runnable originals in `examples/`.
7. Validate any stored document: `python -m mrcd.service validate file.json`.

## Output Contract

Deliver working client code with the error-handling branch, and state that scores are uncalibrated.

## References

- `INTEGRACION.md`, `mrcd/service/server.py`, `mrcd/service/audio_io.py`, `examples/`
