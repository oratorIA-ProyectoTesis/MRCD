---
name: mrcd-troubleshoot
description: "Trigger: MRCD error, failed job, checkpoint_unavailable, queue_full, port in use, healthcheck, OOM, GPU not visible, lento, no funciona. Map a symptom to cause and fix."
license: MIT
metadata:
  author: oratorIA-ProyectoTesis
  version: "1.0"
---

## Activation Contract

Use when the service does not start, a job ends `failed`, or behavior is unexpected.

## Hard Rules

- Get evidence first: `curl -s http://127.0.0.1:8765/v1/health`, `docker logs <container>`, and the job's `result.error.{code,details}`.
- Never "fix" by exposing the port publicly, mounting a checkpoint from an unknown source, or reporting a missing measure as `0`.

## Decision Gates

| Symptom                                                                                       | Cause                                                                 | Fix                                                                                                |
| --------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| Job `failed`, `checkpoint_unavailable`                                                        | No `ckpt_audio_text.pt` at `/models`                                  | Request it (`models/README.md`), mount `./models:/models:ro`, exact filename; restart              |
| `engine_load_failed` / `head_unavailable`                                                     | Corrupt or mismatched checkpoint or head                              | Re-obtain the file; check logs                                                                     |
| `audio_too_short`                                                                             | Under 13 s                                                            | Send at least 13 s (`details.min_duration_s`)                                                      |
| `audio_too_long`                                                                              | Over 600 s                                                            | Split, or raise `--max-duration-s`                                                                 |
| `audio_not_mono`, `unsupported_sample_rate`, `unsupported_audio_format`, `invalid_audio_file` | Not a mono 16 kHz PCM16 WAV                                           | `ffmpeg -i in.mp3 -ac 1 -ar 16000 -c:a pcm_s16le out.wav`                                          |
| `audio_not_finite`                                                                            | NaN or infinite samples                                               | Re-export the audio                                                                                |
| `payload_too_large` (413)                                                                     | Body over 64 MiB                                                      | Shorter or re-encoded audio                                                                        |
| `queue_full` (429, retryable)                                                                 | 8 jobs pending; one runs at a time                                    | Back off and retry                                                                                 |
| `analysis_failed` (500, retryable)                                                            | Unexpected engine failure                                             | Retry once, then read `docker logs`                                                                |
| `asr_unavailable` with a `*_cuda` profile                                                     | Container cannot see the GPU                                          | `--gpus all`, `nvidia-smi` on host, nvidia-container-toolkit; CUDA profiles never fall back to CPU |
| `job_not_found` (404)                                                                         | Jobs are in memory: service restarted, or evicted (finished jobs beyond the most recent 32)      | Resubmit                                                                                           |
| `incremental_disabled` on `/v1/sessions`                                                      | Incremental needs checkpoint and the CPU image                        | Use the CPU image with a checkpoint (the CUDA image runs without `--incremental`)                  |
| `address already in use`                                                                      | Port 8765 taken                                                       | `-p 127.0.0.1:9000:8765`, point the client to 9000                                                 |
| Health slow to become ready                                                                   | Model load and warmup; HEALTHCHECK start period is 180 s (240 s CUDA) | `docker logs -f`; wait for `warmed_up: true`                                                       |
| Container `OOMKilled`                                                                         | Needs about 3 to 4 GB                                                 | Raise Docker Desktop memory                                                                        |
| Compose: write errors                                                                         | `read_only` root; tmpfs only on `/tmp` and `/home/mrcd`               | Do not write elsewhere                                                                             |
| Mac arm64 slow                                                                                | CPU-only Whisper, or amd64 image under emulation                      | Expected on CPU; use the native arm64 image                                                        |
| `/docs` is 404 in Docker                                                                      | Swagger is disabled off-loopback by design                            | Use `GET /v1/schema`                                                                               |
| Browser blocked by CORS                                                                       | Origin not allowed                                                    | Append `--cors-origin <origin>` to the run command                                                 |
| `pull access denied` / not found                                                              | Images appear only after a `v*` tag; package may be private           | `docker build -t mrcd .`                                                                           |

## Output Contract

State the cause with its evidence (log line, code, or JSON field) and the exact fix command.

## References

- `mrcd/service/contract.py` (`ERROR_CODES`), `docker/entrypoint.sh`, `INTEGRACION.md`
