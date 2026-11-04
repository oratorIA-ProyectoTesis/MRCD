---
name: mrcd-install
description: "Trigger: install MRCD, instalar, run MRCD, start service, docker, compose, GPU, checkpoint, pip. Choose an install path, obtain the checkpoint and verify the service."
license: MIT
metadata:
  author: oratorIA-ProyectoTesis
  version: "1.0"
---

## Activation Contract

Use when the user wants to get the MRCD service running. For failures after start, use `mrcd-troubleshoot`.

## Hard Rules

- Publish the port on host loopback only: `-p 127.0.0.1:8765:8765`. The service has no authentication.
- The checkpoint `ckpt_audio_text.pt` is NOT in the repo or images. Never download or fabricate one; point to `models/README.md` (on request, non-commercial research).
- Mount `./models` read-only (`:ro`); only a file from the authors.
- Images exist on `ghcr.io` only after a `v*` tag. If `docker pull` fails, use `docker build -t mrcd .` and run `mrcd`.

## Decision Gates

| Situation                             | Path                                                                                                                        |
| ------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| Default, any OS                       | Docker CPU: `docker run --rm -p 127.0.0.1:8765:8765 -v "$PWD/models:/models:ro" ghcr.io/oratoria-proyectotesis/mrcd:latest` |
| Linux/Windows+WSL2 with NVIDIA GPU    | Docker GPU: add `--gpus all`, image `:latest-cuda` (amd64 only; experimental profile `p0_cuda`; no `/v1/sessions`)          |
| Repo cloned, want hardened defaults   | `docker compose up --build` (GPU: `docker compose --profile gpu up --build mrcd-gpu`)                                       |
| No Docker, need to import the package | `pip install "mrcd[service] @ git+https://github.com/oratorIA-ProyectoTesis/MRCD@<tag>"` (Python >= 3.11)                   |

## Execution Steps

1. Ask which path only if the OS/GPU is unknown; otherwise pick from the table.
2. OS notes: Windows PowerShell uses `${PWD}` (`-v "${PWD}/models:/models:ro"`); Mac Apple Silicon pulls the arm64 CPU image (works, slower; no CUDA image); Linux GPU needs `nvidia-container-toolkit` and a working `nvidia-smi` on the host.
3. Put `ckpt_audio_text.pt` in `./models` (exact name). Override the path with env `MRCD_CKPT`.
4. pip path only: warm Whisper once, then serve:
   `python -c "from faster_whisper import WhisperModel; WhisperModel('small', device='cpu', compute_type='int8')"`
   `OMP_NUM_THREADS=4 python -m mrcd.service serve --ckpt ./models/ckpt_audio_text.pt --device cpu --preload p0 --warmup --incremental`
5. Verify: `curl -sf http://127.0.0.1:8765/v1/health` returns `"status":"ok"`; with a checkpoint wait for `engine_loaded` and `warmed_up` to be true (first start can take minutes).
6. End to end: `pip install jsonschema && python scripts/docker_smoke.py --url http://127.0.0.1:8765 --expect completed` (use `--expect checkpoint_unavailable` when no checkpoint is mounted).

## Output Contract

Report: path chosen, command run, health JSON, smoke result, and whether the checkpoint is present.

## References

- `INTEGRACION.md`, `models/README.md`, `docker/entrypoint.sh`, `docker-compose.yml`
