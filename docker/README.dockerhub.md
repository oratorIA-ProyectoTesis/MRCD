# MRCD: disfluency detection for Spanish speech, as a local HTTP service

MRCD (Motor de Reconocimiento Contextual de Disfluencias) detects **filler words, prolongations and repetitions** in Spanish speech and returns them through the versioned JSON contract `mrcd.analysis/v1`. Everything runs inside the container: no audio leaves the machine.

- Source, tests and protocols: https://github.com/oratorIA-ProyectoTesis/MRCD (branch `mrcd-oe-closure`, MIT license)
- Release used for this image: https://github.com/oratorIA-ProyectoTesis/MRCD/releases/tag/v0.4.0
- Same image on GHCR: `ghcr.io/oratoria-proyectotesis/mrcd`

## Tags

| Tag                         | Content                                                                            |
| --------------------------- | ---------------------------------------------------------------------------------- |
| `0.4.0`, `latest`           | CPU build (Python 3.12, faster-whisper `small` baked in, non-root, health-checked) |
| `0.4.0-cuda`, `latest-cuda` | CUDA build for NVIDIA GPUs                                                         |

## Quick start

The trained checkpoint `ckpt_audio_text.pt` is **not inside the image** and is not distributed in any download. It is available on request from the corresponding author (Ely Cortez Flores, u202215313@upc.edu.pe) for non-commercial research use only. Put it in a local `models/` folder and mount it read-only (see `models/README.md` for its sha256; the service verifies the hash before loading).

```bash
docker run --rm -p 127.0.0.1:8765:8765 -v "$PWD/models:/models:ro" rivacortezf/mrcd:0.4.0
curl http://127.0.0.1:8765/v1/health
```

Without the checkpoint the container starts and `/v1/health` answers, but analysis jobs end in `failed` with the error code `checkpoint_unavailable`.

On PowerShell use `${PWD}` instead of `$PWD`. Extra service flags go after the image name, for example `--cors-origin https://example.org`.

## What you get back

One JSON document per analysed recording with the detected events (type, start, end, confidence) and an explicit **unavailable** state for anything that could not be measured: an unmeasured indicator is never reported as zero. The full contract, the error codes and an integration guide are in `INTEGRACION.md` of the repository.

## Scope

- Observable speech events only. The engine does not diagnose any condition and does not infer emotions or intentions.
- Scores are uncalibrated and the evaluation so far is presence-per-unit on corpora with pre-existing labels; a human-annotated reference is in progress. Read the limitations in the repository before using the numbers.

## Licenses

Code: MIT. The checkpoint is not distributed; it is available on request for non-commercial research use only (see `models/README.md`).

MRCD is part of a software-engineering thesis at Universidad Peruana de Ciencias Aplicadas (Cortez Flores, Ventura Chancafe; advisor Castro Veramendi).
