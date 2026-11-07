#!/bin/sh
# MRCD container entrypoint. Extra arguments (e.g. --cors-origin https://example.org) are appended to `serve`.
#
#   MRCD_CKPT     checkpoint path inside the container   (default /models/ckpt_audio_text.pt)
#   MRCD_DEVICE   "cpu" pins the ASR device; empty = auto (used by the CUDA image; "--device" only accepts cpu)
#   MRCD_PRELOAD  profile(s) loaded at start           (default p0; p0_cuda in the CUDA image)
#
# The trained checkpoint is NOT part of the image. Without it every engine load fails with `checkpoint_unavailable`
# (HTTP 503 / job status "failed"), and preloading would abort the process, so in that case the service starts
# lazily: /v1/health answers 200, /v1/schema works, and analysis jobs report `checkpoint_unavailable`.
set -eu

CKPT="${MRCD_CKPT:-/models/ckpt_audio_text.pt}"
DEVICE="${MRCD_DEVICE-cpu}"
PRELOAD="${MRCD_PRELOAD:-p0}"

# 0.0.0.0 is required inside the container for port publishing; publish it on the HOST loopback only
# (-p 127.0.0.1:8765:8765). The service refuses to start off loopback without MRCD_SERVICE_TOKEN (>= 32 chars). The CLI forces Swagger (/docs) off when
# --allow-non-loopback is used.
set -- --host 0.0.0.0 --allow-non-loopback --ckpt "$CKPT" "$@"
if [ -n "$DEVICE" ]; then
  set -- --device "$DEVICE" "$@"
fi

if [ -f "$CKPT" ]; then
  set -- --preload "$PRELOAD" --warmup "$@"
  # --incremental (/v1/sessions) requires --device cpu, --preload and --warmup.
  if [ "$DEVICE" = "cpu" ]; then
    set -- --incremental "$@"
  fi
else
  echo "mrcd: checkpoint not found at $CKPT; starting without it (analysis returns checkpoint_unavailable)." >&2
  echo "mrcd: mount it with: -v \"\$PWD/models:/models:ro\"" >&2
fi

exec python -m mrcd.service serve "$@"
