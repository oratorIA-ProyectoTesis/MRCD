#!/usr/bin/env bash
# Starts the MRCD dev service with B incremental enabled (fixed order: device cpu -> preload -> warm-up).
# usage: start_mrcd.sh [log-file]   (OMP_NUM_THREADS=4 and HF_HUB_OFFLINE=1 are set here, before Python starts)
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1
exec .venv/Scripts/python.exe -m mrcd.service serve --port 8765 --cors-origin http://localhost:5173 \
  --incremental --device cpu --preload p0 --warmup >> "${1:-/dev/null}" 2>&1
