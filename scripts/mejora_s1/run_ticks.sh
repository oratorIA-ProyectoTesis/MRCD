#!/usr/bin/env bash
# usage: run_ticks.sh <config> <tick> [limit]
set -u
cd "${MRCD_REPO:-$(cd "$(dirname "$0")/../.." && pwd)}"
G="${MRCD_GPU_NVIDIA_DIR:-$(pwd)/../_venv_gpu/Lib/site-packages/nvidia}"
export PATH="$G/cublas/bin:$G/cudnn/bin:$G/cuda_nvrtc/bin:$PATH"
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4
.venv/Scripts/python.exe scripts/mejora_s1/tick_bench.py --config "$1" --tick "$2" --out results/mejora_objetivos_20261101/S1/ticks ${3:+--limit $3} 2>&1 | tail -3
