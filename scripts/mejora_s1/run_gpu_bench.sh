#!/usr/bin/env bash
# Serial S1 GPU/CPU benchmark. Adds the CUDA 12 libraries of the separate venv to PATH (project venv untouched).
set -u
cd "${MRCD_REPO:-$(cd "$(dirname "$0")/../.." && pwd)}"
G="${MRCD_GPU_NVIDIA_DIR:-$(pwd)/../_venv_gpu/Lib/site-packages/nvidia}"
export PATH="$G/cublas/bin:$G/cudnn/bin:$G/cuda_nvrtc/bin:$PATH"
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4
OUT=results/mejora_objetivos_20261101/S1/gpu
for c in "$@"; do
  .venv/Scripts/python.exe scripts/mejora_s1/gpu_bench.py --config "$c" --out "$OUT" ${ROUNDS:+--rounds $ROUNDS} 2>&1 | tail -3
done
