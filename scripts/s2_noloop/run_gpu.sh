#!/usr/bin/env bash
# usage: run_gpu.sh BRANCH [extra args]   (serial; CUDA 12 DLLs of the separate venv on PATH, project venv untouched)
set -u
cd "${MRCD_REPO:-$(cd "$(dirname "$0")/../.." && pwd)}"
G="${MRCD_GPU_NVIDIA_DIR:-$(pwd)/../_venv_gpu/Lib/site-packages/nvidia}"
export PATH="$G/cublas/bin:$G/cudnn/bin:$G/cuda_nvrtc/bin:$PATH"
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4
b="$1"; shift
.venv/Scripts/python.exe scripts/s2_noloop/run_branch.py --branch "$b" --device cuda "$@"
