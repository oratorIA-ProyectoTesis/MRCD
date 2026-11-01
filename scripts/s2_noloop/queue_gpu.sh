#!/usr/bin/env bash
# serial GPU queue: control, P1, the three alternatives, then the determinism re-run of C0
cd "${MRCD_REPO:-$(cd "$(dirname "$0")/../.." && pwd)}"
mkdir -p results/mejora_objetivos_20261101/S2/logs
for b in C0 P1 A B C; do
  bash scripts/s2_noloop/run_gpu.sh $b > results/mejora_objetivos_20261101/S2/logs/gpu_$b.log 2>&1
done
bash scripts/s2_noloop/run_gpu.sh C0 --limit 100 --out-suffix _rep > results/mejora_objetivos_20261101/S2/logs/gpu_C0_rep.log 2>&1
echo QUEUE_DONE > results/mejora_objetivos_20261101/S2/logs/gpu_queue.done
