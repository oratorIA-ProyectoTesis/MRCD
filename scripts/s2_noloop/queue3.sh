#!/usr/bin/env bash
# third serial queue: CPU int8 confirmation of the best branch (D_B, protocol 4.3/4.5); waits for queue2
cd "${MRCD_REPO:-$(cd "$(dirname "$0")/../.." && pwd)}"
L=results/mejora_objetivos_20261101/S2/logs
while [ ! -f $L/queue2.done ]; do sleep 20; done
export OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1
.venv/Scripts/python.exe scripts/s2_noloop/run_branch.py --branch D_B --device cpu > $L/cpu_D_B.log 2>&1
echo DONE > $L/queue3.done
