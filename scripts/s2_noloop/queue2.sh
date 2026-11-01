#!/usr/bin/env bash
# second serial queue (after the GPU analysis): conditional D_B on GPU, CPU int8 confirmation of B, CPU C0 determinism check
cd "${MRCD_REPO:-$(cd "$(dirname "$0")/../.." && pwd)}"
L=results/mejora_objetivos_20261101/S2/logs
bash scripts/s2_noloop/run_gpu.sh D_B > $L/gpu_D_B.log 2>&1
export OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1
.venv/Scripts/python.exe scripts/s2_noloop/run_branch.py --branch B --device cpu > $L/cpu_B.log 2>&1
.venv/Scripts/python.exe - > $L/cpu_ids20.log 2>&1 <<'PY'
import json
u=json.load(open('results/cierre_oe1_oe3_20261031/3b/asr/units.json',encoding='utf8'))
ids=[x['unit_id'] for x in u[::7]][:20]
json.dump(ids,open('results/mejora_objetivos_20261101/S2/ids20.json','w'))
PY
.venv/Scripts/python.exe scripts/s2_noloop/run_branch.py --branch C0 --device cpu --out-suffix _det --ids results/mejora_objetivos_20261101/S2/ids20.json > $L/cpu_C0_det.log 2>&1
echo DONE > $L/queue2.done
