#!/bin/bash
# Phase 3b queue, part 2 (cost+video already done). Serial. Streaming conditions use a 68-unit subset (cost budget).
cd <repo>
export OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1
R=results/cierre_oe1_oe3_20261031/3b
PY=.venv/Scripts/python.exe
$PY scripts/cierre3b/asr_compare.py --out $R/asr/pred --conds off_p0 off_p1 > $R/logs/asr_off.log 2>&1
$PY scripts/cierre3b/asr_compare.py --out $R/asr/pred --subset-ids $R/asr/subset_stream.json --conds str_p0_t2 str_p0_t2_ca str_p1_t2 str_p1_t2_ca > $R/logs/asr_str.log 2>&1
$PY scripts/cierre3b/asr_compare.py --out $R/asr/pred --subset-ids $R/asr/subset_t05.json --conds str_p0_t0.5 str_p0_t0.5_ca > $R/logs/asr_t05.log 2>&1
$PY scripts/cierre3b/live_cfg_compare.py --out $R/live > $R/logs/live.log 2>&1
echo QUEUE_DONE > $R/logs/queue_done.txt
