#!/usr/bin/env bash
# Fault injection: the MRCD service is killed ~15 s into the recording and started again (same flags).
# The app must finish by falling back to the offline path with its complete buffer.
# usage: run_e2e_fault.sh <outDir-name>   (app running with ORATORIA_MRCD_MODE=incremental)
cd "$(dirname "$0")/../.."
OUT="results/mejora_objetivos_20261101/S8_incremental/logs"
scripts/s8_incremental/run_e2e.sh "$1" > "$OUT/$1.out" 2>&1 &
E2E=$!
until [ "$(curl -s http://127.0.0.1:8765/v1/health | python -c 'import sys,json; print(json.load(sys.stdin).get("active_sessions",0))' 2>/dev/null)" -ge 1 ] 2>/dev/null; do sleep 1; done
sleep 15
PID=$(netstat -ano | grep -E "127.0.0.1:8765 .*LISTENING" | awk '{print $5}' | head -1)
echo "killing MRCD pid $PID at $(date +%T)" | tee -a "$OUT/$1.fault.txt"
taskkill //PID "$PID" //F > /dev/null 2>&1
sleep 2
nohup scripts/s8_incremental/start_mrcd.sh "$OUT/mrcd_service.log" > /dev/null 2>&1 &
echo "MRCD restarted at $(date +%T)" | tee -a "$OUT/$1.fault.txt"
wait $E2E
