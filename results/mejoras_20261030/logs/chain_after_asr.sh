cd /c/zArnes-tesis/mrcd
PY=.venv/Scripts/python.exe
L=results/mejoras_20261030/logs
while [ $(ls results/mejoras_20261030/asr_cache/dihana/*.json 2>/dev/null | grep -vc manifest) -lt 6279 ]; do sleep 15; done
sleep 20
pkill -f "asr_cache.py dihana" 2>/dev/null
$PY scripts/improve/asr_cache.py dihana manifest >> $L/chain_dihana.log 2>&1
$PY scripts/improve/asr_cache.py epic manifest >> $L/chain_dihana.log 2>&1
for m in zero noise; do for i in 0 1 2; do $PY scripts/improve/run_features.py dihana $m $i 3 1 >> $L/feat_dihana_${m}_$i.log 2>&1 & done; done
$PY scripts/improve/asr_audit.py 60 12 12 >> $L/asr_audit.log 2>&1 &
wait
echo chain-after-asr-done >> $L/chain_dihana.log
