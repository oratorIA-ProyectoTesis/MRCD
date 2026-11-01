cd /c/zArnes-tesis/mrcd
PY=.venv/Scripts/python.exe
L=results/mejoras_20261030/logs
for c in ameresco epic; do
  n=$([ $c = ameresco ] && echo 98 || echo 136)
  (
    while [ $(ls results/mejoras_20261030/asr_cache/$c/*.json 2>/dev/null | grep -vc manifest) -lt $n ]; do sleep 20; done
    sleep 5
    $PY scripts/improve/asr_cache.py $c manifest >> $L/chain_$c.log 2>&1
    [ $c = ameresco ] && $PY scripts/improve/verify_ameresco_cache.py >> $L/chain_$c.log 2>&1
    $PY scripts/improve/run_features.py $c zero 0 1 1 >> $L/chain_$c.log 2>&1
    echo chain-done >> $L/chain_$c.log
  ) &
done
wait
