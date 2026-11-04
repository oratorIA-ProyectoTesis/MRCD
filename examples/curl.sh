#!/usr/bin/env sh
# Analyse a WAV with curl only. Needs a running service (Docker image or `python -m mrcd.service serve`).
#   sh examples/curl.sh recording.wav [http://127.0.0.1:8765]
# The WAV must be mono, 16 kHz, PCM16, at least 13 s long. No WAV at hand:
#   python scripts/docker_smoke.py --write-wav synthetic.wav
set -eu
WAV="${1:?usage: curl.sh file.wav [url]}"
URL="${2:-http://127.0.0.1:8765}"

curl -sf "$URL/v1/health"; echo
JOB=$(curl -sf -X POST --data-binary @"$WAV" -H "Content-Type: audio/wav" "$URL/v1/jobs?include_text=false")
ID=$(printf '%s' "$JOB" | sed -n 's/.*"job_id": *"\([0-9a-f]*\)".*/\1/p')
echo "job $ID"
while :; do
  STATE=$(curl -sf "$URL/v1/jobs/$ID")
  case "$STATE" in
    *'"status":"completed"'*|*'"status":"failed"'*) break ;;
  esac
  sleep 1
done
printf '%s\n' "$STATE" > result.json
echo "result written to result.json"
