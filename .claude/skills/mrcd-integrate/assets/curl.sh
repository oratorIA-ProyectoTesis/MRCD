#!/usr/bin/env sh
# Submit a WAV and print the finished job. Needs a running service. Usage: sh curl.sh file.wav [url]
set -eu
WAV="${1:?usage: curl.sh file.wav [url]}"; URL="${2:-http://127.0.0.1:8765}"
JOB=$(curl -sf -X POST --data-binary @"$WAV" -H "Content-Type: audio/wav" "$URL/v1/jobs?include_text=false")
ID=$(printf '%s' "$JOB" | sed -n 's/.*"job_id": *"\([0-9a-f]*\)".*/\1/p')
while :; do
  STATE=$(curl -sf "$URL/v1/jobs/$ID")
  case "$STATE" in *'"status":"completed"'*|*'"status":"failed"'*) break ;; esac
  sleep 1
done
printf '%s\n' "$STATE"
