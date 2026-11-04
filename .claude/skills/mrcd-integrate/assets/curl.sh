#!/usr/bin/env sh
# Submit a WAV and print the finished job. Needs a running service. Usage: sh curl.sh file.wav [url]
# Exits 1 on an HTTP error or when the job ends with "status":"failed".
set -eu
WAV="${1:?usage: curl.sh file.wav [url]}"; URL="${2:-http://127.0.0.1:8765}"
JOB=$(curl -sf -X POST --data-binary @"$WAV" -H "Content-Type: audio/wav" "$URL/v1/jobs?include_text=false") \
  || { echo "HTTP error on submit" >&2; exit 1; }
ID=$(printf '%s' "$JOB" | sed -n 's/.*"job_id": *"\([0-9a-f]*\)".*/\1/p')
while :; do
  STATE=$(curl -sf "$URL/v1/jobs/$ID") || { echo "HTTP error polling job $ID (unknown or evicted?)" >&2; exit 1; }
  case "$STATE" in *'"status":"completed"'*|*'"status":"failed"'*) break ;; esac
  sleep 1
done
printf '%s\n' "$STATE"
case "$STATE" in *'"status":"failed"'*) exit 1 ;; esac
