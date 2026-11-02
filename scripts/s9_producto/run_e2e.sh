#!/usr/bin/env bash
# usage: run_e2e.sh <outDir-name> [mode: normal|pause|abrupt] [recSeconds]
# The app must be running on :5173 (incremental mode, push on) and MRCD on :8765.
cd "$(dirname "$0")/../.."
PW="<USER_HOME>/AppData/Roaming/npm/node_modules/@playwright/cli/node_modules/playwright-core"
CH="<USER_HOME>/AppData/Local/ms-playwright/chromium-1228/chrome-win64/chrome.exe"
WAV="$PWD/results/mejora_objetivos_20261101/S7_app/inputs/e2e_synthetic_es.wav"
exec node scripts/s9_producto/e2e_session.mjs "$PW" "$CH" "$WAV" "results/mejora_objetivos_20261101/S9_producto/$1" http://localhost:5173 "${3:-47}" "${2:-normal}" with
