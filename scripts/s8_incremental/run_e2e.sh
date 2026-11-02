#!/usr/bin/env bash
# usage: run_e2e.sh <outDir-name> [mode: normal|abrupt]   (app already running on :5173 with the desired ORATORIA_MRCD_MODE)
cd "$(dirname "$0")/../.."
PW="<USER_HOME>/AppData/Roaming/npm/node_modules/@playwright/cli/node_modules/playwright-core"
CH="<USER_HOME>/AppData/Local/ms-playwright/chromium-1228/chrome-win64/chrome.exe"
WAV="$PWD/results/mejora_objetivos_20261101/S7_app/inputs/e2e_synthetic_es.wav"
exec node scripts/s8_incremental/e2e_session.mjs "$PW" "$CH" "$WAV" "results/mejora_objetivos_20261101/S8_incremental/$1" http://localhost:5173 47 "${2:-normal}" with
