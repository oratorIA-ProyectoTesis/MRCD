#!/usr/bin/env bash
# Compile-check every .mmd in docs/architecture and docs/contracts with a LOCAL Mermaid CLI (no web service).
# Usage (repo root, Git Bash): bash docs/architecture/render_mermaid_check.sh [OUT_DIR]
# Needs: Node >= 18, a local Chrome/Chromium (CHROME env var), and a one-time install of the CLI:
#   mkdir -p "$TOOLS" && cd "$TOOLS" && PUPPETEER_SKIP_DOWNLOAD=1 npm i --prefix . @mermaid-js/mermaid-cli@11.4.2
# TOOLS defaults to ~/.cache/mrcd-mmdc (outside the repo). Exit code != 0 means a diagram failed to compile.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TOOLS="${TOOLS:-$HOME/.cache/mrcd-mmdc}"
CH="${CHROME:-C:/Program Files/Google/Chrome/Application/chrome.exe}"
OUT="${1:-$(mktemp -d)}"
CFG="$OUT/puppeteer.json"
mkdir -p "$OUT"
printf '{"executablePath":"%s","args":["--no-sandbox","--disable-gpu"]}' "$CH" > "$CFG"
fail=0
for f in "$ROOT"/docs/architecture/*.mmd "$ROOT"/docs/contracts/*.mmd; do
  b="$(basename "$f" .mmd)"
  if "$TOOLS/node_modules/.bin/mmdc" -p "$CFG" -i "$f" -o "$OUT/$b.svg" >"$OUT/$b.log" 2>&1; then
    echo "OK    $b"
  else
    echo "FAIL  $b (see $OUT/$b.log)"; fail=1
  fi
done
exit $fail
