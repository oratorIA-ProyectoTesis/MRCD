#!/usr/bin/env bash
# Optional: rasterise the SVGs with the locally installed Chrome (headless). Not needed to read the diagrams.
# Usage (repo root, Git Bash): bash docs/architecture/render_png.sh
set -euo pipefail
CH="${CHROME:-/c/Program Files/Google/Chrome/Application/chrome.exe}"
D="$(cd "$(dirname "$0")" && pwd -W 2>/dev/null || pwd)"
for f in "$D"/c4_*.svg; do
  w=$(sed -n 's/.*width="\([0-9]*\)" height="\([0-9]*\)".*/\1/p' "$f" | head -1)
  h=$(sed -n 's/.*width="\([0-9]*\)" height="\([0-9]*\)".*/\2/p' "$f" | head -1)
  "$CH" --headless=new --disable-gpu --hide-scrollbars --window-size="$w","$h" --screenshot="${f%.svg}.png" "file:///$f" >/dev/null 2>&1
  echo "rendered ${f%.svg}.png (${w}x${h})"
done
