"""Build the GitHub Pages copy of the product page from its single source.

Source of truth: ``mrcd/service/web/index.html`` (served by the service at ``GET /``). This script

1. embeds ``docs/architecture/c4_1_context.svg`` between the ``C4:BEGIN`` / ``C4:END`` markers,
2. embeds the precomputed result ``examples/sample_result.json`` (a real service answer for the sample WAV) in the
   ``<script id="sample-result">`` block,
3. writes the same page, the sample WAV and ``.nojekyll`` into ``site/``.

Steps 1-2 rewrite the package page in place, so the page served by the service and the Pages copy are byte-identical.
``--check`` changes nothing and exits 1 when anything is out of date (used by the tests).

    python scripts/build_site.py            # rebuild
    python scripts/build_site.py --check    # verify
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "mrcd" / "service" / "web" / "index.html"
SAMPLE = ROOT / "mrcd" / "service" / "web" / "sample_es.wav"
C4 = ROOT / "docs" / "architecture" / "c4_1_context.svg"
RESULT = ROOT / "examples" / "sample_result.json"
SITE = ROOT / "site"

_C4 = re.compile(r"(<!-- C4:BEGIN -->\r?\n).*?(<!-- C4:END -->)", re.S)
_RESULT = re.compile(
    r'(<script type="application/json" id="sample-result">).*?(</script>)', re.S
)


def render_page(page: str, svg: str, result: dict | None) -> str:
    svg = svg.strip()
    if svg.startswith("<?xml"):
        svg = svg.split("?>", 1)[1].strip()
    page, n1 = _C4.subn(lambda m: m.group(1) + svg + "\n" + m.group(2), page)
    payload = (
        "null"
        if result is None
        else json.dumps(result, ensure_ascii=False, separators=(",", ":"))
    )
    payload = payload.replace(
        "</", "<\\/"
    )  # never close the script element from inside the JSON
    page, n2 = _RESULT.subn(lambda m: m.group(1) + payload + m.group(2), page)
    if n1 != 1 or n2 != 1:
        raise SystemExit(
            f"markers not found exactly once in {PAGE} (C4: {n1}, result: {n2})"
        )
    return page


def expected() -> dict[Path, bytes]:
    result = json.loads(RESULT.read_text(encoding="utf-8")) if RESULT.exists() else None
    page = render_page(
        PAGE.read_text(encoding="utf-8"), C4.read_text(encoding="utf-8"), result
    ).encode("utf-8")
    return {
        PAGE: page,
        SITE / "index.html": page,
        SITE / "sample_es.wav": SAMPLE.read_bytes(),
        SITE / ".nojekyll": b"",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--check", action="store_true", help="verify only; exit 1 when out of date"
    )
    args = ap.parse_args(argv)
    stale = [
        p for p, data in expected().items() if not p.exists() or p.read_bytes() != data
    ]
    if args.check:
        for p in stale:
            print(f"out of date: {p.relative_to(ROOT)}")
        return 1 if stale else 0
    for p, data in expected().items():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    print(
        f"site built: {', '.join(str(p.relative_to(ROOT)) for p in stale) or 'already up to date'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
