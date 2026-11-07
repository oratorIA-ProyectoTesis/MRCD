"""Full-page screenshots of the product page at 1280 px and 400 px (Playwright, local Chrome).

    python scripts/dx/capture_page.py URL OUT_PREFIX [--analyze]

``--analyze`` clicks "Analizar la muestra" first and waits for the result (page served by the service).
Opening ``site/index.html`` through a static server shows the Pages mode (precomputed result).
"""

from __future__ import annotations

import argparse
import sys

from playwright.sync_api import sync_playwright


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("out_prefix")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--timeout-s", type=float, default=240)
    args = ap.parse_args(argv)
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(channel="chrome")
        except Exception:  # noqa: BLE001 - fall back to the bundled Chromium
            browser = p.chromium.launch()
        for width, height in ((1280, 900), (400, 860)):
            page = browser.new_page(
                viewport={"width": width, "height": height}, device_scale_factor=1
            )
            errors: list[str] = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(args.url, wait_until="networkidle")
            page.wait_for_timeout(800)
            if args.analyze:
                page.click("#btn-sample")
                page.wait_for_selector(
                    "#result .timeline, #result .note.err",
                    timeout=args.timeout_s * 1000,
                )
                page.wait_for_timeout(300)
            out = f"{args.out_prefix}_{width}.png"
            page.screenshot(path=out, full_page=True)
            print(
                out,
                "status:",
                page.inner_text("#status") or "-",
                "| mode:",
                page.inner_text("#mode-text")[:90].replace("\n", " "),
            )
            if errors:
                print("page errors:", errors)
                return 1
            page.close()
        browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
