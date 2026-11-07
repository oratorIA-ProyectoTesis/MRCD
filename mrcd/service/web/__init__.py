"""Product page and synthetic sample shipped with the service.

``index.html`` is the single source of the product page: the service serves it at ``GET /`` (only when the
documentation is enabled), and ``scripts/build_site.py`` copies it, with the sample, into ``site/`` for GitHub Pages.
The page detects where it runs: served by the service it analyses audio against the same origin; anywhere else it
shows a real precomputed result of the sample and can point at a local service.

``sample_es.wav`` is synthetic speech (espeak-ng, voice ``es-419``), 16 kHz mono PCM16; see ``examples/README.md``.
"""

from __future__ import annotations

from pathlib import Path

WEB_DIR = Path(__file__).resolve().parent
INDEX_HTML = WEB_DIR / "index.html"
SAMPLE_WAV = WEB_DIR / "sample_es.wav"
#: sha256 of ``sample_es.wav`` (also recorded in the page's precomputed result as ``request.audio_sha256``)
SAMPLE_WAV_SHA256 = "357cf2696b539210a856b1f2b046efe3220e1bf6a792b4ddb7fc689b6e2d1cb3"

#: Content-Security-Policy of the page when the service serves it: everything inline or same origin, no third party.
PAGE_CSP = (
    "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src 'self' data:; "
    "media-src 'self' blob:; connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
)
