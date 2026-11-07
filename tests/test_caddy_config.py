"""The reverse-proxy file is the only TLS story; guard the properties the docs promise (Caddy itself is not run here)."""

import re
from pathlib import Path

from mrcd.service.server import DEFAULT_MAX_BODY_BYTES

CADDYFILE = (Path(__file__).resolve().parents[1] / "deploy" / "caddy" / "Caddyfile").read_text("utf8")


def test_proxies_to_loopback_only_and_keeps_sse_unbuffered():
    assert "reverse_proxy 127.0.0.1:8765" in CADDYFILE
    assert "flush_interval -1" in CADDYFILE


def test_credentials_are_deleted_from_the_access_log():
    assert "request>headers>Authorization delete" in CADDYFILE
    assert "request>headers>X-Session-Token delete" in CADDYFILE


def test_no_plaintext_http_site_and_hsts_is_set():
    assert "http://" not in CADDYFILE.replace("# ", "")
    assert "Strict-Transport-Security" in CADDYFILE


def test_no_secret_value_is_committed():
    assert "MRCD_SERVICE_TOKEN=<" in CADDYFILE  # placeholder only


# go-humanize (what Caddy uses): MB/KB are SI (1000^n), MiB/KiB are binary (1024^n).
_UNITS = {"b": 1, "kb": 10**3, "mb": 10**6, "gb": 10**9, "kib": 2**10, "mib": 2**20, "gib": 2**30}


def test_caddy_body_cap_equals_the_service_cap():
    m = re.search(r"^\s*max_size\s+(\d+)\s*([A-Za-z]*)\s*$", CADDYFILE, re.M)
    assert m, "request_body max_size not found"
    size = int(m.group(1)) * _UNITS[(m.group(2) or "b").lower()]
    assert size == DEFAULT_MAX_BODY_BYTES


def test_clickjacking_and_referrer_headers_are_set():
    assert 'X-Frame-Options "DENY"' in CADDYFILE
    assert "frame-ancestors 'none'" in CADDYFILE
    assert 'Referrer-Policy "no-referrer"' in CADDYFILE


def test_server_timeouts_are_set_but_write_timeout_does_not_cut_sse():
    block = re.search(r"timeouts\s*\{([^}]*)\}", CADDYFILE)
    assert block, "timeouts block not found"
    assert "read_header" in block.group(1) and "read_body" in block.group(1)
    assert not re.search(r"^\s*write", block.group(1), re.M)
