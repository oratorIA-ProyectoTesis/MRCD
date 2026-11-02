import hashlib
import json
import re
from pathlib import Path

from mrcd.optimize import space as SP

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "mejora_objetivos_20261101" / "O_campaign"


def test_protocol_is_frozen_and_matches_the_space():
    fz = json.loads((OUT / "PROTOCOL.sha256.json").read_text("utf8"))
    body = (OUT / "PROTOCOL.md").read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(body).hexdigest() == fz["sha256"]
    assert fz["space_sha256"] == SP.space_hash()
    assert fz["budget"] == SP.BUDGET and fz["patience"] == SP.PATIENCE
    text = body.decode("utf8")
    for c, v in SP.describe().items():
        row = re.search(rf"\|\s*{c}\s*\|.*", text).group(0)
        assert str(v["size"]) in row
    assert "60 configurations" in text and "20 consecutive" in text


def test_heavy_compute_is_gated_on_s8_readme():
    src = (ROOT / "scripts" / "o_campaign" / "run_campaign.py").read_text("utf8")
    assert "gate()" in src
    assert "README_S8.md" in (ROOT / "scripts" / "o_campaign" / "common.py").read_text(
        "utf8"
    )
