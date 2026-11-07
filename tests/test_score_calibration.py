import json

import numpy as np
import pytest

from mrcd import score_calibration as SC
from mrcd.calibration import ABSENT, PRESENT, UNCERTAIN, fit_calibrator


def _classes(band=0.1):
    rng = np.random.default_rng(0)
    s = rng.uniform(0, 1, 400)
    y = (rng.uniform(0, 1, 400) < s**2).astype(int)
    return {
        "filler_word": {
            "calibrator": fit_calibrator(s, y, "platt"),
            "threshold": 0.3,
            "band": band,
        }
    }


def test_roundtrip_and_flags(tmp_path):
    SC.save(tmp_path, _classes(), {"protocol_sha256": "x"})
    cal = SC.load(tmp_path)
    assert cal.default is False and cal.status == "exploratory"
    e = cal.estimate("filler_word", 0.9)
    assert 0.0 <= e["estimate"] <= 1.0 and "development data" in e["note"]
    assert "probability" in e["note"] and "not a validated" in e["note"]


def test_three_states_with_band(tmp_path):
    SC.save(tmp_path, _classes(band=0.1), {})
    pol = SC.load(tmp_path).classes["filler_word"]
    lo, hi = pol.threshold - pol.band, pol.threshold + pol.band
    ps = {PRESENT: 1.0, ABSENT: 0.0}
    assert pol.state(1.0) == PRESENT and pol.state(0.0) == ABSENT
    # a score whose estimate falls inside the band is an abstention, never a negative
    grid = np.linspace(0, 1, 2001)
    inside = [s for s in grid if lo <= pol.estimate(s) < hi]
    assert inside and all(pol.state(s) == UNCERTAIN for s in inside)
    assert ps


def test_no_band_means_no_abstention(tmp_path):
    SC.save(tmp_path, _classes(band=None), {})
    pol = SC.load(tmp_path).classes["filler_word"]
    assert {pol.state(s) for s in np.linspace(0, 1, 101)} <= {PRESENT, ABSENT}


def test_tamper_detected(tmp_path):
    SC.save(tmp_path, _classes(), {})
    p = tmp_path / SC.CALIBRATORS_FILE
    d = json.loads(p.read_text("utf8"))
    d["filler_word"]["threshold"] = 0.99
    p.write_text(json.dumps(d), "utf8")
    with pytest.raises(SC.ManifestError):
        SC.load(tmp_path)


def test_default_true_rejected(tmp_path):
    SC.save(tmp_path, _classes(), {})
    m = tmp_path / SC.MANIFEST_FILE
    d = json.loads(m.read_text("utf8"))
    d["default"] = True
    m.write_text(json.dumps(d), "utf8")
    with pytest.raises(SC.ManifestError):
        SC.load(tmp_path)
