import json

import numpy as np
import pytest

from mrcd.optimize import product_pipeline as PP
from mrcd.optimize import search as SE
from mrcd.optimize import space as SP
from tests.o_campaign_synth import make_data


@pytest.mark.parametrize("cls", ["filler_word", "prolongation"])
def test_record_path_equals_matrix_path(cls):
    d = make_data(cls, seed=7)
    tr = np.flatnonzero(d.outer != 0)
    te = np.flatnonzero(d.outer == 0)
    sel = SE.search(d, tr, 3, budget=3, patience=3)["best"]
    pipe = PP.fit_class_pipeline(d, tr, sel, "synthetic")
    m, X = SE.fit_config(d, tr, sel["config"])
    ref = m.score(X[te])
    out = pipe.predict([d.store.recs[i] for i in te])
    assert np.allclose(out["score"], ref, atol=1e-9)
    assert set(out["state"]) <= {"present", "absent", "uncertain"}
    assert out["decision"].shape == (len(te),)


def test_save_load_verifies_hash(tmp_path):
    d = make_data(seed=8)
    tr = np.arange(len(d.y))
    sel = SE.search(d, tr, 3, budget=2, patience=2)["best"]
    pipe = PP.fit_class_pipeline(d, tr, sel, "synthetic")
    man = PP.save_pipelines({"filler_word": pipe}, tmp_path, {"note": "test"})
    assert man["default"] is False and man["calibration_claim"] == "none"
    loaded = PP.load_pipelines(tmp_path)["filler_word"]
    recs = d.store.recs[:5]
    assert np.allclose(loaded.predict(recs)["score"], pipe.predict(recs)["score"])
    f = tmp_path / "o_pipeline_filler_word.pkl"
    f.write_bytes(f.read_bytes() + b"x")
    with pytest.raises(ValueError, match="sha256"):
        PP.load_pipelines(tmp_path)


def test_manifest_records_space_hash(tmp_path):
    d = make_data(seed=9)
    sel = SE.search(d, np.arange(len(d.y)), 3, budget=2, patience=2)["best"]
    PP.save_pipelines({"filler_word": PP.fit_class_pipeline(d, np.arange(len(d.y)), sel, "x")}, tmp_path, {})
    assert json.loads((tmp_path / "MANIFEST.json").read_text())["space_sha256"] == SP.space_hash()


def test_fold5_loader_is_only_referenced_by_the_final_script():
    import subprocess
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    hits = subprocess.run(["rg", "-l", "load_fold5_historical", "mrcd", "scripts/o_campaign", "tests"], cwd=root,
                          capture_output=True, text=True).stdout.split()
    allowed = {"mrcd/optimize/data.py", "scripts/o_campaign/run_fold5_historical.py", "tests/test_o_campaign_pipeline.py"}
    assert {h.replace("\\", "/") for h in hits} <= allowed
