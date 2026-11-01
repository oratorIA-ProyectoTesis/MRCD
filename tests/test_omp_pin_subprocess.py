"""OMP_NUM_THREADS handling verified in FRESH interpreters (the order of imports is the whole point)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def run(code: str, *, omp: str | None, extra_env: dict | None = None) -> dict:
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("OMP_NUM_THREADS", "MRCD_NO_OMP_PIN")
    }
    if omp is not None:
        env["OMP_NUM_THREADS"] = omp
    env.update(extra_env or {})
    env["PYTHONPATH"] = str(REPO)
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


STATUS = (
    "import json, os\n"
    "{pre}\n"
    "import mrcd\n"
    "{post}\n"
    "from mrcd.reproducibility import omp_status, omp_warning\n"
    "s = omp_status(); print(json.dumps({{'status': s, 'warning': omp_warning(s)}}))\n"
)


def status(pre="", post="", **kw):
    return run(STATUS.format(pre=pre, post=post), **kw)


def test_unset_in_a_fresh_process_is_reported_and_warned():
    r = status(omp=None)
    assert (
        r["status"]["set_before_numpy"] is False and r["status"]["effective"] is False
    )
    assert r["warning"] and "not set" in r["warning"]


def test_variable_in_the_environment_is_effective_and_silent():
    r = status(omp="4")
    assert r["status"]["set_before_numpy"] is True and r["status"]["effective"] is True
    assert r["warning"] is None


def test_assigning_after_import_mrcd_is_flagged_as_ineffective():
    """The exact mistake that produced the invalid ASR caches: os.environ shows 4 but it is too late."""
    r = status(omp=None, post="os.environ['OMP_NUM_THREADS'] = '4'")
    st = r["status"]
    assert (
        st["omp_num_threads"] == "4"
        and st["set_before_numpy"] is False
        and st["effective"] is False
    )
    assert st["assigned_after_import"] is True
    assert r["warning"] and "after numpy" in r["warning"]


def test_value_other_than_4_is_recorded_not_pinned_and_warned():
    r = status(omp="8")
    assert r["status"]["omp_num_threads"] == "8" and r["status"]["effective"] is False
    assert r["warning"] and "differs from the pinned value" in r["warning"]


def test_numpy_imported_before_mrcd_is_unverifiable():
    r = status(omp="4", pre="import numpy")
    assert r["status"]["numpy_imported_before_mrcd"] is True
    assert r["status"]["set_before_numpy"] is None and r["status"]["effective"] is None


# ------------------------------------------------------------------ entry-point helper (scripts/_pin_omp.py)
PIN = (
    "import json, os, sys\n"
    "sys.path.insert(0, 'scripts')\n"
    "{pre}\n"
    "import warnings\n"
    "with warnings.catch_warnings(record=True) as w:\n"
    "    warnings.simplefilter('always')\n"
    "    import _pin_omp\n"
    "import mrcd\n"
    "from mrcd.reproducibility import omp_status\n"
    "print(json.dumps({{'report': _pin_omp.REPORT, 'env': os.environ.get('OMP_NUM_THREADS'),"
    " 'status': omp_status(), 'warned': [str(x.message) for x in w]}}))\n"
)


def pin(pre="", **kw):
    return run(PIN.format(pre=pre), **kw)


def test_entry_point_pin_before_numpy_makes_the_pin_effective():
    r = pin(omp=None)
    assert r["report"]["action"] == "set" and r["env"] == "4"
    assert r["status"]["effective"] is True and r["status"]["set_before_numpy"] is True


def test_entry_point_pin_keeps_the_callers_value():
    r = pin(omp="2")
    assert r["report"]["action"] == "kept" and r["env"] == "2"


def test_entry_point_pin_explicit_opt_out_leaves_the_environment_alone():
    r = pin(omp=None, extra_env={"MRCD_NO_OMP_PIN": "1"})
    assert r["report"]["action"] == "opted_out" and r["env"] is None
    assert r["status"]["effective"] is False


def test_entry_point_pin_opt_out_by_flag():
    r = pin(omp=None, pre="sys.argv.append('--no-omp-pin')")
    assert r["report"]["action"] == "opted_out" and r["env"] is None


def test_entry_point_pin_after_numpy_warns_and_does_not_pretend():
    r = pin(omp=None, pre="import numpy")
    assert r["report"]["action"] == "not_set_too_late" and r["env"] is None
    assert any("NOT affect" in m for m in r["warned"])


def test_live_and_diagnose_entry_points_pin_before_numpy_or_mrcd():
    for name in ("live_demo.py", "diagnose_session.py"):
        src = (REPO / "scripts" / name).read_text("utf8")
        pin_at = src.index("_pin_omp")
        for heavy in ("import numpy", "import torch", "from mrcd", "import mrcd"):
            first = src.find(heavy)
            assert first == -1 or pin_at < first, (
                f"{name}: '{heavy}' precedes the OMP pin"
            )
