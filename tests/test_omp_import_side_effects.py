"""Importing the library must not change a consuming application's global configuration.

Only entry-point scripts pin OMP_NUM_THREADS (scripts/_pin_omp.py, documented there and in
mrcd/reproducibility.py); `import mrcd` and the live modules only RECORD the environment.
"""

from __future__ import annotations

import re

from tests.test_omp_pin_subprocess import REPO, run

SNAPSHOT = (
    "import json, os, sys, warnings, logging\n"
    "def snap():\n"
    "    return {'env': dict(os.environ), 'filters': [repr(f) for f in warnings.filters],\n"
    "            'root_handlers': len(logging.getLogger().handlers), 'root_level': logging.getLogger().level,\n"
    "            'path': list(sys.path), 'cwd': os.getcwd(), 'recursion': sys.getrecursionlimit(),\n"
    "            'switch': sys.getswitchinterval()}\n"
    "import numpy as np\n"
    "nperr = np.geterr(); nppr = np.get_printoptions()\n"
    "import torch; threads = torch.get_num_threads()\n"
    "before = snap()\n"
    "import mrcd, mrcd.engine, mrcd.live.session, mrcd.live.streaming_asr, mrcd.live.mic_stream\n"
    "after = snap()\n"
    "diff = {k: (before[k], after[k]) for k in before if before[k] != after[k]}\n"
    "print(json.dumps({'diff_keys': sorted(diff), 'env_diff': {k: v for k, v in diff.get('env', ({}, {}))[1].items()\n"
    "                  if diff.get('env', ({}, {}))[0].get(k) != v},\n"
    "                  'np_same': nperr == np.geterr() and nppr == np.get_printoptions(),\n"
    "                  'torch_threads_same': threads == torch.get_num_threads()}))\n"
)


def test_import_mrcd_does_not_touch_global_configuration_when_omp_unset():
    out = run(SNAPSHOT, omp=None)
    assert out["env_diff"] == {}, (
        "importing mrcd must not set environment variables (OMP stays unset)"
    )
    assert out["diff_keys"] == []
    assert out["np_same"] and out["torch_threads_same"]


def test_import_mrcd_keeps_a_callers_omp_value():
    out = run(SNAPSHOT, omp="7")
    assert out["env_diff"] == {} and out["diff_keys"] == []


def test_opt_out_variable_is_respected_by_the_library_too():
    out = run(SNAPSHOT, omp=None, extra_env={"MRCD_NO_OMP_PIN": "1"})
    assert out["env_diff"] == {} and out["diff_keys"] == []


def test_no_library_module_pins_the_environment():
    """Only scripts/ entry points may import _pin_omp or assign OMP_NUM_THREADS; mrcd/ may only read it."""
    bad = []
    for f in (REPO / "mrcd").rglob("*.py"):
        text = f.read_text("utf8")
        if (
            "_pin_omp" in text
            or re.search(r"environ\[[\"']OMP_NUM_THREADS[\"']\]\s*=[^=]", text)
            or "putenv" in text
        ):
            bad.append(str(f.relative_to(REPO)))
    # reproducibility.pin_asr_threads is an explicit, opt-in helper that writes only when the CALLER invokes it
    assert [b.replace("\\", "/") for b in bad] in ([], ["mrcd/reproducibility.py"]), bad
