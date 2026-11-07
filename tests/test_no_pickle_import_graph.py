"""The "no pickle" control of the service decision path, as an import-graph check (REV-INT finding 2).

Replaces the text search: walks every first-party module reachable from the decision-profile loader (parent
packages included) and fails on any import of a deserialising module or a ``torch.load``-style call. A runtime
check in a fresh interpreter covers the third-party side (``dill``/``shelve`` must not be loaded, ``joblib``/``cloudpickle`` only as a
dependency of scikit-learn; plain ``pickle`` cannot be asserted because numpy itself imports it).
"""

import subprocess
import sys
from pathlib import Path

import pytest

from tests.import_graph import transitive_imports

ROOT = Path(__file__).resolve().parents[1]
BANNED = {
    "pickle",
    "_pickle",
    "cPickle",
    "cloudpickle",
    "dill",
    "joblib",
    "shelve",
    "marshal",
}
BANNED_CALLS = {
    "pickle.load",
    "pickle.loads",
    "joblib.load",
    "dill.load",
    "cloudpickle.load",
}
# The engine (reached through the parent package ``mrcd``) loads its OWN checkpoint with torch.load; that is the only
# deserialisation left on the path and it is listed here so a new call site fails the test. Residual risk in the report.
TORCH_LOAD_ALLOWED = {"mrcd.inference", "mrcd.adaptation"}
ENTRY_POINTS = ("mrcd.service.decision_profile", "mrcd.optimize.safe_pipeline")


def banned_in(root: Path, start: str):
    seen, names, calls = transitive_imports(root, start)
    bad = sorted(n for n in names if n.split(".")[0] in BANNED or n == "__import__")
    bad_calls = sorted(c for c in calls if c in BANNED_CALLS)
    return seen, bad, bad_calls


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_decision_path_never_imports_a_deserialiser_transitively(entry):
    seen, bad, bad_calls = banned_in(ROOT, entry)
    assert (
        entry in seen and "mrcd.optimize.class_pipeline" in seen
    )  # the walk really crosses modules
    assert bad == [] and bad_calls == []
    assert (
        "mrcd.optimize.product_pipeline" not in seen
    )  # the module that can pickle.loads is NOT reachable


def test_torch_load_call_sites_on_the_path_are_exactly_the_allowlisted_ones():
    from tests.import_graph import _imports, _module_file

    seen, _, _ = transitive_imports(ROOT, "mrcd.service.decision_profile")
    with_torch_load = {
        m for m in seen if "torch.load" in _imports(_module_file(ROOT, m), m)[1]
    }
    assert with_torch_load <= TORCH_LOAD_ALLOWED


def test_the_check_detects_a_transitive_pickle_import(tmp_path):
    pkg = tmp_path / "mrcd"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "top.py").write_text("from mrcd import middle\n")
    (pkg / "middle.py").write_text("from . import leaf\n")
    (pkg / "leaf.py").write_text("import joblib\nx = pickle.load('f')\n")
    seen, bad, bad_calls = banned_in(tmp_path, "mrcd.top")
    assert {"mrcd.top", "mrcd.middle", "mrcd.leaf"} <= seen
    assert bad == ["joblib"] and bad_calls == ["pickle.load"]


def test_a_text_search_would_have_missed_it(tmp_path):
    pkg = tmp_path / "mrcd"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "top.py").write_text("from mrcd import leaf\n")
    (pkg / "leaf.py").write_text("import pickle as p\n")
    assert "import pickle" not in (pkg / "top.py").read_text()
    assert banned_in(tmp_path, "mrcd.top")[1] == ["pickle"]


def test_runtime_import_does_not_load_third_party_deserialisers():
    code = (
        "import sys, mrcd.service.decision_profile\n"
        "bad = sorted(m for m in ('dill','shelve') if m in sys.modules)\n"
        "bad += [m for m in ('joblib','cloudpickle') if m in sys.modules and 'sklearn' not in sys.modules]\n"
        "print(','.join(bad))"
    )
    out = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            f"import sys; sys.path.insert(0, r'{ROOT}'); exec({code!r})",
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=120,
    )
    assert out.returncode == 0, out.stderr[-500:]
    assert out.stdout.strip() == ""
