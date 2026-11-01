"""Verify that the C4 model does not contradict the code (standard library only).

Run (repo root):  .venv/Scripts/python.exe docs/architecture/verify_architecture.py
Exit code 0 = every check passed; 1 = at least one failure (printed as FAIL).

Checks
------
1. Every `modules` path of every non-proposed node exists (file or directory).
2. Every `symbols` entry (class, function, constant or Class.method) exists in its file (AST).
3. Every edge with check="import" is backed by at least one real import (top-level or lazy, via AST) from a module
   of the source component to a module of the target component.
4. Negative claims drawn in the diagrams: `mrcd/decision.py` is imported by no core module (only research kits); the engine and the
   live path do not import the kinesic/face/sync extractors directly; `analyze()` rejects visual modes.
5. The app has no MRCD client (the app -> MRCD edge is PROPOSED); `mrcd/service` exists (commit 955d27c) and is drawn as current.
6. App-side paths (ORATORIA-PROYECTO, read-only) exist when that repo is present; otherwise they are skipped.
7. Informational: library modules that no component lists.
"""

from __future__ import annotations
import os

import ast
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import c4_model  # noqa: E402

APP_ROOT = Path(os.environ.get("MRCD_APP_ROOT", r"<APP_ROOT>"))
fails: list[str] = []
infos: list[str] = []
passed = 0


def ok(msg):
    global passed
    passed += 1
    print("ok   ", msg)


def fail(msg):
    fails.append(msg)
    print("FAIL ", msg)


def files_of(modules):
    out = set()
    for m in modules:
        p = ROOT / m
        if p.is_dir():
            out |= {f.relative_to(ROOT).as_posix() for f in p.rglob("*.py")}
        elif p.exists():
            out.add(m)
    return out


def imported_modules(relpath):
    """Dotted module names imported by a file, including `from pkg import sub` candidates."""
    tree = ast.parse((ROOT / relpath).read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
            names |= {f"{node.module}.{a.name}" for a in node.names}
    return names


def as_file(dotted):
    parts = dotted.replace(".", "/")
    for cand in (f"{parts}.py", f"{parts}/__init__.py"):
        if (ROOT / cand).exists():
            return cand
    return None


def has_symbol(relpath, dotted):
    tree = ast.parse((ROOT / relpath).read_text(encoding="utf-8"))
    head, *rest = dotted.split(".")
    for node in tree.body:
        names = []
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names = [node.name]
        elif isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names = [node.target.id]
        if head not in names:
            continue
        if not rest:
            return True
        if isinstance(node, ast.ClassDef):
            members = [
                n.name
                for n in node.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            members += [
                t.id
                for n in node.body
                if isinstance(n, ast.Assign)
                for t in n.targets
                if isinstance(t, ast.Name)
            ]
            members += [
                n.target.id
                for n in node.body
                if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)
            ]
            return rest[0] in members
    return False


head = subprocess.run(
    ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
).stdout.strip()
print(f"inspected tree: {ROOT}  HEAD={head}")

all_nodes = {}
for key, d in c4_model.DIAGRAMS.items():
    nodes = {n["id"]: n for n in d["nodes"]}
    all_nodes[key] = nodes
    for e in d["edges"]:
        for end in (e["a"], e["b"]):
            if end not in nodes:
                fail(f"{key}: edge references unknown node {end}")

# 1-2 modules and symbols
for key, nodes in all_nodes.items():
    for n in nodes.values():
        for m in n["modules"]:
            if n["kind"] == "proposed":
                fail(f"{key}:{n['id']} is proposed but lists modules {m}")
            elif (ROOT / m).exists():
                ok(f"{key}:{n['id']} module {m}")
            else:
                fail(f"{key}:{n['id']} module {m} does not exist")
        for path, sym in n["symbols"]:
            if not (ROOT / path).exists():
                fail(f"{key}:{n['id']} file {path} missing")
            elif has_symbol(path, sym):
                ok(f"{key}:{n['id']} symbol {path}::{sym}")
            else:
                fail(f"{key}:{n['id']} symbol {path}::{sym} not found")
        for ap in n["app_paths"]:
            if APP_ROOT.exists():
                (ok if (APP_ROOT / ap).exists() else fail)(
                    f"{key}:{n['id']} app path {ap}"
                )
            else:
                infos.append(f"app repo not present; skipped {ap}")

# 3 import edges
for key, d in c4_model.DIAGRAMS.items():
    nodes = all_nodes[key]
    for e in d["edges"]:
        if e["check"] != "import":
            continue
        src, dst = (
            files_of(nodes[e["a"]]["modules"]),
            files_of(nodes[e["b"]]["modules"]),
        )
        found = None
        for f in sorted(src):
            for name in imported_modules(f):
                target = as_file(name)
                if target and target in dst and target != f:
                    found = (f, name)
                    break
            if found:
                break
        label = f"{key}: {e['a']} -> {e['b']}"
        (ok if found else fail)(f"{label} import evidence {found if found else 'NONE'}")

# 4 negative claims
lib = sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "mrcd").rglob("*.py"))
users = [f for f in lib if "mrcd.decision" in imported_modules(f)]
core = [f for f in users if f in ("mrcd/engine.py", "mrcd/inference.py") or f.startswith(("mrcd/live/", "mrcd/service/", "mrcd/models/"))]
(ok if not core else fail)(f"mrcd/decision.py is NOT imported by the engine, inference, live path or service (importers in the library: {users})")
engine_live = ["mrcd/engine.py"] + [f for f in lib if f.startswith("mrcd/live/")]
bad = [f for f in engine_live if any(n.startswith(("mrcd.extractors.kinesic", "mrcd.extractors.face_tracking", "mrcd.sync")) for n in imported_modules(f))]
(ok if not bad else fail)(f"engine and live path do not import kinesic/face/sync directly (found: {bad})")
src = (ROOT / "mrcd/engine.py").read_text(encoding="utf-8")
(ok if 'if self.mode not in ("audio_only", "audio_text"):' in src else fail)("analyze() rejects visual modes (guard present)")
(ok if (ROOT / "models/ckpt_audio_text.pt").exists() and not any((ROOT / "models").glob("ckpt_trimodal*")) else fail)("only the audio_text checkpoint is present locally (no trimodal checkpoint)")
eng_importers = [f for f in lib if f.startswith("mrcd/") and not f.startswith("mrcd/service/") and any(n.startswith("mrcd.service") for n in imported_modules(f))]
(ok if not eng_importers else fail)(f"nothing in the library imports mrcd.service (found: {eng_importers})")

# 5 proposed elements: the app -> MRCD client must not exist
if APP_ROOT.exists():
    hits = [
        p.relative_to(APP_ROOT).as_posix()
        for d in ("backend/src", "frontend/src", "contracts")
        if (APP_ROOT / d).exists()
        for p in (APP_ROOT / d).rglob("*")
        if p.is_file() and p.suffix in {".py", ".ts", ".tsx", ".json"} and "mrcd" in p.read_text(encoding="utf-8", errors="ignore").lower()
    ]
    (ok if not hits else fail)(f"the app contains no reference to MRCD (the app->MRCD edge is correctly drawn as PROPOSED); hits: {hits[:5]}")
else:
    infos.append("app repo not present; could not verify that the app has no MRCD client")
for key, nodes in all_nodes.items():
    for n in nodes.values():
        if n["kind"] == "proposed":
            fail(f"{key}:{n['id']} is drawn as proposed; no proposed NODE should remain once mrcd/service exists")

# 7 informational coverage
listed = set()
for n in all_nodes["c4_3_components"].values():
    listed |= files_of(n["modules"])
unlisted = [f for f in lib if f not in listed and not f.endswith("__init__.py")]
infos.append(
    f"library modules not listed in any component ({len(unlisted)}): "
    + ", ".join(unlisted)
)

print()
for i in infos:
    print("info ", i)
print(f"\n{passed} checks passed, {len(fails)} failed")
sys.exit(1 if fails else 0)
