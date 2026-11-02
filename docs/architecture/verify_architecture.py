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
5. The app (read-only, local branch feat/mrcd-integration) HAS an MRCD client and an opt-in; the integration facts drawn are checked
   in its source: default mode incremental (app commit 960c351; offline stays the automatic fallback), default profile p0, consent text version, SSE route, X-Session-Token header,
   3 default classes in the report, HTTP only (no `import mrcd`). No proposed node remains.
5b. Service facts: /v1/sessions routes, noloop/cuda refused in incremental, preload and warm-up wiring, ASR profiles registered.
5c. Package tags: av, optimize, context_ambiguity, recording_check are imported by no engine, extractor, live or service module;
   fusion_text and calibration are reached by the service only through s3_features (opt-in).
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

# 5 the app integration (read-only inspection of the local branch)
def text_of(base, rel):
    p = base / rel
    return p.read_text(encoding="utf-8", errors="ignore") if p.exists() else ""


def must(cond, msg):
    (ok if cond else fail)(msg)


if APP_ROOT.exists():
    br = subprocess.run(["git", "branch", "--show-current"], cwd=APP_ROOT, capture_output=True, text=True).stdout.strip()
    infos.append(f"app repo branch: {br or '(detached)'}; the diagrams describe feat/mrcd-integration")
    A = "backend/src/oratoria/"
    st = text_of(APP_ROOT, A + "platform/settings.py")
    must('mrcd_mode: Literal["offline", "incremental", "incremental_verify"] = "incremental"' in st, "app: mrcd_mode defaults to incremental (app commit 960c351; offline remains the fallback and an explicit mode)")
    must('mrcd_asr_profile: str = "p0"' in st, "app: default ASR profile is p0")
    must("mrcd_incremental_batch_ms: int = 2000" in st, "app: incremental batch period is 2000 ms")
    must("CONSENT_TEXT_VERSION" in text_of(APP_ROOT, A + "platform/mrcd/consent.py"), "app: consent decision is tied to a text version")
    cl = text_of(APP_ROOT, A + "platform/mrcd/client.py")
    must('"X-Session-Token"' in cl and "/v1/jobs" in cl and "subscription_token" in cl, "app: client uses /v1/jobs (offline) and the session token header")
    must("/v1/sessions" in cl, "app: client uses /v1/sessions (incremental)")
    must("/mrcd-analysis/events" in text_of(APP_ROOT, A + "bootstrap/mrcd_http.py"), "app: authenticated SSE route /sessions/{id}/mrcd-analysis/events")
    prog = text_of(APP_ROOT, "frontend/src/shared/api/mrcd-progress.ts")
    must("fetch" in prog and "Authorization" in prog, "app: frontend consumes the SSE with fetch + Authorization (not EventSource)")
    must("counts" in text_of(APP_ROOT, A + "platform/mrcd/progress.py").lower(), "app: progress hub documents counts-only events")
    dev = text_of(APP_ROOT, "frontend/src/contexts/feedback/components/DisfluencyEventsSection.tsx")
    # the scope is decided by the service (default output_scope "evaluated" = 3 classes, request sent by the app) and honoured per event via in_scope
    must('"output_scope": "evaluated"' in text_of(APP_ROOT, A + "platform/mrcd/analysis.py") and "isOutOfScope" in dev and "in_scope" in dev, "app: requests output_scope evaluated (3 classes) and the events section honours in_scope")
    must("SessionDisfluencySection" in text_of(APP_ROOT, "frontend/src/contexts/feedback/pages/ReportPage.tsx"), "app: ReportPage renders the disfluency section")
    leak = [
        p.relative_to(APP_ROOT).as_posix()
        for d in ("backend/src", "frontend/src")
        for p in (APP_ROOT / d).rglob("*")
        if p.is_file() and p.suffix in {".py", ".ts", ".tsx"} and "import mrcd" in p.read_text(encoding="utf-8", errors="ignore")
    ]
    must(not leak, f"app: nothing imports the mrcd Python package directly (HTTP only); offenders: {leak[:5]}")
else:
    infos.append("app repo not present; app-side checks skipped")

# 5b service facts
RT = text_of(ROOT, "mrcd/service/incremental/routes.py")
must(
    all(f in RT for f in ('"/v1/sessions"', '"/v1/sessions/{session_id}/audio/{seq}"', '"/v1/sessions/{session_id}/close"', '"/v1/sessions/{session_id}/events"', "text/event-stream", "subscription_token")) and "@app.delete" in RT,
    "service: /v1/sessions open (token), PUT audio/{seq}, close, SSE events and DELETE exist",
)
SE = text_of(ROOT, "mrcd/service/incremental/session.py")
must("profile_not_supported_incremental" in SE and "loop_redecode" in SE and '"cuda" in options.asr_profile' in SE, "service: noloop (loop_redecode) and cuda profiles are refused for incremental sessions")
must("counts only" in SE.lower(), "service: SSE events are documented as counts only")
SV = text_of(ROOT, "mrcd/service/server.py")
must("def prepare" in SV and "analyzer.preload" in SV and "warm_up" in SV and '"--warmup"' in text_of(ROOT, "mrcd/service/cli.py"), "service: preload and optional warm-up are wired (--warmup)")
must('"/v1/jobs"' in SV and '"/v1/jobs/{job_id}"' in SV, "service: offline job endpoints remain (reference and fallback)")
AP = text_of(ROOT, "mrcd/asr_profiles.py")
must(
    'DEFAULT_PROFILE = "p0"' in AP and all(k in AP for k in ('"p1_disfluent_prompt"', 'PROFILES["p0_cuda"]', 'PROFILES["p1_cuda"]', 'PROFILES["p0_noloop"]', 'PROFILES["p1_noloop"]')),
    "profiles: p0 default; p1, cuda and noloop variants registered",
)
must("loop_redecode=True" in AP, "profiles: noloop variants set loop_redecode")

# 5c package tags
product = ("mrcd/engine.py", "mrcd/inference.py", "mrcd/live/", "mrcd/service/", "mrcd/models/", "mrcd/extractors/", "mrcd/dataset.py", "mrcd/windowing.py")
for pkg in ("mrcd.av", "mrcd.context_ambiguity", "mrcd.recording_check"):
    hits = [f for f in lib if f.startswith(product) and any(n == pkg or n.startswith(pkg + ".") for n in imported_modules(f))]
    must(not hits, f"{pkg} is imported by no engine, extractor, live or service module (found: {hits})")
for pkg in ("mrcd.fusion_text", "mrcd.calibration"):
    users_pkg = sorted(f for f in lib if f != pkg.replace(".", "/") + ".py" and any(n == pkg or n.startswith(pkg + ".") for n in imported_modules(f)))
    svc_users = [f for f in users_pkg if f.startswith("mrcd/service/")]
    must(set(svc_users) <= {"mrcd/service/s3_features.py"}, f"{pkg} is reached inside mrcd/service only through s3_features (found: {svc_users})")
    must(not [f for f in users_pkg if f in ("mrcd/engine.py", "mrcd/inference.py") or f.startswith("mrcd/live/")], f"{pkg} is not imported by the engine, inference or live path")
must("DEFAULT_PIPELINE = False" in text_of(ROOT, "mrcd/optimize/product_pipeline.py"), "optimize.product_pipeline is opt-in (DEFAULT_PIPELINE False)")
# O integration (contract rev 1.4): the service adopts the optimised pipelines as its default decision profile, through
# ONE module and ONE optimize module (the pickle-free JSON loader). The engine, inference and live path never import optimize.
outside = sorted(f for f in lib if not f.startswith("mrcd/optimize/") and any(n == "mrcd.optimize" or n.startswith("mrcd.optimize.") for n in imported_modules(f)))
must(outside == ["mrcd/service/decision_profile.py"], f"only mrcd/service/decision_profile.py imports mrcd.optimize from outside it (found: {outside})")
opt_used = sorted(n for n in imported_modules("mrcd/service/decision_profile.py") if n == "mrcd.optimize" or n.startswith("mrcd.optimize."))
must(set(opt_used) <= {"mrcd.optimize", "mrcd.optimize.safe_pipeline"} and "mrcd.optimize.safe_pipeline" in opt_used, f"the service reaches optimize only through safe_pipeline (found: {opt_used})")
must(not [f for f in lib if f in ("mrcd/engine.py", "mrcd/inference.py") or f.startswith("mrcd/live/")
          if any(n == "mrcd.optimize" or n.startswith("mrcd.optimize.") for n in imported_modules(f))], "engine, inference and live never import optimize")
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
