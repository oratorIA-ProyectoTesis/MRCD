#!/usr/bin/env python3
"""Orquestador end-to-end (reanudable). Genera RUN_REPORT.md y el resumen final en consola.

  python scripts/run_all.py                 # todo
  python scripts/run_all.py --skip-mining   # si ya minaste
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.constants import RANDOM_SEED, SMOKE_HEADER  # noqa: E402

PY = sys.executable
LOG: list[dict] = []


def stage(name: str, cmd: list[str], critical: bool = True) -> bool:
    print(f"\n{'='*70}\n▶ {name}\n  $ {' '.join(cmd)}\n{'='*70}", flush=True)
    t0 = time.time()
    import os
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         encoding="utf-8", errors="replace", env=env)
    tail: list[str] = []
    for line in p.stdout:  # type: ignore
        print(line, end="", flush=True)
        tail = (tail + [line.rstrip()])[-25:]
    rc = p.wait()
    LOG.append(dict(stage=name, cmd=" ".join(cmd), seconds=round(time.time() - t0, 1), rc=rc, tail=tail if rc else []))
    write_report()
    if rc != 0 and critical:
        print(f"\n✗ Etapa '{name}' falló (rc={rc}). Revisa RUN_REPORT.md. Puedes relanzar: el pipeline es reanudable.")
        sys.exit(rc)
    return rc == 0


def versions() -> dict:
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for m in ("torch", "faster_whisper", "ctranslate2", "mediapipe", "yt_dlp", "librosa", "silero_vad", "numpy"):
        try:
            mod = __import__(m)
            out[m] = getattr(mod, "__version__", getattr(getattr(mod, "version", None), "__version__", "?"))
        except Exception:
            out[m] = "no instalado"
    try:
        import torch
        out["cuda"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no (CPU)"
    except Exception:
        pass
    return out


def write_report() -> None:
    L = ["# RUN_REPORT", "", f"Generado: {time.strftime('%Y-%m-%d %H:%M:%S')} · semilla aleatoria: {RANDOM_SEED}", "",
         "## Versiones", ""] + [f"- {k}: {v}" for k, v in versions().items()]
    L += ["", "## Etapas", "", "| Etapa | Duración | rc | Comando |", "|---|---|---|---|"]
    L += [f"| {s['stage']} | {s['seconds']/60:.1f} min | {s['rc']} | `{s['cmd']}` |" for s in LOG]
    fails = [s for s in LOG if s["rc"]]
    if fails:
        L += ["", "## Fallos", ""]
        for s in fails:
            L += [f"### {s['stage']}", "```", *s["tail"], "```"]
    st_p = ROOT / "data/raw/state.json"
    if st_p.exists():
        st = json.loads(st_p.read_text(encoding="utf-8"))
        rej = {k: v for k, v in st.items() if v.get("status") == "rejected"}
        L += ["", "## Videos", "", f"- extraídos: {sum(v.get('status') == 'extracted' for v in st.values())}",
              f"- descargados sin extraer: {sum(v.get('status') == 'downloaded' for v in st.values())}",
              f"- descartados: {len(rej)}", "", "| video_id | etapa | motivo |", "|---|---|---|"]
        L += [f"| {k} | {v.get('stage','')} | {v.get('reason','')} |" for k, v in rej.items()]
    (ROOT / "RUN_REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def final_summary() -> None:
    print("\n" + "#" * 70 + "\nRESUMEN FINAL\n" + "#" * 70)
    st_p = ROOT / "data/raw/state.json"
    if st_p.exists():
        st = json.loads(st_p.read_text(encoding="utf-8"))
        print(f"(a) Videos procesados: {sum(v.get('status') == 'extracted' for v in st.values())} | "
              f"descartados: {sum(v.get('status') == 'rejected' for v in st.values())} (motivos en data/raw/rejected.csv)")
    cs = ROOT / "results/class_support.json"
    if cs.exists():
        d = json.loads(cs.read_text(encoding="utf-8"))
        print("(b) Soporte por clase (eventos conf ≥ 0.5 | ventanas):")
        for c, n in d["events"].items():
            print(f"     {c:17s} {n:6d} | {d['windows'].get(c, '—')}")
        if d["zero_classes"]:
            print(f"     Clases con soporte 0: {', '.join(d['zero_classes'])} (ver results/class_support.md)")
    tab = ROOT / "results/ablation_table.md"
    if tab.exists():
        print("(c) Tabla de ablación:\n" + tab.read_text(encoding="utf-8"))
    print(f"(d) Paquete de anotación: {ROOT / 'data/annotation_batch_01'}")
    print(f"\nRecordatorio: {SMOKE_HEADER}. La evidencia real sale de windows_human.npz tras la anotación.")


def keep_awake() -> None:
    """Windows: evita la suspensión SOLO mientras corre este proceso (no cambia la configuración)."""
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
        except Exception:
            pass


def main():
    keep_awake()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-tests", action="store_true")
    ap.add_argument("--skip-mining", action="store_true")
    ap.add_argument("--whisper", default="small")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--bootstrap", type=int, default=300)
    a = ap.parse_args()

    if not a.skip_tests:
        stage("tests previos", [PY, "-m", "pytest", "-q", "tests"])
    stage("modelos", [PY, "scripts/download_models.py", "--whisper", a.whisper])
    if not a.skip_mining:
        stage("minería (seeds + discover-auto + extract)", [PY, "scripts/data_mining.py", "all", "--whisper", a.whisper])
    xl = [PY, "scripts/extract_and_label.py", "--whisper", a.whisper, "--recall-mode"]
    if a.workers:
        xl += ["--workers", str(a.workers)]
    stage("extracción 3 señales + pre-anotación", xl)
    stage("ventanas", [PY, "scripts/build_windows.py", "--labels", "auto"])
    stage("ablación (prueba de humo)", [PY, "scripts/run_ablation.py", "--data", "data/windows_auto.npz",
                                        "--allow-auto-labels", "--bootstrap", str(a.bootstrap)], critical=False)
    stage("soporte por clase", [PY, "scripts/report_support.py"], critical=False)
    stage("paquete de anotación", [PY, "scripts/sample_for_annotation.py"], critical=False)
    if not a.skip_tests:
        stage("tests finales", [PY, "-m", "pytest", "-q", "tests"], critical=False)
    write_report()
    final_summary()


if __name__ == "__main__":
    main()
