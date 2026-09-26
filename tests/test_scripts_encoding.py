"""Los .ps1 deben ser ASCII puro.

Windows PowerShell 5.1 lee un .ps1 sin BOM como Windows-1252, no como UTF-8.
Un guion largo escrito en UTF-8 (bytes E2 80 94) se decodifica entonces como
`â€"` — y esa comilla ROMPE la cadena que lo contiene. El error que aparece
("Falta la cadena en el terminador") apunta a una línea muy posterior, así que
cuesta relacionarlo con la causa.

Cualquier carácter no ASCII dentro de una cadena es un fallo esperando ocurrir,
según qué bytes produzca al mal decodificarse. La regla simple y segura es no
tener ninguno.
"""
from __future__ import annotations

from pathlib import Path
import os
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
PS1 = sorted(ROOT.glob("*.ps1"))


def test_there_are_scripts_to_check():
    assert PS1, "no se encontró ningún .ps1"


@pytest.mark.parametrize("p", PS1, ids=lambda p: p.name)
def test_script_is_pure_ascii(p: Path):
    raw = p.read_bytes()
    bad = [(i, b) for i, b in enumerate(raw) if b > 127]
    if bad:
        i = bad[0][0]
        ctx = raw[max(i - 40, 0):i + 40].decode("utf-8", "replace")
        pytest.fail(f"{p.name}: {len(bad)} bytes no ASCII; el primero en la posición {i}.\n"
                    f"Contexto: ...{ctx}...\n"
                    f"PowerShell 5.1 los leerá como Windows-1252 y puede romper el script.")


@pytest.mark.parametrize("p", PS1, ids=lambda p: p.name)
def test_script_quotes_and_braces_are_balanced(p: Path):
    t = p.read_bytes().decode("ascii")
    assert t.count('"') % 2 == 0, f"{p.name}: número impar de comillas dobles"
    assert t.count("{") == t.count("}"), f"{p.name}: llaves desbalanceadas"


@pytest.mark.parametrize("p", PS1, ids=lambda p: p.name)
def test_no_whitespace_after_a_line_continuation(p: Path):
    """Un backtick de continuación con un espacio detrás deja de continuar la
    línea, y el comando se parte en dos sin avisar."""
    for i, line in enumerate(p.read_bytes().decode("ascii").split("\r\n"), 1):
        assert not (line.endswith("` ") or line.endswith("`\t")), \
            f"{p.name}:{i} espacio detrás del backtick de continuación"


@pytest.mark.skipif(shutil.which("pwsh") is None, reason="PowerShell unavailable")
@pytest.mark.parametrize("provider,model,key,expected", [
    ("auto", "", "OPENAI_API_KEY", "openai; model: gpt-4o"),
    ("gemini", "", "GEMINI_API_KEY", "gemini; model: gemini-2.5-flash"),
    ("groq", "vision-test", "GROQ_API_KEY", "groq; model: vision-test"),
    ("ollama", "local-vision", None, "ollama; model: local-vision"),
])
def test_agentic_provider_resolution_without_running_pipeline(tmp_path, provider, model, key, expected):
    script = tmp_path / "run_agentic.ps1"
    script.write_bytes((ROOT / "run_agentic.ps1").read_bytes())
    env = {k: v for k, v in os.environ.items() if k not in
           ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GROQ_API_KEY")}
    if key:
        env[key] = "test-key"
    cmd = ["pwsh", "-NoProfile", "-File", str(script), "-ValidateOnly", "-Provider", provider]
    if model:
        cmd += ["-Model", model]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
    assert expected in result.stdout


@pytest.mark.skipif(shutil.which("pwsh") is None, reason="PowerShell unavailable")
def test_agentic_groq_requires_model_even_with_credential(tmp_path):
    script = tmp_path / "run_agentic.ps1"
    script.write_bytes((ROOT / "run_agentic.ps1").read_bytes())
    env = {**os.environ, "GROQ_API_KEY": "test-key"}
    result = subprocess.run(["pwsh", "-NoProfile", "-File", str(script), "-ValidateOnly",
                             "-Provider", "groq"], capture_output=True, text=True, env=env, timeout=15)
    assert result.returncode != 0
    assert "requires an explicit -Model" in result.stdout + result.stderr


def test_agentic_agreement_receives_the_same_experiment_identity():
    script = (ROOT / "run_agentic.ps1").read_text(encoding="ascii")
    agreement_lines = [line for line in script.splitlines() if '"scripts\\judge_agreement.py"' in line]
    assert len(agreement_lines) == 2
    assert all('"--provider",$resolvedProvider,"--model",$Model' in line for line in agreement_lines)
    assert all('"--gold-manifest","data/dataset_manifest.json"' in line for line in agreement_lines)
