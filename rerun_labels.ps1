# Re-etiquetado rapido tras cambiar reglas (NO vuelve a descargar ni a correr ASR/MediaPipe)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUTF8 = "1"
$py = ".\.venv\Scripts\python.exe"
& $py -m pytest -q tests
& $py scripts\relabel.py
& $py scripts\build_windows.py --labels auto
& $py scripts\run_ablation.py --data data/windows_auto.npz --allow-auto-labels --bootstrap 300
& $py scripts\report_support.py
& $py scripts\sample_for_annotation.py
