# MRCD - instalacion + ejecucion completa (Windows PowerShell)
# Uso:  powershell -ExecutionPolicy Bypass -File .\setup_and_run.ps1
#       (argumentos extra se pasan a run_all.py, p. ej. --skip-mining)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUTF8 = "1"

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Host "Instalando ffmpeg con winget..."
    winget install -e --id Gyan.FFmpeg --accept-source-agreements --accept-package-agreements
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
}

if (-not (Test-Path .\.venv\Scripts\python.exe)) {
    $py = $null
    foreach ($v in @("3.11","3.12","3.10")) { try { py -$v -c "import sys" 2>$null; if ($LASTEXITCODE -eq 0) { $py = "py -$v"; break } } catch {} }
    if (-not $py) { $py = "python" }
    Write-Host "Creando entorno virtual con $py ..."
    Invoke-Expression "$py -m venv .venv"
}
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\run_all.py @args 2>&1 | Tee-Object -FilePath run_log.txt
