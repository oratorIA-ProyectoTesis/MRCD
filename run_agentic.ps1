# Ciclo 100 % automatizado: juez multimodal -> test independiente -> ablacion
# Requiere una API key en el entorno:
#   $env:ANTHROPIC_API_KEY="sk-ant-..."     (o)   $env:OPENAI_API_KEY="sk-..."
# Uso:  powershell -ExecutionPolicy Bypass -File .\run_agentic.ps1 [-Model claude-sonnet-4-5]
param([string]$Model = "claude-sonnet-4-5", [int]$PerClass = 60, [int]$Negatives = 120, [int]$Bootstrap = 300)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUTF8 = "1"
$py = ".\.venv\Scripts\python.exe"

function Step($desc, $cmd) {
    Write-Host "`n=== $desc ===" -ForegroundColor Cyan
    & $py @cmd
    if ($LASTEXITCODE -ne 0) { Write-Host "`nSe detuvo en: $desc (codigo $LASTEXITCODE)" -ForegroundColor Red; exit $LASTEXITCODE }
}

if (-not $env:OPENAI_API_KEY -and -not $env:ANTHROPIC_API_KEY -and -not (Test-Path .\.env)) {
    Write-Host "Falta la API key. Crea un archivo .env con  OPENAI_API_KEY=sk-...  o define la variable de entorno." -ForegroundColor Red
    exit 1
}

Step "tests"                 @("-m","pytest","-q","tests")
Step "ventanas de entrenamiento (silver)" @("scripts\build_windows.py","--labels","auto")
Step "juez multimodal"       @("scripts\llm_judge.py","--model",$Model,"--per-class","$PerClass","--negatives","$Negatives")
Step "ventanas de test (gold)" @("scripts\build_windows.py","--labels","gold")
Step "fiabilidad del juez"   @("scripts\judge_agreement.py")
Step "plantilla de validacion humana" @("scripts\judge_agreement.py","--make-template","60")
Step "ablacion"              @("scripts\run_ablation.py","--data","data/windows_auto.npz","--test-data","data/windows_gold.npz","--bootstrap","$Bootstrap")
Step "soporte por clase"     @("scripts\report_support.py")
Write-Host "`nListo. Revisa results\ablation_table.md y results\judge_agreement.md."
Write-Host "Pendiente para que el test sea defendible: llenar data\judge\human_check.csv (60 items) y volver a correr judge_agreement.py."
