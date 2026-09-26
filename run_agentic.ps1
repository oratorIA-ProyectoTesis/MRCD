# Automated cycle: multimodal judge -> independent test -> ablation.
# Examples:
#   .\run_agentic.ps1                         # auto: Anthropic, OpenAI, then Gemini
#   .\run_agentic.ps1 -Provider openai         # OpenAI-only environment
#   .\run_agentic.ps1 -Provider groq -Model <vision-model>
#   .\run_agentic.ps1 -Provider ollama -Model <local-vision-model>
param(
    [ValidateSet("auto", "anthropic", "openai", "gemini", "groq", "ollama")]
    [string]$Provider = "auto",
    [string]$Model = "",
    [int]$PerClass = 60,
    [int]$Negatives = 120,
    [int]$Bootstrap = 300,
    [switch]$ValidateOnly
)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:PYTHONIOENCODING = "utf-8"; $env:PYTHONUTF8 = "1"
$py = ".\.venv\Scripts\python.exe"

function Step($desc, $cmd) {
    Write-Host "`n=== $desc ===" -ForegroundColor Cyan
    & $py @cmd
    if ($LASTEXITCODE -ne 0) { Write-Host "`nSe detuvo en: $desc (codigo $LASTEXITCODE)" -ForegroundColor Red; exit $LASTEXITCODE }
}

function Has-Key([string]$Name) {
    if ([Environment]::GetEnvironmentVariable($Name)) { return $true }
    if (-not (Test-Path .\.env)) { return $false }
    foreach ($line in Get-Content .\.env) {
        $parts = $line.Split('=', 2)
        if ($parts.Count -ne 2 -or $parts[0].Trim() -ne $Name) { continue }
        $value = $parts[1].Trim().Trim([char]34).Trim([char]39)
        if ($value) { return $true }
    }
    return $false
}

$resolvedProvider = $Provider
if ($resolvedProvider -eq "auto") {
    if ($Model) {
        if ($Model -match '^claude') { $resolvedProvider = "anthropic" }
        elseif ($Model -match '^gemini') { $resolvedProvider = "gemini" }
        elseif ($Model -match '^(llama-|mixtral-|gemma-)') { $resolvedProvider = "groq" }
        else { $resolvedProvider = "openai" }
    }
    elseif (Has-Key "ANTHROPIC_API_KEY") { $resolvedProvider = "anthropic" }
    elseif (Has-Key "OPENAI_API_KEY") { $resolvedProvider = "openai" }
    elseif (Has-Key "GEMINI_API_KEY") { $resolvedProvider = "gemini" }
    else {
        Write-Host "No default provider key found. Set ANTHROPIC_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY; for Groq/Ollama pass -Provider and -Model." -ForegroundColor Red
        exit 1
    }
}

if (-not $Model) {
    switch ($resolvedProvider) {
        "anthropic" { $Model = "claude-sonnet-4-5" }
        "openai"    { $Model = "gpt-4o" }
        "gemini"    { $Model = "gemini-2.5-flash" }
        default {
            Write-Host "Provider $resolvedProvider requires an explicit -Model with vision support." -ForegroundColor Red
            exit 1
        }
    }
}

$keyNames = @{ anthropic = "ANTHROPIC_API_KEY"; openai = "OPENAI_API_KEY";
               gemini = "GEMINI_API_KEY"; groq = "GROQ_API_KEY" }
if ($resolvedProvider -ne "ollama" -and -not (Has-Key $keyNames[$resolvedProvider])) {
    Write-Host "Missing $($keyNames[$resolvedProvider]) for provider $resolvedProvider (environment or .env)." -ForegroundColor Red
    exit 1
}
Write-Host "Judge provider: $resolvedProvider; model: $Model"
if ($ValidateOnly) { return }

Step "tests"                 @("-m","pytest","-q","tests")
Step "ventanas de entrenamiento (silver)" @("scripts\build_windows.py","--labels","auto")
Step "juez multimodal"       @("scripts\llm_judge.py","--provider",$resolvedProvider,"--model",$Model,"--per-class","$PerClass","--negatives","$Negatives")
Step "ventanas de test (gold)" @("scripts\build_windows.py","--labels","gold")
Step "fiabilidad del juez"   @("scripts\judge_agreement.py","--provider",$resolvedProvider,"--model",$Model,"--gold-manifest","data/dataset_manifest.json")
Step "plantilla de validacion humana" @("scripts\judge_agreement.py","--provider",$resolvedProvider,"--model",$Model,"--gold-manifest","data/dataset_manifest.json","--make-template","60")
Step "ablacion"              @("scripts\run_ablation.py","--data","data/windows_auto.npz","--test-data","data/windows_gold.npz","--bootstrap","$Bootstrap")
Step "soporte por clase"     @("scripts\report_support.py")
Write-Host "`nListo. Revisa results\ablation_table.md y results\judge_agreement.md."
Write-Host "Pendiente para que el test sea defendible: llenar data\judge\human_check.csv (60 items) y volver a correr judge_agreement.py."
