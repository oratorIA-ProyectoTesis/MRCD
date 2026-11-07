# ============================================================================
#  Cadena v4: corpus re-extraido con transcripcion por trozos (--asr-chunk-s 8)
# ============================================================================
#
#  Motivo del corpus v4
#  --------------------
#  Whisper borra repeticiones y muletillas cuando transcribe mucho contexto de
#  una vez. Demostracion causal (results/asr_chunk_study.md): el mismo audio, el
#  mismo checkpoint y la misma ventana dan `prolongation` 44 % con el archivo
#  entero y `repetition` 95 % con trozos de 8 s. La extraccion anterior
#  transcribia segmentos completos, y 64 de las 87 grabaciones superan los 30 s,
#  longitud a la que sobreviven 0 repeticiones.
#
#  Que NO demuestra esta cadena
#  ----------------------------
#  Mas eventos etiquetados NO es mas exactitud: son etiquetas heuristicas. Si el
#  F1 de `repetition` sube en v4, puede ser simplemente que el conjunto de
#  entrenamiento cambio. La comparacion valida es contra el gold del juez, con
#  particion por hablante, y con el kappa juez-humano aun pendiente.
#
#  Uso:
#      powershell -ExecutionPolicy Bypass -File .\run_v4.ps1            # sin juez
#      powershell -ExecutionPolicy Bypass -File .\run_v4.ps1 -Judge     # con juez (CUESTA DINERO)
# ============================================================================

param(
    [switch]$Judge,          # ejecutar el juez VLM (llamadas de API de pago)
    [switch]$DryRunJudge,    # muestreo del juez sin gastar API
    [int]$PerClass = 60,
    [int]$Negatives = 120,
    [string]$JudgeModel = "gpt-4o"   # mismo juez que v3: aisla el cambio de corpus
)

$ErrorActionPreference = "Stop"
$ROOT = Split-Path $PSScriptRoot -Parent  # legacy/ -> repository root
$PY = Join-Path $ROOT ".venv\Scripts\python.exe"
Set-Location $ROOT

function Step($n, $msg) { Write-Host "`n=== [$n] $msg ===" -ForegroundColor Cyan }
function Ok($msg)       { Write-Host "  OK  $msg" -ForegroundColor Green }
function Warn($msg)     { Write-Host "  !!  $msg" -ForegroundColor Yellow }
function Die($msg)      { Write-Host "  XX  $msg" -ForegroundColor Red; exit 1 }

# ---------------------------------------------------------------- 0. guardias
Step 0 "Verificaciones previas"

if (-not (Test-Path $PY)) { Die "No existe $PY. Corre setup_and_run.ps1 primero." }

$manV4 = Join-Path $ROOT "data\v4\dataset_manifest.json"
if (-not (Test-Path $manV4)) { Die "Falta $manV4. La extraccion v4 no ha corrido." }

# La extraccion escribe el manifest tras CADA grabacion, asi que un manifest
# incompleto es indistinguible de uno terminado si no se cuenta. Se delega el
# conteo a un script en vez de incrustar Python en la linea: PowerShell parsea
# las comillas del one-liner antes de pasarlo, y rompe.
$countScript = Join-Path $ROOT "scripts\_count_recordings.py"
$nV4 = (& $PY $countScript $manV4).Trim()
$manV3 = Join-Path $ROOT "data\dataset_manifest.json"
$nV3 = (& $PY $countScript $manV3).Trim()
Write-Host "  grabaciones: v3=$nV3  v4=$nV4"
if ([int]$nV4 -lt [int]$nV3) {
    Warn "v4 tiene MENOS grabaciones que v3. La extraccion sigue corriendo o fallo."
    Warn "Comparar corpus de tamanos distintos atribuiria al cambio de ASR lo que es"
    Warn "diferencia de cobertura."
    $r = Read-Host "  Continuar de todos modos? (s/N)"
    if ($r -ne "s") { exit 1 }
}
Ok "manifests presentes"

# ------------------------------------------------- 1. comparacion de etiquetas
Step 1 "Comparacion de etiquetas v3 vs v4"
& $PY scripts/compare_extractions.py `
    --a data/dataset_manifest.json `
    --b data/v4/dataset_manifest.json `
    --label-a "archivo entero" --label-b "trozos 8s" `
    --out results/extraction_comparison.md
if ($LASTEXITCODE -ne 0) { Die "compare_extractions fallo" }
Ok "results/extraction_comparison.md"

# ------------------------------------------------------- 2. ventanas de train
Step 2 "Ventanas de entrenamiento (etiquetas auto, norm cinesica por hablante)"
& $PY scripts/build_windows.py `
    --manifest data/v4/dataset_manifest.json `
    --labels auto --kinesic-norm speaker `
    --out data/v4/windows_auto.npz
if ($LASTEXITCODE -ne 0) { Die "build_windows (auto) fallo" }
Ok "data/v4/windows_auto.npz"

# ------------------------------------------------------------------ 3. juez
if ($DryRunJudge) {
    Step 3 "Juez - DRY RUN (sin gasto de API)"
    & $PY scripts/llm_judge.py --manifest data/v4/dataset_manifest.json `
        --judge-dir data/v4/judge --per-class $PerClass --negatives $Negatives `
        --model $JudgeModel --dry-run
    Ok "muestreo mostrado; no se llamo a la API"
    Write-Host "`nPara ejecutarlo de verdad: .\run_v4.ps1 -Judge" -ForegroundColor Yellow
    exit 0
}

if (-not $Judge) {
    Step 3 "Juez - OMITIDO"
    Warn "Sin gold nuevo no se puede evaluar v4: el gold de v3 se construyo sobre"
    Warn "candidatos de otra extraccion y sus intervalos no corresponden."
    Write-Host "`nSiguiente paso:" -ForegroundColor Yellow
    Write-Host "  .\run_v4.ps1 -DryRunJudge     # ver el muestreo sin gastar"
    Write-Host "  .\run_v4.ps1 -Judge           # ejecutar (CUESTA DINERO)"
    exit 0
}

Step 3 "Juez multimodal ciego sobre el corpus v4"
# llm_judge.py lee .env ademas del entorno, asi que la guardia mira ambos.
$hasEnvKey = $env:OPENAI_API_KEY -or $env:ANTHROPIC_API_KEY -or $env:GEMINI_API_KEY
$hasDotEnv = (Test-Path (Join-Path $ROOT ".env")) -and `
             (Select-String -Path (Join-Path $ROOT ".env") -Pattern "^\s*(OPENAI|ANTHROPIC|GEMINI)_API_KEY\s*=\s*\S" -Quiet)
if (-not $hasEnvKey -and -not $hasDotEnv) {
    Die "Falta una API key. Ponla en .env (OPENAI_API_KEY=sk-...) o en el entorno."
}
Write-Host "  juez: $JudgeModel  (v3 se juzgo con gpt-4o)"
Warn "Esto hace llamadas de API de pago. El cache es reanudable:"
Warn "si se corta, volver a lanzar retoma donde quedo."
& $PY scripts/llm_judge.py --manifest data/v4/dataset_manifest.json `
    --judge-dir data/v4/judge --per-class $PerClass --negatives $Negatives `
    --model $JudgeModel
if ($LASTEXITCODE -ne 0) { Die "llm_judge fallo (el cache conserva lo ya juzgado)" }
Ok "data/v4/judge/judgments.jsonl"

# --------------------------------------------------------- 4. ventanas gold
Step 4 "Ventanas de prueba (etiquetas del juez)"
& $PY scripts/build_windows.py `
    --manifest data/v4/dataset_manifest.json `
    --labels gold --kinesic-norm speaker `
    --norm-stats data/v4/kinesic_norm_stats.json `
    --out data/v4/windows_gold.npz
if ($LASTEXITCODE -ne 0) { Die "build_windows (gold) fallo" }
Ok "data/v4/windows_gold.npz"

# --------------------------------------------------- 5. fiabilidad del juez
Step 5 "Fiabilidad del juez"
& $PY scripts/judge_agreement.py `
    --judgments data/v4/judge/judgments.jsonl `
    --human data/v4/judge/human_check.csv `
    --out results/v4
if ($LASTEXITCODE -ne 0) { Warn "judge_agreement fallo; continuo" }

# -------------------------------------------------------------- 6. ablacion
Step 6 "Ablacion v4 (particion por hablante, sin fuga)"
& $PY scripts/run_ablation.py `
    --data data/v4/windows_auto.npz `
    --test-data data/v4/windows_gold.npz `
    --partition speaker --modality-dropout 0.0 --bootstrap 1000 `
    --out results/v4
if ($LASTEXITCODE -ne 0) { Die "run_ablation fallo" }
Ok "results/v4/ablation_table.md"

# ------------------------------------------------------------------ resumen
Write-Host "`n=== LISTO ===" -ForegroundColor Green
Write-Host "  results/extraction_comparison.md   etiquetas v3 vs v4"
Write-Host "  results/v4/judge_agreement.md      fiabilidad del juez"
Write-Host "  results/v4/ablation_table.md       ablacion con el corpus v4"
Write-Host ""
Write-Host "AL LEER LOS RESULTADOS:" -ForegroundColor Yellow
Write-Host "  - Un F1 mayor en v4 NO prueba un modelo mejor: cambio el corpus de"
Write-Host "    entrenamiento Y el gold. Son dos sistemas distintos, no dos versiones"
Write-Host "    del mismo medidas con la misma vara."
Write-Host "  - La prediccion registrada de antemano es que el acuerdo del juez con"
Write-Host "    'prolongation' SUBA respecto al 12/95 de v3, porque desaparecen falsos"
Write-Host "    positivos que en realidad eran muletillas que el ASR no transcribia."
Write-Host "    Si no sube, esa lectura es incorrecta."
Write-Host "  - Sigue pendiente el kappa juez-humano. Sin el, nada de esto es citable."
