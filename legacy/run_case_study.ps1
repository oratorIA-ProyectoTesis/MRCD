# Estudio de caso sobre un video suelto: ingesta -> checkpoints -> inferencia -> QA.
#
# Uso:
#   powershell -ExecutionPolicy Bypass -File .\run_case_study.ps1 -Url "https://www.youtube.com/watch?v=Ka_okSSytes"
#
# Etapas 1-4 no cuestan API. La 5 (adjudicacion ciega) si, y es opcional:
# se lanza aparte cuando quieras convertir las discrepancias en una medicion.

param(
    [string]$Url = "https://www.youtube.com/watch?v=Ka_okSSytes",
    [string]$Vid = "",
    [int]$Epochs = 15
)

$ErrorActionPreference = "Stop"
$py = ".\.venv\Scripts\python.exe"

if (-not $Vid) {
    if ($Url -match "[?&]v=([A-Za-z0-9_-]{6,})") { $Vid = $Matches[1] }
    elseif ($Url -match "youtu\.be/([A-Za-z0-9_-]{6,})") { $Vid = $Matches[1] }
    else { Write-Host "No pude extraer el id del video de la URL. Pasa -Vid." -ForegroundColor Red; exit 1 }
}
Write-Host "Video: $Vid" -ForegroundColor Cyan

function Step($desc, $cmd) {
    Write-Host ""
    Write-Host "=== $desc ===" -ForegroundColor Cyan
    Invoke-Expression $cmd
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FALLO en '$desc' (codigo $LASTEXITCODE). Se detiene aqui." -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

Step "tests" "$py -m pytest -q tests"

# 1. Ingesta: descarga, pistas, ASR verbatim y cinesica con seguimiento de identidad.
Step "ingesta del video" "$py scripts\ingest_single_video.py --url `"$Url`""

# 2. Checkpoints. Solo se entrenan si no existen: reentrenar cada vez cambiaria
#    las predicciones entre corridas y el estudio de caso dejaria de ser comparable.
if (-not (Test-Path .\models\ckpt_trimodal.pt)) {
    if (-not (Test-Path .\data\windows_auto.npz)) {
        Step "ventanas de entrenamiento" "$py scripts\build_windows.py --labels auto"
    }
    Step "entrenamiento de los 3 checkpoints" "$py scripts\train_checkpoints.py --epochs $Epochs"
} else {
    Write-Host ""
    Write-Host "=== checkpoints ===" -ForegroundColor Cyan
    Write-Host "Ya existen en .\models\. Borra la carpeta para reentrenar."
}

# 3. Inferencia con los tres modos -> artefacto JSON.
Step "inferencia (audio_only / audio_text / trimodal)" "$py scripts\infer_video.py --vid $Vid"

# 4. Informe de QA + cola de adjudicacion.
Step "informe de control de calidad" "$py scripts\qa_report.py --vid $Vid"

Write-Host ""
Write-Host "Listo." -ForegroundColor Green
Write-Host "  data\results_$Vid.json     artefacto por ventana y por evento"
Write-Host "  case_study_interview.md    informe de QA"
Write-Host ""
Write-Host "El apartado (b) del informe esta SIN RESPONDER a proposito." -ForegroundColor Yellow
Write-Host "Decir que el trimodal corrigio al bimodal exige saber cual acerto, y este"
Write-Host "video no tiene etiquetas de verdad. Para medirlo en vez de afirmarlo:"
Write-Host ""
Write-Host "  $py scripts\adjudicate.py --vid $Vid --model gpt-4o   # cuesta API"
Write-Host "  $py scripts\qa_report.py --vid $Vid                   # reescribe el informe"
