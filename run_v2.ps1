# Reconstruye ventanas y repite la ablacion con los rasgos contextuales v2.
#
# Coste de API: CERO. Los rasgos nuevos (frontera prosodica, ventana linguistica
# ampliada, normalizacion tonal por hablante) se calculan desde los .npz ya
# cacheados. No se vuelve a correr ASR, ni MediaPipe, ni el juez.
#
# Uso:  powershell -ExecutionPolicy Bypass -File .\run_v2.ps1

$ErrorActionPreference = "Stop"
$py = ".\.venv\Scripts\python.exe"

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

Step "ventanas de entrenamiento (silver, con rasgos v2)" `
     "$py scripts\build_windows.py --labels auto"

if (Test-Path .\data\judge\judgments.jsonl) {
    Step "ventanas de test (gold, con rasgos v2)" `
         "$py scripts\build_windows.py --labels gold"

    Step "ablacion v2 - contexto + prosodia" `
         "$py scripts\run_ablation.py --data data\windows_auto.npz --test-data data\windows_gold.npz --out results\v2"
} else {
    Write-Host ""
    Write-Host "No hay juicios del juez todavia; se corre la ablacion en modo humo." -ForegroundColor Yellow
    Step "ablacion v2 (humo)" `
         "$py scripts\run_ablation.py --data data\windows_auto.npz --allow-auto-labels --out results\v2"
}

Write-Host ""
Write-Host "Listo. Compara results\v2\ablation_table.md contra results\ablation_table.md." -ForegroundColor Green
Write-Host "Lo que hay que mirar, en este orden:" -ForegroundColor Green
Write-Host "  1. rhetorical_pause y neutral_pause: son las clases que los rasgos nuevos atacan."
Write-Host "  2. macro-F1 de audio_only: la prosodia entra en los TRES modos, asi que"
Write-Host "     tambien deberia subir. Si solo sube trimodal, algo esta mal conectado."
Write-Host "  3. trimodal-audio_text: si sigue cruzando cero, el video no aporta y eso"
Write-Host "     se reporta como resultado, no se esconde."
