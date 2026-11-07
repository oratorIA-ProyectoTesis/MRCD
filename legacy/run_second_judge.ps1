param(
    [string]$Vid = "Ka_okSSytes"
)

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
Step "segundo juez Gemini" "$py scripts\second_judge_gemini.py --vid $Vid --model gemini-2.5-flash"
Step "informe de QA" "$py scripts\qa_report.py --vid $Vid"

Write-Host ""
Write-Host "Listo." -ForegroundColor Green