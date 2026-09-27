# run_dashboard.ps1  –  Always launches Streamlit from the project venv
# Usage: .\run_dashboard.ps1

$venvStreamlit = Join-Path $PSScriptRoot ".venv\Scripts\streamlit.exe"

if (-not (Test-Path $venvStreamlit)) {
    Write-Error "Venv not found. Run: python -m venv .venv && .venv\Scripts\pip install -r requirements.txt"
    exit 1
}

Write-Host "Starting XAI-GT dashboard via venv Streamlit..." -ForegroundColor Cyan
& $venvStreamlit run dashboard/app.py
