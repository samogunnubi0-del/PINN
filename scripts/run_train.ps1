# Train PINN — uses venv Python (works when `python` is not on PATH)
# Run from anywhere:  & "path\to\run_train.ps1"
# Or from this folder:  .\run_train.ps1

$ProjectRoot = Split-Path $PSScriptRoot -Parent
$py = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    $py = Join-Path $ProjectRoot "venv\Scripts\python.exe"
}
$train = Join-Path $ProjectRoot "train.py"

if (-not (Test-Path $py)) {
    Write-Host "[X] venv not found. Run: python -m venv .venv  then  pip install -r requirements.txt" -ForegroundColor Red
    exit 1
}
Set-Location $ProjectRoot
& $py $train @args
