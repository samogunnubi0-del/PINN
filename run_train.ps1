# Train PINN — uses venv Python (works when `python` is not on PATH)
# Run from anywhere:  & "path\to\run_train.ps1"
# Or from this folder:  .\run_train.ps1

$here = $PSScriptRoot
$py = Join-Path $here "venv\Scripts\python.exe"
$train = Join-Path $here "train.py"

if (-not (Test-Path $py)) {
    Write-Host "[X] venv not found. Run: python -m venv venv  then  pip install -r requirements.txt" -ForegroundColor Red
    exit 1
}
Set-Location $here
& $py $train @args
