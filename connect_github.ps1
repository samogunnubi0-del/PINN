# After you create an EMPTY repo on GitHub (no README), run from this folder:
#   .\connect_github.ps1
# Or pass the URL:
#   .\connect_github.ps1 -RepoUrl "https://github.com/YOU/ac225-pinn.git"

param(
    [string]$RepoUrl = ""
)

$gitExe = "C:\Program Files\Git\bin\git.exe"
if (-not (Test-Path $gitExe)) {
    $gitExe = "git"
}

if ($RepoUrl -eq "") {
    $RepoUrl = Read-Host "Paste your GitHub HTTPS URL (e.g. https://github.com/you/repo.git)"
}

if ($RepoUrl -notmatch "^https://github\.com/.+\.git$") {
    Write-Host "[!] URL should look like: https://github.com/username/repo.git" -ForegroundColor Yellow
}

Set-Location $PSScriptRoot

Write-Host "[*] Removing old origin (if any)..." -ForegroundColor DarkGray
& $gitExe remote remove origin 2>$null

Write-Host "[*] Adding origin..." -ForegroundColor Cyan
& $gitExe remote add origin $RepoUrl

Write-Host "[*] Pushing branch main..." -ForegroundColor Cyan
& $gitExe push -u origin main

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "[OK] Next: https://share.streamlit.io → New app → pick this repo → app.py" -ForegroundColor Green
    Write-Host "     Requirements file: requirements-streamlit-cloud.txt" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "[X] Push failed. Common fixes:" -ForegroundColor Red
    Write-Host "    - Create the empty repo on GitHub first" -ForegroundColor Yellow
    Write-Host "    - Sign in: Git Credential Manager or a Personal Access Token" -ForegroundColor Yellow
}
