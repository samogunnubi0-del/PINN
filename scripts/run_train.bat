@echo off
REM Train PINN — uses venv Python (works when python is not on PATH)
cd /d "%~dp0\.."
if exist ".venv\Scripts\python.exe" (
    set "PY=.venv\Scripts\python.exe"
) else if exist "venv\Scripts\python.exe" (
    set "PY=venv\Scripts\python.exe"
) else (
    echo [X] venv not found. Create it and: pip install -r requirements.txt
    pause
    exit /b 1
)
"%PY%" train.py %*
