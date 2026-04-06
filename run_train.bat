@echo off
REM Train PINN — uses venv Python (works when python is not on PATH)
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" (
    echo [X] venv not found. Create it and: pip install -r requirements.txt
    pause
    exit /b 1
)
venv\Scripts\python.exe train.py %*
