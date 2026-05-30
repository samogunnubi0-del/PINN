@echo off
REM Full epochs & data; CPU/runtime speedups only (see train.py docstring).
cd /d "%~dp0\.."
set PINN_FAST_CPU=1
if exist ".venv\Scripts\python.exe" (
    set "PY=.venv\Scripts\python.exe"
) else if exist "venv\Scripts\python.exe" (
    set "PY=venv\Scripts\python.exe"
) else (
    echo [X] venv not found.
    exit /b 1
)
"%PY%" train.py %*
