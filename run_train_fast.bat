@echo off
REM Full epochs & data; CPU/runtime speedups only (see train.py docstring).
cd /d "%~dp0"
set PINN_FAST_CPU=1
if not exist "venv\Scripts\python.exe" (
    echo [X] venv not found.
    exit /b 1
)
venv\Scripts\python.exe train.py %*
