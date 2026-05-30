@echo off
REM Quick launcher for PINN Dashboard
cd /d "%~dp0\.."

echo.
echo ==========================================
echo  PINN Isotope Transmutation Dashboard
echo ==========================================
echo.

if exist ".venv\" (
    set "VENV=.venv"
) else if exist "venv\" (
    set "VENV=venv"
) else (
    echo [X] Virtual environment not found!
    echo.
    echo Please run:
    echo   python -m venv .venv
    echo   .\.venv\Scripts\Activate.ps1
    echo   pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

echo [*] Starting Streamlit dashboard...
echo.
echo    This PC: http://localhost:8501
echo    Press Ctrl+C to stop the server
echo.

"%VENV%\Scripts\python.exe" -m streamlit run app.py

pause
