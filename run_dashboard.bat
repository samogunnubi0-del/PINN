@echo off
REM Quick launcher for PINN Dashboard
REM This script activates the venv and starts the Streamlit app

echo.
echo ==========================================
echo  PINN Isotope Transmutation Dashboard
echo ==========================================
echo.

if not exist "venv\" (
    echo [X] Virtual environment not found!
    echo.
    echo Please run:
    echo   python -m venv venv
    echo   .\venv\Scripts\Activate.ps1
    echo   pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

echo [*] Starting Streamlit dashboard...
echo.
echo    This PC: http://localhost:8501
echo    Phones on same Wi-Fi: use QR in app ^(tab: Open on phone^) or http://YOUR-IP:8501
echo    Terminal ASCII QR is available in: .\run_dashboard.ps1
echo    Press Ctrl+C to stop the server
echo.

REM Use venv Python directly — avoids broken streamlit.exe after moving the project folder
.\venv\Scripts\python.exe -m streamlit run app.py

pause
