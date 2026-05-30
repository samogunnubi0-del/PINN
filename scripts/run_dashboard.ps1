# PowerShell launcher for PINN Dashboard
# Run from project root: .\scripts\run_dashboard.ps1

$ProjectRoot = Split-Path $PSScriptRoot -Parent
Set-Location $ProjectRoot

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host " PINN Isotope Transmutation Dashboard" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Check if venv exists
$venvDir = if (Test-Path ".\.venv") { ".\.venv" } else { ".\venv" }
if (-not (Test-Path $venvDir)) {
    Write-Host "[X] Virtual environment not found!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please run:" -ForegroundColor Yellow
    Write-Host "  python -m venv venv"
    Write-Host "  .\venv\Scripts\Activate.ps1"
    Write-Host "  pip install -r requirements.txt"
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

function Test-PortAvailable([int]$port) {
    try {
        $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Any, $port)
        $listener.Start()
        $listener.Stop()
        return $true
    } catch {
        return $false
    }
}

function Get-LanIp {
    try {
        $udp = [System.Net.Sockets.UdpClient]::new()
        $udp.Connect("8.8.8.8", 53)
        $ep = $udp.Client.LocalEndPoint
        $udp.Close()
        return $ep.Address.ToString()
    } catch {
        return "127.0.0.1"
    }
}

$port = 8501
if (-not (Test-PortAvailable -port $port)) {
    Write-Host "[!] Port 8501 is busy. Trying another port..." -ForegroundColor Yellow
    for ($p = 8502; $p -le 8510; $p++) {
        if (Test-PortAvailable -port $p) {
            $port = $p
            break
        }
    }
}

$lanIp = Get-LanIp
$localUrl = "http://localhost:$port"
$lanUrl = "http://${lanIp}:$port"

Write-Host "[*] Starting Streamlit dashboard..." -ForegroundColor Green
Write-Host ""
Write-Host "    This PC:  $localUrl" -ForegroundColor Cyan
Write-Host "    Same Wi-Fi: $lanUrl" -ForegroundColor Cyan
Write-Host "    Server binds to 0.0.0.0 (see .streamlit/config.toml)" -ForegroundColor DarkGray
Write-Host "    Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

# Terminal QR (ASCII) for quick phone scan from the same Wi-Fi.
$tmpQr = Join-Path $env:TEMP ("pinn_qr_" + [guid]::NewGuid().ToString() + ".py")
$qrScript = @"
import socket
import sys
import os

try:
    import qrcode
except Exception:
    print("[i] Install optional QR terminal support with: pip install qrcode[pil]")
    sys.exit(0)

url = os.environ.get("PINN_QR_URL", "http://127.0.0.1:8501")
print(f"[i] Terminal QR URL: {url}")
qr = qrcode.QRCode(border=1)
qr.add_data(url)
qr.make(fit=True)
qr.print_ascii(invert=True)
"@

$qrScript | Set-Content -Path $tmpQr -Encoding UTF8
$env:PINN_QR_URL = $lanUrl
& "$venvDir\Scripts\python.exe" "$tmpQr"
Remove-Item "$tmpQr" -ErrorAction SilentlyContinue
Write-Host ""

# Use venv Python directly — avoids broken streamlit.exe after moving the project folder
& "$venvDir\Scripts\python.exe" -m streamlit run app.py --server.port $port
