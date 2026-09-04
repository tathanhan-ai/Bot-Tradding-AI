@echo off
chcp 65001 >nul
title Binance Futures AI Quant Bot (Port 8000 + 9Router 8039)
echo ==================================================================
echo [PORTABLE LAUNCHER] Khoi dong Binance Futures AI Bot
echo ==================================================================
cd /d "%~dp0"

if exist "gateway_9router\run_9router.js" (
    echo [9Router] Dang khoi dong AI Gateway tren cong 8039...
    start "9Router AI Gateway (Port 8039)" cmd /c "cd /d %~dp0gateway_9router && node run_9router.js"
)

echo [UI Server] Khoi dong Giao dien Bot tren cong 8000...
python -u ui/server.py
pause
