@echo off
title 9Router AI Gateway (Port 8039)
cd /d "%~dp0"
echo ==========================================================
echo [9Router AI Gateway] Khoi dong cong 8039 cho Binance Bot
echo ==========================================================
if not exist "node_modules\9router" (
    echo [9Router] Dang cai dat thu vien 9Router lan dau...
    npm install --no-fund --no-audit
)
set PORT=8039
set HOST=127.0.0.1
node run_9router.js
pause
