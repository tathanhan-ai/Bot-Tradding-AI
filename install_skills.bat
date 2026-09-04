@echo off
chcp 65001 >nul
echo ==================================================================
echo [1-CLICK INSTALLER] Cai dat Ky nang Giao dich & 9Router AI Gateway
echo ==================================================================
cd /d "%~dp0"
python install_trading_skills.py

if exist "gateway_9router" (
    echo.
    echo [9Router] Cai dat thu vien 9Router AI Gateway cho cong 8039...
    cd /d "%~dp0gateway_9router"
    call npm install --no-fund --no-audit
    cd /d "%~dp0"
)

echo.
echo ==================================================================
echo HOAN TAT! Ban co the chay run_bot.bat de bat dau.
echo ==================================================================
pause
