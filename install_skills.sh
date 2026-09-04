#!/bin/bash
echo "=================================================================="
echo "[1-CLICK INSTALLER] Cai dat Ky nang Giao dich & 9Router AI Gateway"
echo "=================================================================="
cd "$(dirname "$0")"
python3 install_trading_skills.py

if [ -d "gateway_9router" ]; then
    echo ""
    echo "[9Router] Cai dat thu vien 9Router AI Gateway cho cong 8039..."
    cd gateway_9router
    npm install --no-fund --no-audit
    cd ..
fi

echo ""
echo "=================================================================="
echo "HOAN TAT! Ban co the chay bash run_bot.sh de bat dau."
echo "=================================================================="
