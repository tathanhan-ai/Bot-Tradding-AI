#!/bin/bash
echo "=================================================================="
echo "[PORTABLE LAUNCHER] Khoi dong Binance Futures AI Bot"
echo "=================================================================="
cd "$(dirname "$0")"

if [ -f "gateway_9router/run_9router.js" ]; then
    echo "[9Router] Dang khoi dong AI Gateway tren cong 8039..."
    (cd gateway_9router && export PORT=8039 && export HOST=127.0.0.1 && node run_9router.js > /dev/null 2>&1 &)
fi

echo "[UI Server] Khoi dong Giao dien Bot tren cong 8000..."
python3 -u ui/server.py
