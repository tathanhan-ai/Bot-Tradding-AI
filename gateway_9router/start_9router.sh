#!/bin/bash
cd "$(dirname "$0")"
echo "=========================================================="
echo "[9Router AI Gateway] Khoi dong cong 8039 cho Binance Bot"
echo "=========================================================="
if [ ! -d "node_modules/9router" ]; then
    echo "[9Router] Dang cai dat thu vien 9Router lan dau..."
    npm install --no-fund --no-audit
fi
export PORT=8039
export HOST=127.0.0.1
node run_9router.js
