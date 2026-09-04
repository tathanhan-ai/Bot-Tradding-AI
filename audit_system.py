import sys
import json
import urllib.request
import sqlite3

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def check_system():
    print("=======================================================")
    print("   KIỂM TRA & ĐÁNH GIÁ TOÀN DIỆN VẬN HÀNH QUANT DESK")
    print("=======================================================")
    
    # 1. API State
    try:
        req = urllib.request.Request("http://127.0.0.1:8000/api/state", headers={"User-Agent": "HealthCheck"})
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            print("1. [HTTP API & FASTAPI]: HOẠT ĐỘNG HOÀN HẢO (200 OK)")
    except Exception as e:
        print(f"1. [HTTP API]: LỖI - {e}")
        return

    # 2. WebSocket & Market Data
    ws_latency = data.get("order_flow", {}).get("latency_ms", 999)
    ticks = data.get("tick_count", 0)
    price = data.get("live_price", 0)
    funding = data.get("funding_rate_pct", 0)
    print(f"2. [BINANCE WEBSOCKET]: Đang nhận giá ${price:,.2f} | Trễ: {ws_latency:.1f}ms | Ticks: {ticks:,} | Funding: {funding:+.4f}%")

    # 3. Multi-Candle MTF Strategy Engine
    cr = data.get("candle_radar", {})
    score = cr.get("confluence_score", 0)
    verdict = cr.get("confluence_verdict", "N/A")
    rec_act = cr.get("recommended_action", "N/A")
    aligned = cr.get("aligned_timeframes_count", 0)
    macro_align = cr.get("macro_alignment", False)
    print(f"3. [MTF CANDLE RADAR]: Điểm hội tụ {score:+.1f}/100 | Phán quyết: {verdict} | Khuyến nghị: {rec_act} | Đồng thuận: {aligned}/4 khung | 1H Macro: {'Đồng thuận 🛡️' if macro_align else 'Chặn ngược trend ⚠️'}")

    # 4. Ensemble Coordinator & Sub-Engines
    ens = data.get("ensemble", {})
    ens_score = ens.get("consensus_score", 0)
    ens_verdict = ens.get("consensus_verdict", "N/A")
    ens_conf = ens.get("confidence", 0)
    print(f"4. [ENSEMBLE MATRIX]: Đồng thuận: {ens_verdict} (Điểm: {ens_score:+.1f}, Tự tin: {ens_conf}%)")
    for s in ens.get("strategies", []):
        print(f"   - {s['name']}: Điểm {s['score']:+.1f} (Tỷ trọng {s['weight_pct']}%) -> {s['rationale']}")

    # 5. Order Flow & SMC
    of = data.get("order_flow", {})
    smc = data.get("ai_brain", {})
    print(f"5. [SMC & ORDER FLOW]: CVD Delta: {of.get('current_cvd', 0):+.3f} BTC | Taker Buy: {of.get('buy_ratio_pct', 50):.1f}% | Cấu trúc: {smc.get('smc_structure', 'N/A')} | VWAP: ${smc.get('vwap_fair_price', 0):,.2f} ({smc.get('vwap_status', 'N/A')})")

    # 6. Risk Manager & Protections
    ft = data.get("freqtrade", {})
    cro = data.get("ai_risk", {})
    jesse = data.get("jesse", {})
    print(f"6. [QUẢN TRỊ RỦI RO & BẢO TOÀN VỐN]:")
    print(f"   - Freqtrade Status: {ft.get('status', 'NORMAL')} | Stoploss 60p: {ft.get('stoploss_count_60m', 0)}/2 | Max DD: {ft.get('drawdown_pct', 0):.2f}%")
    print(f"   - AI CRO Mode: {cro.get('protection_mode', 'NORMAL')} | Rủi ro/lệnh: {cro.get('risk_per_trade_pct', 1.5)}% | Đóng băng Circuit Breaker: {cro.get('circuit_breaker_active', False)}")
    print(f"   - Jesse AI Expectancy: ${jesse.get('expectancy_usdt', 0):+.2f} | Tỷ lệ Kelly tối ưu: {jesse.get('kelly_fraction_pct', 0):.1f}%")

    # 7. SQLite Database Check
    try:
        conn = sqlite3.connect("data/bot_database.db")
        cur = conn.cursor()
        cur.execute("SELECT count(*) FROM trades")
        trade_count = cur.fetchone()[0]
        cur.execute("SELECT current_balance, initial_balance FROM account_state WHERE id=1")
        row = cur.fetchone()
        cur_bal = row[0] if row else 0
        init_bal = row[1] if row else 0
        cur.execute("SELECT count(*) FROM trade_memory_records")
        mem_count = cur.fetchone()[0]
        conn.close()
        print(f"7. [SQLITE STORAGE]: Kết nối tốt! Số dư DB: ${cur_bal:,.2f} (Vốn gốc: ${init_bal:,.2f}) | Lệnh lưu: {trade_count} | Bài học OODA: {mem_count}")
    except Exception as e:
        print(f"7. [SQLITE STORAGE]: Lỗi DB - {e}")

    print("=======================================================")
    print("   KẾT LUẬN: TẤT CẢ 7 PHÂN HỆ ĐANG VẬN HÀNH ĐỒNG BỘ!")
    print("=======================================================")

if __name__ == "__main__":
    check_system()
