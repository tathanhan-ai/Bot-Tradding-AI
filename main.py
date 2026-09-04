"""
Binance USD(S)-M Futures Automated Trading System - CLI Controller
Allows running backtesting, strategy comparisons, and live paper trading without API keys.
"""
import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Ensure utf-8 encoding on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from config.settings import (
    DEFAULT_RISK,
    DEFAULT_STRATEGY,
    DEFAULT_FEES,
    StrategyConfig,
    RiskConfig,
)
from data.fetcher import BinanceDataFetcher
from strategy.mtf_trend_atr import MTFTrendATRStrategy
from backtester.engine import BacktestEngine
from backtester.metrics import print_performance_report
from simulator.paper_trader import run_paper_trader
from tabulate import tabulate


def run_single_backtest(symbol: str = "BTCUSDT", total_candles: int = 2000, initial_balance: float = 1000.0, risk_pct: float = 0.015):
    risk_cfg = RiskConfig(
        initial_balance=initial_balance,
        risk_per_trade_pct=risk_pct,
        default_leverage=DEFAULT_RISK.default_leverage,
        max_leverage=DEFAULT_RISK.max_leverage
    )

    print(f"\n🚀 KHỞI CHẠY BACKTEST ĐỊNH LƯỢNG CHO {symbol.upper()}...")
    print(f"   Vốn ban đầu thiết lập: ${initial_balance:,.2f} USDT")
    print(f"   Đòn bẩy: {risk_cfg.default_leverage}x | Rủi ro mỗi lệnh: {risk_pct*100:.1f}% (${initial_balance * risk_pct:,.2f})")
    print(f"   Khung xu hướng: {DEFAULT_STRATEGY.trend_timeframe} (EMA {DEFAULT_STRATEGY.ema_trend_period})")
    print(f"   Khung vào lệnh: {DEFAULT_STRATEGY.entry_timeframe} (Donchian {DEFAULT_STRATEGY.donchian_period} + ATR {DEFAULT_STRATEGY.atr_period})\n")

    fetcher = BinanceDataFetcher()
    data_map = {
        DEFAULT_STRATEGY.trend_timeframe: fetcher.fetch_klines(
            symbol, DEFAULT_STRATEGY.trend_timeframe, total_candles=max(500, total_candles // 4)
        ),
        DEFAULT_STRATEGY.entry_timeframe: fetcher.fetch_klines(
            symbol, DEFAULT_STRATEGY.entry_timeframe, total_candles=total_candles
        ),
    }

    cfg = StrategyConfig(symbol=symbol)
    strategy = MTFTrendATRStrategy(cfg)
    engine = BacktestEngine(strategy=strategy, risk_config=risk_cfg, fees_config=DEFAULT_FEES)
    
    result = engine.run(data_map=data_map, symbol=symbol)
    return result


def run_multi_comparison(symbols: list[str] = ["BTCUSDT", "ETHUSDT", "SOLUSDT"], total_candles: int = 2000, initial_balance: float = 1000.0):
    print(f"\n📊 ĐANG SO SÁNH HIỆU NĂNG DANH MỤC VỚI SỐ VỐN ${initial_balance:,.2f} TRÊN {len(symbols)} CẶP TIỀN...")
    summary_rows = []

    for sym in symbols:
        try:
            res = run_single_backtest(sym, total_candles, initial_balance=initial_balance)
            m = res["metrics"]
            summary_rows.append([
                sym,
                f"${m['initial_balance']:,.2f}",
                f"${m['final_balance']:,.2f}",
                f"{m['net_profit']:+,.2f}",
                f"{m['total_return_pct']:+.2f}%",
                f"{m['win_rate']:.1f}%",
                f"{m['profit_factor']:.2f}",
                f"-{m['max_drawdown_pct']:.2f}%",
                f"{m['sharpe_ratio']:.2f}",
                m["total_trades"]
            ])
        except Exception as e:
            print(f"[Lỗi] Không thể backtest {sym}: {e}")

    headers = ["Cặp Coin", "Vốn Đầu", "Vốn Cuối", "Lãi/Lỗ ($)", "Tỷ Suất (%)", "Win Rate", "Profit Factor", "Max DD", "Sharpe", "Tổng Lệnh"]
    print("\n" + "=" * 90)
    print(f"       BẢNG TỔNG KẾT HIỆU NĂNG DANH MỤC (VỐN KHỞI ĐIỂM: ${initial_balance:,.2f})")
    print("=" * 90)
    print(tabulate(summary_rows, headers=headers, tablefmt="fancy_grid"))
    print("=" * 90 + "\n")


def optimize_parameters(symbol: str = "BTCUSDT", total_candles: int = 2000, initial_balance: float = 1000.0):
    """
    Scan multiple parameter combinations to find optimal settings
    """
    print(f"\n🔬 ĐANG QUÉT TỐI ƯU HÓA THAM SỐ CHO {symbol.upper()} (Vốn ${initial_balance:,.2f})...")
    fetcher = BinanceDataFetcher()
    data_map = {
        "1h": fetcher.fetch_klines(symbol, "1h", total_candles=max(500, total_candles // 4)),
        "15m": fetcher.fetch_klines(symbol, "15m", total_candles=total_candles),
    }

    rr_ratios = [1.5, 2.0, 2.5]
    atr_multipliers = [1.2, 1.5, 2.0]
    donchian_periods = [15, 20, 30]

    results = []
    print(f"Đang kiểm tra {len(rr_ratios) * len(atr_multipliers) * len(donchian_periods)} tổ hợp tham số...")

    for donchian in donchian_periods:
        for atr_m in atr_multipliers:
            for rr in rr_ratios:
                cfg = StrategyConfig(
                    symbol=symbol,
                    donchian_period=donchian,
                    atr_sl_multiplier=atr_m,
                    risk_reward_ratio=rr
                )
                strategy = MTFTrendATRStrategy(cfg)
                risk_cfg = RiskConfig(initial_balance=initial_balance)
                engine = BacktestEngine(strategy=strategy, risk_config=risk_cfg, fees_config=DEFAULT_FEES)
                df_entry = strategy.generate_signals(data_map)
                res = engine.run(data_map=data_map, symbol=symbol)
                m = res["metrics"]
                results.append({
                    "donchian": donchian,
                    "atr_m": atr_m,
                    "rr": rr,
                    "net_profit": m["net_profit"],
                    "return_pct": m["total_return_pct"],
                    "win_rate": m["win_rate"],
                    "profit_factor": m["profit_factor"],
                    "max_dd": m["max_drawdown_pct"],
                    "trades": m["total_trades"]
                })

    # Sort by Net Profit descending
    results.sort(key=lambda x: x["net_profit"], reverse=True)
    table_rows = []
    for r in results[:10]:
        table_rows.append([
            f"Donchian={r['donchian']}, ATR={r['atr_m']}, RR={r['rr']}",
            f"${r['net_profit']:+,.2f}",
            f"{r['return_pct']:+.2f}%",
            f"{r['win_rate']:.1f}%",
            f"{r['profit_factor']:.2f}",
            f"-{r['max_dd']:.2f}%",
            r["trades"]
        ])

    headers = ["Cấu hình (Donchian / ATR / RR)", "Lợi Nhuận ($)", "Tỷ Suất (%)", "Win Rate", "Profit Factor", "Max DD", "Lệnh"]
    print("\n" + "=" * 85)
    print(f"       TOP 10 CẤU HÌNH HIỆU QUẢ NHẤT CHO {symbol.upper()} (VỐN: ${initial_balance:,.2f})")
    print("=" * 85)
    print(tabulate(table_rows, headers=headers, tablefmt="fancy_grid"))
    print("=" * 85 + "\n")


def interactive_menu():
    current_balance = 1000.0  # Default USD
    candles_count = 2500

    while True:
        print("\n" + "=" * 60)
        print("🤖 HỆ THỐNG GIAO DỊCH TỰ ĐỘNG BINANCE FUTURES (MTF-ATR)")
        print(f"   [Cấu hình hiện tại: Vốn = ${current_balance:,.2f} | Dữ liệu = {candles_count} nến]")
        print("=" * 60)
        print("1. Chạy Backtest cặp Bitcoin (BTCUSDT)")
        print("2. Chạy Backtest cặp Ethereum (ETHUSDT)")
        print("3. Chạy Backtest cặp Solana (SOLUSDT)")
        print("4. So sánh hiệu năng danh mục cả 3 cặp (BTC, ETH, SOL)")
        print("5. 🔬 Quét tìm Bộ Tham Số Tối Ưu Lợi Nhuận (Optimizer)")
        print("6. Khởi động Giả lập Realtime Paper Trading (WebSocket)")
        print("7. 💰 Thay đổi Số Vốn Ban Đầu và Số Lượng Nến Kiểm Thử")
        print("8. Backtest cặp coin tùy chọn bất kỳ")
        print("0. Thoát")
        print("=" * 60)

        choice = input("Vui lòng chọn chức năng (0-8): ").strip()
        if choice == "1":
            run_single_backtest("BTCUSDT", candles_count, initial_balance=current_balance)
        elif choice == "2":
            run_single_backtest("ETHUSDT", candles_count, initial_balance=current_balance)
        elif choice == "3":
            run_single_backtest("SOLUSDT", candles_count, initial_balance=current_balance)
        elif choice == "4":
            run_multi_comparison(["BTCUSDT", "ETHUSDT", "SOLUSDT"], candles_count, initial_balance=current_balance)
        elif choice == "5":
            sym = input("Nhập cặp coin cần tối ưu (mặc định BTCUSDT): ").strip().upper() or "BTCUSDT"
            optimize_parameters(sym, candles_count, initial_balance=current_balance)
        elif choice == "6":
            print(f"\nKhởi động Paper Trader với số vốn ảo: ${current_balance:,.2f}...")
            run_paper_trader(initial_balance=current_balance)
        elif choice == "7":
            val = input(f"Nhập số vốn ban đầu cụ thể bằng USD (hiện tại ${current_balance:,.2f}): ").strip()
            if val:
                try:
                    current_balance = float(val.replace("$", "").replace(",", ""))
                    print(f"-> Đã cập nhật vốn ban đầu thành: ${current_balance:,.2f}")
                except ValueError:
                    print("Số tiền không hợp lệ.")
            c_val = input(f"Nhập số lượng nến kiểm thử (hiện tại {candles_count}, ví dụ: 3000, 5000): ").strip()
            if c_val:
                try:
                    candles_count = int(c_val)
                    print(f"-> Đã cập nhật số nến kiểm thử thành: {candles_count}")
                except ValueError:
                    print("Số nến không hợp lệ.")
        elif choice == "8":
            sym = input("Nhập mã cặp giao dịch Binance Futures (ví dụ: DOGEUSDT, BNBUSDT, NEARUSDT): ").strip().upper()
            if sym:
                run_single_backtest(sym, candles_count, initial_balance=current_balance)
        elif choice == "9":
            print(f"\n🌐 Đang mở UI Quản Trị Web Dashboard tại: http://127.0.0.1:8000 ...")
            import uvicorn
            uvicorn.run("ui.server:app", host="127.0.0.1", port=8000, reload=False)
        elif choice == "0":
            print("Tạm biệt!")
            break
        else:
            print("Lựa chọn không hợp lệ, vui lòng thử lại.")


def main():
    parser = argparse.ArgumentParser(description="Binance USD(S)-M Futures Algorithmic Trading System")
    parser.add_argument("--backtest", action="store_true", help="Run historical backtest")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading pair symbol (e.g. BTCUSDT)")
    parser.add_argument("--balance", type=float, default=1000.0, help="Initial balance in USD (e.g. 500, 1000, 5000)")
    parser.add_argument("--candles", type=int, default=2500, help="Number of candles to test (e.g. 2500, 5000)")
    parser.add_argument("--compare", action="store_true", help="Compare multiple top symbols")
    parser.add_argument("--optimize", action="store_true", help="Run grid search parameter optimization")
    parser.add_argument("--paper", action="store_true", help="Start real-time paper trading in terminal")
    parser.add_argument("--ui", action="store_true", help="Launch Web Admin Dashboard on http://localhost:8000")

    args = parser.parse_args()

    if args.ui:
        print(f"\n🌐 Đang mở UI Quản Trị Web Dashboard tại: http://127.0.0.1:8000 ...")
        import uvicorn
        uvicorn.run("ui.server:app", host="127.0.0.1", port=8000, reload=False)
    elif args.paper:
        run_paper_trader(initial_balance=args.balance)
    elif args.optimize:
        optimize_parameters(args.symbol.upper(), args.candles, initial_balance=args.balance)
    elif args.compare:
        run_multi_comparison(["BTCUSDT", "ETHUSDT", "SOLUSDT"], args.candles, initial_balance=args.balance)
    elif args.backtest:
        run_single_backtest(args.symbol.upper(), args.candles, initial_balance=args.balance)
    else:
        interactive_menu()


if __name__ == "__main__":
    main()
