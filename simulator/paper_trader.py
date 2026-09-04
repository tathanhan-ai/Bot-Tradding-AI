"""
Real-Time Paper Trading Simulator via Binance Public WebSocket
Simulates live futures trading in real-time with zero API keys required.
"""
import asyncio
import json
import logging
import sys
from datetime import datetime
import time
from typing import Optional

# Ensure utf-8 encoding on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import pandas as pd
import websockets

from config.settings import (
    DEFAULT_RISK,
    DEFAULT_STRATEGY,
    DEFAULT_SYSTEM,
    RiskConfig,
    StrategyConfig,
)
from data.fetcher import BinanceDataFetcher
from risk.risk_manager import FuturesRiskManager, PositionProposal
from strategy.mtf_trend_atr import MTFTrendATRStrategy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class PaperTrader:
    def __init__(
        self,
        strategy_config: StrategyConfig = DEFAULT_STRATEGY,
        risk_config: RiskConfig = DEFAULT_RISK,
    ):
        self.strategy_config = strategy_config
        self.risk_config = risk_config
        self.risk_manager = FuturesRiskManager(risk_config)
        self.strategy = MTFTrendATRStrategy(strategy_config)
        self.fetcher = BinanceDataFetcher()

        # In-memory buffer of recent candles
        self.data_map = {}
        self.current_position: Optional[dict] = None
        self.virtual_balance = risk_config.initial_balance
        self.total_trades_count = 0

    def initialize_history(self):
        """Fetch initial historical data so indicators can calculate immediately"""
        sym = self.strategy_config.symbol
        tf_trend = self.strategy_config.trend_timeframe
        tf_entry = self.strategy_config.entry_timeframe

        print(f"[PaperTrader] Khởi tạo lịch sử nến cho {sym} ({tf_trend} & {tf_entry})...", flush=True)
        self.data_map[tf_trend] = self.fetcher.fetch_klines(sym, tf_trend, total_candles=300)
        self.data_map[tf_entry] = self.fetcher.fetch_klines(sym, tf_entry, total_candles=500)
        print("[PaperTrader] Đã nạp xong bộ đệm lịch sử. Sẵn sàng kết nối WebSocket.", flush=True)

    def on_candle_closed(self, candle: dict):
        """Called when an entry timeframe candle closes"""
        sym = self.strategy_config.symbol
        tf_entry = self.strategy_config.entry_timeframe
        
        # Append new closed candle to DataFrame
        ts = pd.to_datetime(candle["t"], unit="ms")
        new_row = pd.DataFrame([{
            "open": float(candle["o"]),
            "high": float(candle["h"]),
            "low": float(candle["l"]),
            "close": float(candle["c"]),
            "volume": float(candle["v"]),
            "quote_volume": float(candle["q"])
        }], index=[ts])

        self.data_map[tf_entry] = pd.concat([self.data_map[tf_entry], new_row])
        # Keep buffer limited to 600 candles to save memory
        if len(self.data_map[tf_entry]) > 600:
            self.data_map[tf_entry] = self.data_map[tf_entry].iloc[-600:]

        # Run strategy signal generation
        df_signals = self.strategy.generate_signals(self.data_map)
        latest = df_signals.iloc[-1]
        sig = int(latest["signal"])
        close_price = latest["close"]
        sl = latest["stop_loss"]
        tp = latest["take_profit"]

        print(f"\n[PaperTrader | {ts.strftime('%H:%M:%S')}] Nến {tf_entry} đóng cửa: {close_price:,.2f} USDT | RSI: {latest['rsi']:.1f} | ATR: {latest['atr']:.2f}")

        # If already in position, check trailing / SL / TP
        if self.current_position:
            self.check_position_update(close_price, latest["high"], latest["low"], latest["atr"])

        # If no position and signal triggered, open paper position
        if not self.current_position and sig in (1, -1):
            proposal = self.risk_manager.evaluate_order(
                symbol=sym,
                direction=sig,
                entry_price=close_price,
                stop_loss=sl,
                take_profit=tp,
                leverage=self.risk_config.default_leverage
            )

            if proposal.approved:
                side_str = "LONG 🟢" if sig == 1 else "SHORT 🔴"
                print("=" * 60)
                print(f"🚀 [PAPER TRADE ĐƯỢC MỞ] {side_str} {sym}")
                print(f"   Giá vào (Entry): {close_price:,.2f} USDT")
                print(f"   Khối lượng (Units): {proposal.units} {sym.replace('USDT', '')} (${proposal.notional_value:,.2f})")
                print(f"   Ký quỹ (Margin 3x): ${proposal.required_margin:,.2f}")
                print(f"   Cắt lỗ (Stop Loss): {sl:,.2f} USDT (Rủi ro: ${proposal.risk_amount:,.2f})")
                print(f"   Chốt lời (Take Profit): {tp:,.2f} USDT")
                print(f"   Ước tính giá thanh lý: {proposal.est_liquidation_price:,.2f} USDT")
                print("=" * 60)

                self.current_position = {
                    "direction": sig,
                    "entry_price": close_price,
                    "units": proposal.units,
                    "notional": proposal.notional_value,
                    "stop_loss": sl,
                    "take_profit": tp,
                    "initial_risk": abs(close_price - sl),
                    "peak_price": close_price,
                    "entry_time": ts
                }

    def check_position_update(self, current_price: float, high: float, low: float, atr: float):
        pos = self.current_position
        if not pos:
            return

        direction = pos["direction"]
        exit_price = None
        exit_reason = None

        if direction == 1:  # Long
            if high > pos["peak_price"]:
                pos["peak_price"] = high
                # Move to breakeven if reached 1R
                if (pos["peak_price"] - pos["entry_price"]) >= pos["initial_risk"]:
                    new_sl = max(pos["stop_loss"], pos["entry_price"])
                    pos["stop_loss"] = max(new_sl, pos["peak_price"] - (1.0 * atr))

            if high >= pos["take_profit"]:
                exit_price = pos["take_profit"]
                exit_reason = "TAKE_PROFIT 🎯"
            elif low <= pos["stop_loss"]:
                exit_price = pos["stop_loss"]
                exit_reason = "STOP_LOSS 🛑" if pos["stop_loss"] < pos["entry_price"] else "TRAILING_STOP 🛡️"

        else:  # Short
            if low < pos["peak_price"]:
                pos["peak_price"] = low
                if (pos["entry_price"] - pos["peak_price"]) >= pos["initial_risk"]:
                    new_sl = min(pos["stop_loss"], pos["entry_price"])
                    pos["stop_loss"] = min(new_sl, pos["peak_price"] + (1.0 * atr))

            if low <= pos["take_profit"]:
                exit_price = pos["take_profit"]
                exit_reason = "TAKE_PROFIT 🎯"
            elif high >= pos["stop_loss"]:
                exit_price = pos["stop_loss"]
                exit_reason = "STOP_LOSS 🛑" if pos["stop_loss"] > pos["entry_price"] else "TRAILING_STOP 🛡️"

        if exit_price is not None:
            pnl = (exit_price - pos["entry_price"]) * pos["units"] * direction
            # Deduct standard fees
            fees = (pos["notional"] + (exit_price * pos["units"])) * 0.0005
            net_pnl = pnl - fees
            self.virtual_balance += net_pnl
            self.total_trades_count += 1
            self.risk_manager.update_balance(self.virtual_balance)

            print("=" * 60)
            print(f"🏁 [PAPER TRADE ĐÃ ĐÓNG] {exit_reason}")
            print(f"   Giá thoát: {exit_price:,.2f} USDT")
            print(f"   PnL ròng: {net_pnl:+,.2f} USDT")
            print(f"   Số dư tài khoản ảo hiện tại: ${self.virtual_balance:,.2f}")
            print("=" * 60)

            self.current_position = None
        else:
            # Print unrealized PnL heartbeat
            unrealized = (current_price - pos["entry_price"]) * pos["units"] * direction
            print(f"   [Vị thế đang mở] Giá hiện tại: {current_price:,.2f} | SL: {pos['stop_loss']:,.2f} | TP: {pos['take_profit']:,.2f} | PnL tạm tính: {unrealized:+,.2f} USDT")

    async def start_streaming(self):
        """Connect to Binance Futures WebSocket Kline stream"""
        self.initialize_history()
        sym = self.strategy_config.symbol.lower()
        tf = self.strategy_config.entry_timeframe
        ws_url = f"{DEFAULT_SYSTEM.binance_ws_url}/{sym}@kline_{tf}"

        print(f"\n[WebSocket] Đang kết nối tới Binance Futures Public Stream: {ws_url}", flush=True)
        print("[Thông báo] Chế độ Paper Trading đang chạy (Theo dõi giá thời gian thực). Bấm Ctrl+C để dừng.\n", flush=True)

        last_tick_time = 0.0
        while True:
            try:
                async with websockets.connect(ws_url, ping_interval=20, ping_timeout=20) as ws:
                    print("[WebSocket] Kết nối THÀNH CÔNG! Đang nhận dữ liệu trực tiếp...", flush=True)
                    async for message in ws:
                        event = json.loads(message)
                        if "k" in event:
                            kline = event["k"]
                            is_closed = kline["x"]
                            now = time.time()
                            if is_closed:
                                self.on_candle_closed(kline)
                            elif now - last_tick_time >= 3.0:
                                last_tick_time = now
                                curr_p = float(kline["c"])
                                high_p = float(kline["h"])
                                low_p = float(kline["l"])
                                vol = float(kline["v"])
                                print(f"⚡ [Binance Live Tick] {sym.upper()} Giá: ${curr_p:,.2f} | Cao: ${high_p:,.2f} | Thấp: ${low_p:,.2f} | KL: {vol:,.2f}", flush=True)
            except (websockets.ConnectionClosed, Exception) as e:
                print(f"[WebSocket] Mất kết nối ({e}), đang tự động kết nối lại sau 5s...", flush=True)
                await asyncio.sleep(5)


def run_paper_trader(initial_balance: float = 1000.0):
    risk_cfg = RiskConfig(initial_balance=initial_balance)
    trader = PaperTrader(risk_config=risk_cfg)
    try:
        asyncio.run(trader.start_streaming())
    except KeyboardInterrupt:
        print("\n[PaperTrader] Đã dừng giả lập Paper Trading.")


if __name__ == "__main__":
    run_paper_trader()
