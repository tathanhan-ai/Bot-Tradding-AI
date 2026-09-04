"""
Ultra-Low Latency Direct Binance Futures Multi-Channel WebSocket Engine
Connects directly to Binance USD-M Futures WebSocket via dedicated streams:
1. @bookTicker: Instant Best Bid, Best Ask, Spread (sub-10ms)
2. @aggTrade: Real-time executed trades with Buyer Maker flag for Order Flow CVD
3. @kline_1m: Real-time live forming candlestick updates
Includes auto-reconnect with exponential backoff and connection health heartbeats.
"""
import asyncio
import json
import time
from typing import Callable, Optional, Dict, Any, List
import websockets


class BinanceFuturesWebSocketEngine:
    def __init__(
        self,
        symbol: str = "BTCUSDT",
        on_book_ticker: Optional[Callable[[float, float, float], None]] = None,
        on_agg_trade: Optional[Callable[[float, float, bool, int], None]] = None,
        on_kline: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_latency_update: Optional[Callable[[float], None]] = None
    ):
        self.symbol = symbol.lower()
        self.on_book_ticker = on_book_ticker
        self.on_agg_trade = on_agg_trade
        self.on_kline = on_kline
        self.on_latency_update = on_latency_update

        self.is_running = False
        self._tasks: List[asyncio.Task] = []
        self.last_event_time = 0.0
        self.current_latency_ms = 0.0
        self.total_ticks_received = 0

    async def start(self):
        self.is_running = True
        self._tasks.append(asyncio.create_task(self._run_book_ticker_loop()))
        self._tasks.append(asyncio.create_task(self._run_agg_trade_loop()))
        self._tasks.append(asyncio.create_task(self._run_kline_loop()))

    async def stop(self):
        self.is_running = False
        for t in self._tasks:
            t.cancel()
        self._tasks.clear()

    async def _run_book_ticker_loop(self):
        url = f"wss://fstream.binance.com/ws/{self.symbol}@bookTicker"
        backoff = 1.0
        while self.is_running:
            try:
                print(f"[WebSocket Engine] ⚡ Kết nối BookTicker: {url}...", flush=True)
                async with websockets.connect(url, ping_interval=20, ping_timeout=10, max_size=2**22) as ws:
                    backoff = 1.0
                    print(f"🚀 [WebSocket Engine] BookTicker đã kết nối thành công cho {self.symbol.upper()}!", flush=True)
                    while self.is_running:
                        raw = await ws.recv()
                        data = json.loads(raw)
                        self.total_ticks_received += 1
                        event_time_ms = data.get("E", 0)
                        if event_time_ms > 0:
                            now_ms = time.time() * 1000.0
                            self.current_latency_ms = round(max(1.0, now_ms - event_time_ms), 1)
                            if self.on_latency_update and self.total_ticks_received % 25 == 0:
                                self.on_latency_update(self.current_latency_ms)

                        bid = float(data.get("b", 0.0))
                        ask = float(data.get("a", 0.0))
                        mid = (bid + ask) / 2.0 if (bid > 0 and ask > 0) else bid
                        if self.on_book_ticker and mid > 0:
                            self.on_book_ticker(bid, ask, mid)
            except asyncio.CancelledError:
                break
            except Exception as e:
                await asyncio.sleep(backoff)
                backoff = min(backoff * 1.5, 10.0)

    async def _run_agg_trade_loop(self):
        url = f"wss://fstream.binance.com/market/ws/{self.symbol}@aggTrade"
        backoff = 1.0
        while self.is_running:
            try:
                print(f"[WebSocket Engine] 🌊 Kết nối Order Flow aggTrade: {url}...", flush=True)
                async with websockets.connect(url, ping_interval=20, ping_timeout=10, max_size=2**22) as ws:
                    backoff = 1.0
                    print(f"🚀 [WebSocket Engine] Order Flow aggTrade đã kết nối cho {self.symbol.upper()}!", flush=True)
                    while self.is_running:
                        raw = await ws.recv()
                        data = json.loads(raw)
                        if data.get("e") == "aggTrade":
                            price = float(data.get("p", 0.0))
                            qty = float(data.get("q", 0.0))
                            is_buyer_maker = bool(data.get("m", False))
                            trade_time = int(data.get("T", 0))
                            if self.on_agg_trade and price > 0 and qty > 0:
                                self.on_agg_trade(price, qty, is_buyer_maker, trade_time)
            except asyncio.CancelledError:
                break
            except Exception as e:
                await asyncio.sleep(backoff)
                backoff = min(backoff * 1.5, 10.0)

    async def _run_kline_loop(self):
        url = f"wss://fstream.binance.com/market/ws/{self.symbol}@kline_1m"
        backoff = 1.0
        while self.is_running:
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=10, max_size=2**22) as ws:
                    backoff = 1.0
                    while self.is_running:
                        raw = await ws.recv()
                        data = json.loads(raw)
                        k = data.get("k", {})
                        if self.on_kline and k:
                            kline_dict = {
                                "time": int(k.get("t", 0) / 1000),
                                "open": float(k.get("o", 0.0)),
                                "high": float(k.get("h", 0.0)),
                                "low": float(k.get("l", 0.0)),
                                "close": float(k.get("c", 0.0)),
                                "volume": float(k.get("v", 0.0)),
                                "is_closed": bool(k.get("x", False))
                            }
                            self.on_kline(kline_dict)
            except asyncio.CancelledError:
                break
            except Exception as e:
                await asyncio.sleep(backoff)
                backoff = min(backoff * 1.5, 10.0)
