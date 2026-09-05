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
import math
import time
import urllib.request
from typing import Callable, Optional, Dict, Any, List
import websockets


def normalize_depth_message(data: Dict[str, Any]) -> Optional[tuple[List[List[float]], List[List[float]], int]]:
    """Accept only an exchange-supplied, complete top-20 book snapshot."""
    raw_bids = data.get("b", data.get("bids", []))
    raw_asks = data.get("a", data.get("asks", []))
    if len(raw_bids) < 20 or len(raw_asks) < 20:
        return None
    try:
        bids = [[float(price), float(quantity)] for price, quantity, *_ in raw_bids[:20]]
        asks = [[float(price), float(quantity)] for price, quantity, *_ in raw_asks[:20]]
    except (TypeError, ValueError):
        return None
    if any(price <= 0 or quantity <= 0 or not math.isfinite(price) or not math.isfinite(quantity) for price, quantity in bids + asks):
        return None
    if bids[0][0] >= asks[0][0] or any(a[0] <= b[0] for a, b in zip(bids, bids[1:])) or any(a[0] >= b[0] for a, b in zip(asks, asks[1:])):
        return None
    return bids, asks, int(data.get("E", data.get("T", 0)))


def normalize_kline_message(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Normalise Binance's OHLCV payload without inventing tick-volume bars."""
    k = data.get("k", {})
    if not k:
        return None
    try:
        return {
            "timeframe": str(k["i"]),
            "time": int(k["t"]) // 1000,
            "open": float(k["o"]),
            "high": float(k["h"]),
            "low": float(k["l"]),
            "close": float(k["c"]),
            "volume": float(k["v"]),
            "quote_volume": float(k["q"]),
            "is_closed": bool(k["x"]),
            "event_time": int(data.get("E", 0)),
        }
    except (KeyError, TypeError, ValueError):
        return None


class BinanceFuturesWebSocketEngine:
    def __init__(
        self,
        symbol: str = "BTCUSDT",
        on_book_ticker: Optional[Callable[[float, float, float], None]] = None,
        on_depth: Optional[Callable[[List[List[float]], List[List[float]], int], None]] = None,
        on_agg_trade: Optional[Callable[[float, float, bool, int], None]] = None,
        on_kline: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_latency_update: Optional[Callable[[float], None]] = None,
        on_spot_ticker: Optional[Callable[[float], None]] = None,
        kline_intervals: Optional[List[str]] = None,
        is_testnet: bool = False,
    ):
        self.symbol = symbol.lower()
        self.is_testnet = is_testnet
        self.base_url = "wss://stream.binancefuture.com" if is_testnet else "wss://fstream.binance.com"
        self.on_book_ticker = on_book_ticker
        self.on_depth = on_depth
        self.on_agg_trade = on_agg_trade
        self.on_kline = on_kline
        self.on_latency_update = on_latency_update
        self.on_spot_ticker = on_spot_ticker
        self.kline_intervals = kline_intervals or ["1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d", "1w", "1M"]
        self.last_message_at: Dict[str, float] = {}
        self.last_error: Dict[str, str] = {}
        self.last_agg_id = -1
        self.last_depth_id = -1
        self.clock_offset_ms = None
        self.clock_synced_at = 0.0

        self.is_running = False
        self._tasks: List[asyncio.Task] = []
        self.last_event_time = 0.0
        self.last_book_event_time = 0.0
        self.current_latency_ms = 0.0
        self.total_ticks_received = 0

    async def start(self):
        self.is_running = True
        self._tasks.append(asyncio.create_task(self._clock_sync_loop()))
        self._tasks.append(asyncio.create_task(self._run_book_ticker_loop()))
        self._tasks.append(asyncio.create_task(self._run_depth_loop()))
        self._tasks.append(asyncio.create_task(self._run_agg_trade_loop()))
        self._tasks.append(asyncio.create_task(self._run_kline_loop()))
        if self.on_spot_ticker:
            self._tasks.append(asyncio.create_task(self._run_spot_ticker_loop()))

    async def stop(self):
        self.is_running = False
        for t in self._tasks:
            t.cancel()
        self._tasks.clear()

    async def _clock_sync_loop(self):
        url = ("https://demo-fapi.binance.com" if self.is_testnet else "https://fapi.binance.com") + "/fapi/v1/time"
        def sync():
            start = time.time() * 1000
            with urllib.request.urlopen(url, timeout=4) as response:
                server = json.loads(response.read())["serverTime"]
            end = time.time() * 1000
            return server - (start + end) / 2
        while self.is_running:
            try:
                self.clock_offset_ms = await asyncio.to_thread(sync)
                self.clock_synced_at = time.time()
            except Exception as exc:
                self.last_error["clock"] = type(exc).__name__
            await asyncio.sleep(30)

    def fresh_event(self, timestamp):
        if self.clock_offset_ms is None or time.time() - self.clock_synced_at > 90:
            return False
        lag = time.time() * 1000 + self.clock_offset_ms - float(timestamp or 0)
        return -1000 <= lag <= 3000

    async def _run_book_ticker_loop(self):
        url = f"{self.base_url}{'' if self.is_testnet else '/public'}/ws/{self.symbol}@bookTicker"
        backoff = 1.0
        while self.is_running:
            try:
                print(f"[WebSocket Engine] ⚡ Kết nối BookTicker: {url}...", flush=True)
                async with websockets.connect(url, ping_interval=20, ping_timeout=10, max_size=2**22) as ws:
                    backoff = 1.0
                    print(f"🚀 [WebSocket Engine] BookTicker đã kết nối thành công cho {self.symbol.upper()}!", flush=True)
                    while self.is_running:
                        raw = await asyncio.wait_for(ws.recv(), 15)
                        data = json.loads(raw)
                        if data.get("s") != self.symbol.upper() or data.get("st", 1) != 1:
                            continue
                        if not self.fresh_event(data.get("E")):
                            continue
                        self.last_message_at["book_ticker"] = time.time()
                        self.total_ticks_received += 1
                        event_time_ms = data.get("E", 0)
                        if event_time_ms > 0:
                            self.last_book_event_time = event_time_ms / 1000.0
                            now_ms = time.time() * 1000.0
                            self.current_latency_ms = round(now_ms + self.clock_offset_ms - event_time_ms, 1)
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
                self.last_error["book_ticker"] = f"{type(e).__name__}: {e}"
                await asyncio.sleep(backoff)
                backoff = min(backoff * 1.5, 10.0)

    async def _run_spot_ticker_loop(self):
        url = f"wss://stream.binance.com:9443/ws/{self.symbol}@bookTicker"
        backoff = 1.0
        while self.is_running:
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=10, max_size=2**20) as ws:
                    backoff = 1.0
                    while self.is_running:
                        raw = await asyncio.wait_for(ws.recv(), 15)
                        data = json.loads(raw)
                        bid = float(data.get("b", 0.0))
                        ask = float(data.get("a", 0.0))
                        mid = (bid + ask) / 2.0 if (bid > 0 and ask > 0) else bid
                        if self.on_spot_ticker and mid > 0:
                            self.on_spot_ticker(mid)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.last_error["spot_ticker"] = f"{type(e).__name__}: {e}"
                await asyncio.sleep(backoff)
                backoff = min(backoff * 1.5, 10.0)

    async def _run_agg_trade_loop(self):
        url = f"{self.base_url}{'' if self.is_testnet else '/market'}/ws/{self.symbol}@aggTrade"
        backoff = 1.0
        while self.is_running:
            try:
                print(f"[WebSocket Engine] 🌊 Kết nối Order Flow aggTrade: {url}...", flush=True)
                async with websockets.connect(url, ping_interval=20, ping_timeout=10, max_size=2**22) as ws:
                    backoff = 1.0
                    print(f"🚀 [WebSocket Engine] Order Flow aggTrade đã kết nối cho {self.symbol.upper()}!", flush=True)
                    while self.is_running:
                        raw = await asyncio.wait_for(ws.recv(), 15)
                        data = json.loads(raw)
                        if data.get("e") == "aggTrade" and data.get("s") == self.symbol.upper() and data.get("st", 1) == 1:
                            if not self.fresh_event(data.get("E")):
                                continue
                            agg_id = int(data.get("a", -1))
                            if agg_id <= self.last_agg_id:
                                continue
                            self.last_agg_id = agg_id
                            self.last_message_at["agg_trade"] = time.time()
                            price = float(data.get("p", 0.0))
                            qty = float(data.get("q", 0.0))
                            is_buyer_maker = bool(data.get("m", False))
                            trade_time = int(data.get("T", 0))
                            if self.on_agg_trade and price > 0 and qty > 0:
                                self.on_agg_trade(price, qty, is_buyer_maker, trade_time)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.last_error["agg_trade"] = f"{type(e).__name__}: {e}"
                await asyncio.sleep(backoff)
                backoff = min(backoff * 1.5, 10.0)

    async def _run_depth_loop(self):
        # Binance partial-book stream sends the actual best 20 price levels; it is
        # intentionally kept separate from bookTicker, which only has level one.
        url = f"{self.base_url}{'' if self.is_testnet else '/public'}/ws/{self.symbol}@depth20@100ms"
        backoff = 1.0
        while self.is_running:
            try:
                print(f"[WebSocket Engine] 📚 Kết nối L2 Depth 20: {url}...", flush=True)
                async with websockets.connect(url, ping_interval=20, ping_timeout=10, max_size=2**22) as ws:
                    backoff = 1.0
                    while self.is_running:
                        payload = json.loads(await asyncio.wait_for(ws.recv(), 15))
                        if payload.get("s") != self.symbol.upper() or payload.get("st", 1) != 1:
                            continue
                        update_id = int(payload.get("u", -1))
                        if not self.fresh_event(payload.get("E")) or update_id <= self.last_depth_id:
                            continue
                        normalized = normalize_depth_message(payload)
                        if normalized and self.on_depth:
                            self.last_depth_id = update_id
                            self.last_message_at["depth"] = time.time()
                            self.on_depth(*normalized)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.last_error["depth"] = f"{type(e).__name__}: {e}"
                await asyncio.sleep(backoff)
                backoff = min(backoff * 1.5, 10.0)

    async def _run_kline_loop(self):
        streams = "/".join(f"{self.symbol}@kline_{interval}" for interval in self.kline_intervals)
        url = f"{self.base_url}{'' if self.is_testnet else '/market'}/stream?streams={streams}"
        backoff = 1.0
        while self.is_running:
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=10, max_size=2**22) as ws:
                    backoff = 1.0
                    while self.is_running:
                        message = json.loads(await asyncio.wait_for(ws.recv(), 15))
                        payload = message.get("data", message)
                        if payload.get("s") != self.symbol.upper() or payload.get("st", 1) != 1:
                            continue
                        kline = normalize_kline_message(payload)
                        if self.on_kline and kline and self.fresh_event(payload.get("E")):
                            self.last_message_at["kline"] = time.time()
                            self.on_kline(kline)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.last_error["kline"] = f"{type(e).__name__}: {e}"
                await asyncio.sleep(backoff)
                backoff = min(backoff * 1.5, 10.0)
