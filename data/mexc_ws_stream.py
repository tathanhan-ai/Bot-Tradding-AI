"""Public MEXC Contract stream used only for paper-market parity."""
import asyncio
import json
import math
import time
from typing import Any, Callable, Dict, List, Optional

import websockets


MEXC_INTERVALS = {
    "1m": "Min1", "5m": "Min5", "15m": "Min15", "30m": "Min30",
    "1h": "Min60", "4h": "Hour4", "1d": "Day1", "1w": "Week1", "1M": "Month1",
}


def _event_ms(value: Any) -> int:
    try:
        timestamp = int(float(value))
        return timestamp * 1000 if 0 < timestamp < 10_000_000_000 else timestamp
    except (TypeError, ValueError, OverflowError):
        return 0


def normalize_mexc_depth(message: Dict[str, Any]) -> Optional[tuple[List[List[float]], List[List[float]], int]]:
    """Accept MEXC's complete 20-level public-book payload, never deltas."""
    data = message.get("data", message)
    if not isinstance(data, dict):
        return None
    raw_bids, raw_asks = data.get("bids", []), data.get("asks", [])
    if len(raw_bids) < 20 or len(raw_asks) < 20:
        return None
    try:
        bids = [[float(level[0]), float(level[1])] for level in raw_bids[:20]]
        asks = [[float(level[0]), float(level[1])] for level in raw_asks[:20]]
    except (TypeError, ValueError, IndexError):
        return None
    if (any(not math.isfinite(price) or not math.isfinite(qty) or price <= 0 or qty <= 0 for price, qty in bids + asks)
            or bids[0][0] >= asks[0][0]
            or any(a[0] <= b[0] for a, b in zip(bids, bids[1:]))
            or any(a[0] >= b[0] for a, b in zip(asks, asks[1:]))):
        return None
    event_time = _event_ms(data.get("timestamp") or message.get("ts") or message.get("timestamp"))
    return bids, asks, event_time


def normalize_mexc_deal(message: Dict[str, Any]) -> Optional[tuple[float, float, bool, int]]:
    data = message.get("data", message)
    if isinstance(data, list):
        data = data[-1] if data else {}
    if not isinstance(data, dict):
        return None
    try:
        price, qty, side = float(data["p"]), float(data["v"]), int(data["T"])
    except (KeyError, TypeError, ValueError):
        return None
    event_time = _event_ms(data.get("t") or message.get("ts"))
    if price <= 0 or qty <= 0 or side not in (1, 2) or event_time <= 0:
        return None
    # MEXC T=2 is an aggressive sell, the equivalent of Binance buyer-maker.
    return price, qty, side == 2, event_time


def normalize_mexc_kline(message: Dict[str, Any], timeframe: str) -> Optional[Dict[str, Any]]:
    data = message.get("data", message)
    if not isinstance(data, dict):
        return None
    try:
        row = {
            "timeframe": timeframe,
            "time": _event_ms(data["t"]) // 1000,
            "open": float(data["o"]), "high": float(data["h"]),
            "low": float(data["l"]), "close": float(data["c"]),
            "volume": float(data["v"]), "quote_volume": float(data.get("a", data.get("q", 0))),
        }
    except (KeyError, TypeError, ValueError):
        return None
    if (row["time"] <= 0 or any(not math.isfinite(row[key]) or row[key] < 0 for key in ("open", "high", "low", "close", "volume", "quote_volume"))
            or row["open"] <= 0 or row["high"] < max(row["open"], row["close"]) or row["low"] > min(row["open"], row["close"])):
        return None
    return row


class MEXCFuturesWebSocketEngine:
    """One public connection for L2, deals and OHLCV; MEXC execution stays paper-only."""

    base_url = "wss://contract.mexc.com/edge"

    def __init__(self, symbol: str, on_book_ticker: Optional[Callable] = None,
                 on_depth: Optional[Callable] = None, on_agg_trade: Optional[Callable] = None,
                 on_kline: Optional[Callable] = None, on_latency_update: Optional[Callable] = None,
                 base_url: Optional[str] = None):
        self.symbol = symbol.replace("USDT", "_USDT") if "_" not in symbol else symbol
        self.base_url = (base_url or self.base_url).rstrip("/")
        self.on_book_ticker, self.on_depth = on_book_ticker, on_depth
        self.on_agg_trade, self.on_kline, self.on_latency_update = on_agg_trade, on_kline, on_latency_update
        self.is_testnet = False
        self.clock_offset_ms = 0.0
        self.last_book_event_time = 0.0
        self.last_message_at: Dict[str, float] = {}
        self.last_error: Dict[str, str] = {}
        self.total_ticks_received = 0
        self.current_latency_ms = 0.0
        self.is_running = False
        self._task: Optional[asyncio.Task] = None
        self._forming: Dict[str, Dict[str, Any]] = {}

    async def start(self):
        if self.is_running:
            return
        self.is_running = True
        self._task = asyncio.create_task(self._run())

    async def stop(self):
        self.is_running = False
        if self._task:
            self._task.cancel()
            self._task = None

    async def _subscribe(self, ws):
        requests = [
            {"method": "sub.depth.full", "param": {"symbol": self.symbol, "limit": 20}, "id": 1},
            {"method": "sub.deal", "param": {"symbol": self.symbol}, "id": 2},
        ]
        for index, interval in enumerate(MEXC_INTERVALS.items(), 3):
            timeframe, mexc_interval = interval
            requests.append({"method": "sub.kline", "param": {"symbol": self.symbol, "interval": mexc_interval}, "id": index})
        for request in requests:
            await ws.send(json.dumps(request))

    @staticmethod
    def _channel(message: Dict[str, Any]) -> str:
        return str(message.get("channel") or message.get("c") or "")

    async def _run(self):
        backoff = 1.0
        while self.is_running:
            try:
                async with websockets.connect(self.base_url, ping_interval=20, ping_timeout=10, max_size=2 ** 22) as ws:
                    await self._subscribe(ws)
                    backoff = 1.0
                    while self.is_running:
                        message = json.loads(await asyncio.wait_for(ws.recv(), 15))
                        if not isinstance(message, dict):
                            continue
                        if "ping" in message:
                            await ws.send(json.dumps({"pong": message["ping"]}))
                            continue
                        channel = self._channel(message)
                        if channel == "push.depth.full":
                            result = normalize_mexc_depth(message)
                            if result and self.on_depth:
                                bids, asks, event_time = result
                                received = time.time()
                                if event_time and received - event_time / 1000.0 > 3:
                                    continue
                                self.last_book_event_time = event_time / 1000.0 if event_time else 0.0
                                self.last_message_at["depth"] = received
                                self.total_ticks_received += 1
                                self.current_latency_ms = round(max(0.0, received * 1000 - event_time), 1) if event_time else 0.0
                                self.on_depth(bids, asks, event_time or int(received * 1000))
                                if self.on_book_ticker:
                                    self.on_book_ticker(bids[0][0], asks[0][0], (bids[0][0] + asks[0][0]) / 2)
                        elif channel == "push.deal":
                            result = normalize_mexc_deal(message)
                            if result and self.on_agg_trade:
                                self.last_message_at["agg_trade"] = time.time()
                                self.on_agg_trade(*result)
                        elif channel == "push.kline":
                            data = message.get("data", {})
                            interval = str(data.get("interval") or message.get("interval") or "")
                            timeframe = next((key for key, value in MEXC_INTERVALS.items() if value == interval), None)
                            if not timeframe:
                                continue
                            row = normalize_mexc_kline(message, timeframe)
                            previous = self._forming.get(timeframe)
                            if row and previous and row["time"] > previous["time"] and self.on_kline:
                                previous["is_closed"] = True
                                self.on_kline(previous)
                            if row:
                                row["is_closed"] = False
                                self._forming[timeframe] = row
                                if self.on_kline:
                                    self.on_kline(row)
                                self.last_message_at[f"kline_{timeframe}"] = time.time()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self.last_error["stream"] = f"{type(exc).__name__}: {exc}"
                await asyncio.sleep(backoff)
                backoff = min(backoff * 1.5, 10.0)
