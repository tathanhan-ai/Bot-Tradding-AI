"""Closed MEXC Contract candles for paper-market parity."""
import json
import time
import urllib.parse
import urllib.request
from typing import Optional

import pandas as pd

from data.mexc_ws_stream import MEXC_INTERVALS
from trading.pipeline import candle_end, resample_closed_candles


_SECONDS = {"1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "4h": 14400, "1d": 86400, "1w": 604800, "1M": 2592000}


class MEXCDataFetcher:
    base_url = "https://contract.mexc.com"

    def __init__(self, base_url: Optional[str] = None, proxy_url: str = ""):
        self.base_url = (base_url or self.base_url).rstrip("/")
        self.proxy_url = proxy_url.strip()

    def _open(self, request):
        if not self.proxy_url:
            return urllib.request.urlopen(request, timeout=12)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({"http": self.proxy_url, "https": self.proxy_url}))
        return opener.open(request, timeout=12)

    @staticmethod
    def _symbol(symbol: str) -> str:
        return symbol if "_" in symbol else f"{symbol[:-4]}_USDT"

    @staticmethod
    def _frame(payload: dict, interval: str) -> pd.DataFrame:
        data = payload.get("data", payload)
        if not isinstance(data, dict):
            raise ValueError("MEXC kline payload is not an object")
        fields = {"time": "timestamp", "open": "open", "high": "high", "low": "low", "close": "close", "vol": "volume", "amount": "quote_volume"}
        columns = {}
        for remote, local in fields.items():
            values = data.get(remote)
            if not isinstance(values, list):
                raise ValueError(f"MEXC kline has no {remote}")
            columns[local] = values
        frame = pd.DataFrame(columns)
        if frame.empty:
            raise ValueError("MEXC kline is empty")
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="s", utc=True).dt.tz_localize(None)
        for name in ("open", "high", "low", "close", "volume", "quote_volume"):
            frame[name] = pd.to_numeric(frame[name], errors="raise")
        frame = frame.set_index("timestamp").sort_index()
        now = pd.Timestamp(time.time(), unit="s")
        frame = frame.loc[[candle_end(index, interval) <= now for index in frame.index]]
        return frame[["open", "high", "low", "close", "volume", "quote_volume"]]

    def fetch_klines(self, symbol: str, interval: str, total_candles: int = 250, use_cache: bool = False) -> pd.DataFrame:
        if interval == "3m":
            base = self.fetch_klines(symbol, "1m", min(2000, total_candles * 3 + 6), use_cache=False)
            return resample_closed_candles(base, "3m", time.time()).tail(total_candles)
        if interval not in MEXC_INTERVALS:
            raise ValueError(f"Unsupported MEXC interval: {interval}")
        end = int(time.time())
        start = end - _SECONDS[interval] * min(2000, total_candles + 6)
        query = urllib.parse.urlencode({"interval": MEXC_INTERVALS[interval], "start": start, "end": end})
        url = f"{self.base_url}/api/v1/contract/kline/{self._symbol(symbol)}?{query}"
        request = urllib.request.Request(url, headers={"User-Agent": "BinanceFuturesBot/1.0"})
        with self._open(request) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if isinstance(payload, dict) and payload.get("success") is False:
            raise ValueError(f"MEXC kline rejected: {payload}")
        return self._frame(payload, interval).tail(total_candles)
