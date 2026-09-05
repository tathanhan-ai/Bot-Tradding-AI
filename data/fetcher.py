"""
Binance Futures Public Market Data Fetcher
Fetches OHLCV candlestick data directly from Binance Public REST API (No API Key required).
"""
import json
import time
import sys
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Optional

# Ensure utf-8 encoding on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import pandas as pd

from config.settings import DATA_DIR, DEFAULT_SYSTEM


def discard_forming_candle(frame: pd.DataFrame, interval_seconds: int, now: Optional[float] = None) -> pd.DataFrame:
    """Keep only bars whose Binance close time must already have elapsed."""
    if frame.empty:
        return frame
    now = time.time() if now is None else now
    cutoff = pd.Timestamp(now - interval_seconds, unit="s")
    return frame.loc[frame.index <= cutoff].copy()


class BinanceDataFetcher:
    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or DEFAULT_SYSTEM.binance_fapi_url

    def fetch_klines(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "15m",
        total_candles: int = 1500,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch historical candles with pagination and local caching.
        interval: '1m', '5m', '15m', '1h', '4h', '1d'
        """
        environment = "testnet" if "demo-" in self.base_url or "testnet" in self.base_url else "mainnet"
        cache_file = DATA_DIR / f"{environment}_{symbol.upper()}_{interval}_{total_candles}.csv"

        if use_cache and cache_file.exists():
            # Check if cache is reasonably fresh (< 1 day old)
            file_age = time.time() - cache_file.stat().st_mtime
            if file_age < 30:
                print(f"[Data] Đọc dữ liệu từ cache: {cache_file.name}")
                df = pd.read_csv(cache_file)
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df.set_index("timestamp", inplace=True)
                seconds = {"1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "2h": 7200, "4h": 14400, "1d": 86400, "1w": 604800, "1M": 2592000}.get(interval, 60)
                return discard_forming_candle(df, seconds)

        print(f"[Data] Đang tải {total_candles} nến {interval} của {symbol} từ Binance Futures API...")
        all_data = []
        limit_per_call = 1000
        fetched = 0
        end_time = None

        # Fetch backwards from current time
        while fetched < total_candles:
            call_limit = min(limit_per_call, total_candles - fetched)
            endpoint = f"{self.base_url}/fapi/v1/klines?symbol={symbol.upper()}&interval={interval}&limit={call_limit}"
            if end_time:
                endpoint += f"&endTime={end_time}"

            req = urllib.request.Request(
                endpoint,
                headers={"User-Agent": "BinanceFuturesBot/1.0"}
            )
            try:
                with urllib.request.urlopen(req, timeout=15) as response:
                    raw_json = response.read().decode("utf-8")
                    data = json.loads(raw_json)
            except Exception as e:
                print(f"[Data] Cảnh báo lỗi kết nối Binance API ({e}), đang thử lại...")
                time.sleep(2)
                try:
                    with urllib.request.urlopen(req, timeout=15) as response:
                        data = json.loads(response.read().decode("utf-8"))
                except Exception as ex:
                    print(f"[Data] Lỗi tải dữ liệu: {ex}")
                    break

            if not data or len(data) == 0:
                break

            # Binance returns oldest to newest
            all_data = data + all_data
            fetched += len(data)

            # Move end_time before the oldest candle in this batch
            end_time = data[0][0] - 1
            time.sleep(0.15)  # Respect Binance API rate limits

            if len(data) < call_limit:
                break

        if not all_data:
            raise ValueError(f"Không thể lấy dữ liệu cho cặp {symbol}")

        # Columns description from Binance Documentation
        columns = [
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore"
        ]

        df = pd.DataFrame(all_data, columns=columns)
        # Drop duplicates if any due to batch overlap
        df.drop_duplicates(subset=["open_time"], inplace=True)
        df.sort_values(by="open_time", inplace=True)
        df = df[df["close_time"].astype(float) <= time.time() * 1000.0]

        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
        for col in ["open", "high", "low", "close", "volume", "quote_volume"]:
            df[col] = df[col].astype(float)

        df.set_index("timestamp", inplace=True)
        df = df[["open", "high", "low", "close", "volume", "quote_volume"]]

        # Save to cache
        df.to_csv(cache_file)
        print(f"[Data] Đã tải thành công {len(df)} nến từ {df.index[0]} đến {df.index[-1]}. Đã lưu cache.")
        return df


if __name__ == "__main__":
    fetcher = BinanceDataFetcher()
    df = fetcher.fetch_klines("BTCUSDT", "15m", 500)
    print(df.tail(3))
