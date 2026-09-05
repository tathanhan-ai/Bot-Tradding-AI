"""
Real-Time Order Flow & Cumulative Volume Delta (CVD) Engine
Tracks aggressive market orders (Taker Buy vs Taker Sell) directly from Binance Futures aggTrade WebSocket:
1. Cumulative Volume Delta (CVD): Net aggressive volume imbalance
2. Buy / Sell Ratio: Immediate buy vs sell pressure in %
3. Delta Absorption & Divergence: Detects when limit orders absorb aggressive flow (institutional footprint)
"""
from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from threading import RLock
import math
import time


@dataclass
class TradeTick:
    price: float
    qty: float
    is_buyer_maker: bool   # False = Taker Buy 🟢, True = Taker Sell 🔴
    timestamp_ms: int


@dataclass
class OrderFlowVerdict:
    current_cvd: float               # Cumulative Volume Delta (in BTC/base asset)
    buy_volume_1m: float             # Taker buy volume in last 60s
    sell_volume_1m: float            # Taker sell volume in last 60s
    buy_ratio_pct: float             # % buy volume vs total volume
    delta_momentum: str              # 'STRONG_BUY_PRESSURE', 'STRONG_SELL_PRESSURE', 'ABSORPTION_BUY', 'ABSORPTION_SELL', 'NEUTRAL'
    absorption_divergence: bool      # True if institutional absorption detected
    flow_rationale: str              # Human explanation

    @property
    def absorption_signal(self) -> str:
        return {"ABSORPTION_BUY": "BULL_ABSORPTION", "ABSORPTION_SELL": "BEAR_ABSORPTION"}.get(self.delta_momentum, "NONE")

    @property
    def cvd_delta_60s(self) -> float:
        return self.buy_volume_1m - self.sell_volume_1m


class OrderFlowCVDEngine:
    def __init__(self, max_ticks: int = 2000, window_seconds: int = 60):
        if not isinstance(window_seconds, (int, float)) or not math.isfinite(window_seconds) or window_seconds <= 0:
            raise ValueError("CVD window must be finite and positive")
        self.max_ticks = max_ticks
        self.window_seconds = window_seconds
        # Bound by event age; a fixed tick cap silently truncates the 60s volume during busy markets.
        self.ticks = deque()
        self.cumulative_delta = 0.0
        self.last_price_high = 0.0
        self.last_price_low = float('inf')
        self.last_cvd_high = 0.0
        self.last_cvd_low = float('inf')
        self._lock = RLock()
        self.last_trade_timestamp_ms = 0

    def add_trade(self, price: float, qty: float, is_buyer_maker: bool, timestamp_ms: int = 0):
        """Processes an incoming raw trade tick from Binance aggTrade WebSocket"""
        arrival_ms = int(time.time() * 1000)
        if (not isinstance(is_buyer_maker, bool) or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in (price, qty))
                or not math.isfinite(price * qty)):
            return False
        if not isinstance(timestamp_ms, (int, float)) or not math.isfinite(timestamp_ms) or timestamp_ms < 0:
            return False
        event_ms = arrival_ms if timestamp_ms == 0 else (timestamp_ms * 1000 if timestamp_ms < 1e11 else timestamp_ms)
        if event_ms < arrival_ms - self.window_seconds * 1000 or event_ms > arrival_ms + 2000:
            return False
        tick = TradeTick(price=price, qty=qty, is_buyer_maker=is_buyer_maker, timestamp_ms=int(event_ms))
        with self._lock:
            delta = self.cumulative_delta + (-qty if is_buyer_maker else qty)
            if not math.isfinite(delta):
                return False
            cutoff = arrival_ms - self.window_seconds * 1000
            while self.ticks and self.ticks[0].timestamp_ms < cutoff:
                self.ticks.popleft()
            self.ticks.append(tick)
            self.cumulative_delta = delta
            self.last_trade_timestamp_ms = max(self.last_trade_timestamp_ms, tick.timestamp_ms)
        return True

    def evaluate(self, current_price: float) -> OrderFlowVerdict:
        """Evaluates live order flow delta over recent window"""
        now_ms = time.time() * 1000
        cutoff_ms = now_ms - (self.window_seconds * 1000)

        with self._lock:
            recent_ticks = sorted((t for t in self.ticks if cutoff_ms <= t.timestamp_ms <= now_ms), key=lambda tick: tick.timestamp_ms)
            cumulative_delta = self.cumulative_delta
        if not recent_ticks:
            return OrderFlowVerdict(
                current_cvd=round(cumulative_delta, 3),
                buy_volume_1m=0.0,
                sell_volume_1m=0.0,
                buy_ratio_pct=50.0,
                delta_momentum="NEUTRAL",
                absorption_divergence=False,
                flow_rationale="Đang tích lũy dữ liệu dòng lệnh WebSocket..."
            )

        buy_vol = sum(t.qty for t in recent_ticks if not t.is_buyer_maker)
        sell_vol = sum(t.qty for t in recent_ticks if t.is_buyer_maker)
        total_vol = buy_vol + sell_vol
        buy_ratio = (buy_vol / total_vol * 100.0) if total_vol > 0 else 50.0
        recent_delta = buy_vol - sell_vol

        # Price trajectory in window
        p_start = recent_ticks[0].price
        p_end = recent_ticks[-1].price
        price_change_pct = ((p_end - p_start) / p_start) * 100.0 if p_start > 0 else 0.0

        # Institutional Absorption Divergence Detection
        # Case A: Bullish Absorption - Aggressive sells dominate (Sell Vol > Buy Vol), yet price REFUSES to drop
        bull_absorption = (buy_ratio < 40.0) and (price_change_pct >= -0.05)
        # Case B: Bearish Absorption - Aggressive buys dominate (Buy Vol > Sell Vol), yet price REFUSES to rise
        bear_absorption = (buy_ratio > 60.0) and (price_change_pct <= 0.05)

        momentum = "NEUTRAL"
        divergence = False
        rationale = ""

        if bull_absorption:
            momentum = "ABSORPTION_BUY"
            divergence = True
            rationale = f"🛡️ CÁ MẬP HẤP THỤ MUA (BULL ABSORPTION): Phe bán xả chủ động ({100-buy_ratio:.1f}%) nhưng giá không giảm (-{abs(price_change_pct):.2f}%). Lệnh Limit gom hàng của tổ chức đang chặn đáy!"
        elif bear_absorption:
            momentum = "ABSORPTION_SELL"
            divergence = True
            rationale = f"🛡️ CÁ MẬP HẤP THỤ BÁN (BEAR ABSORPTION): Phe mua đua lệnh ({buy_ratio:.1f}%) nhưng giá không tăng (+{price_change_pct:.2f}%). Tường lệnh Bán ẩn (Iceberg) của tổ chức đang xả ngầm!"
        elif buy_ratio >= 65.0:
            momentum = "STRONG_BUY_PRESSURE"
            rationale = f"🟢 ÁP LỰC MUA CHỦ ĐỘNG ÁP ĐẢO ({buy_ratio:.1f}% Taker Buy). Dòng tiền thị trường đang đẩy giá lên!"
        elif buy_ratio <= 35.0:
            momentum = "STRONG_SELL_PRESSURE"
            rationale = f"🔴 ÁP LỰC BÁN CHỦ ĐỘNG ÁP ĐẢO ({100-buy_ratio:.1f}% Taker Sell). Lực xả lệnh thị trường rất mạnh!"
        else:
            momentum = "BALANCED"
            rationale = f"Cân bằng dòng lệnh ({buy_ratio:.1f}% Mua / {100-buy_ratio:.1f}% Bán). Thị trường đang sideway chờ nhịp bứt phá."

        return OrderFlowVerdict(
            current_cvd=round(cumulative_delta, 3),
            buy_volume_1m=round(buy_vol, 3),
            sell_volume_1m=round(sell_vol, 3),
            buy_ratio_pct=round(buy_ratio, 1),
            delta_momentum=momentum,
            absorption_divergence=divergence,
            flow_rationale=rationale
        )
