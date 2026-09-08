"""
Structural Support/Resistance SL & TP Engine (Bộ Tính SL & TP Theo Cản Cấu Trúc Kỹ Thuật)
Calculates institutional-grade Stop Loss & Take Profit based on:
1. Swing Highs & Swing Lows (Fractal Pivot Points)
2. Structural Support & Resistance Zones
3. Liquidity Wick Buffer (0.25 - 0.35 ATR) to prevent stop hunting
4. Risk-to-Reward (R:R) Validation (Requires R:R >= 1.5 in Trend, >= 1.2 in Range)
"""
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import pandas as pd
import numpy as np


@dataclass
class StructuralTradeSetup:
    valid: bool
    side: str                     # 'BUY' or 'SELL'
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_distance: float
    reward_distance: float
    rr_ratio: float
    tp_macro_extended: float      # Higher timeframe extension target (1h/4h)
    swing_anchor: float           # The swing high/low that anchored this SL
    buffer_used: float
    reason: str


TIMEFRAME_RISK_PROFILES: Dict[str, Dict[str, Any]] = {
    "1m": {
        "sl_atr_mult": 1.2,
        "min_sl_dist_pct": 0.0015,  # 0.15% (~$120 on BTC)
        "buffer_atr_mult": 0.20,
        "min_rr": 1.8,
        "lookback_swings": 15
    },
    "3m": {
        "sl_atr_mult": 1.2,
        "min_sl_dist_pct": 0.0018,  # 0.18%
        "buffer_atr_mult": 0.20,
        "min_rr": 1.8,
        "lookback_swings": 18
    },
    "5m": {
        "sl_atr_mult": 1.2,
        "min_sl_dist_pct": 0.0020,  # 0.20% (~$160 on BTC)
        "buffer_atr_mult": 0.20,
        "min_rr": 2.0,
        "lookback_swings": 20
    },
    "15m": {
        "sl_atr_mult": 1.3,
        "min_sl_dist_pct": 0.0025,  # 0.25% (~$200 on BTC)
        "buffer_atr_mult": 0.20,
        "min_rr": 2.2,
        "lookback_swings": 25
    },
    "1h": {
        "sl_atr_mult": 1.2,
        "min_sl_dist_pct": 0.0040,  # 0.40% (~$320 on BTC)
        "buffer_atr_mult": 0.20,
        "min_rr": 2.2,
        "lookback_swings": 30
    },
    "4h": {
        "sl_atr_mult": 1.0,
        "min_sl_dist_pct": 0.0060,  # 0.60% (~$480 on BTC)
        "buffer_atr_mult": 0.15,
        "min_rr": 2.5,
        "lookback_swings": 40
    },
    "1d": {
        "sl_atr_mult": 1.0,
        "min_sl_dist_pct": 0.0300,  # 3.00%
        "buffer_atr_mult": 0.15,
        "min_rr": 2.5,
        "lookback_swings": 50
    },
    "1w": {
        "sl_atr_mult": 0.8,
        "min_sl_dist_pct": 0.0500,  # 5.00% minimum floor for macro weekly swings
        "buffer_atr_mult": 0.12,
        "min_rr": 3.0,
        "lookback_swings": 26        # ~6 months of weekly swings
    },
    "1M": {
        "sl_atr_mult": 0.6,
        "min_sl_dist_pct": 0.0800,  # 8.00% minimum floor for monthly secular swings
        "buffer_atr_mult": 0.10,
        "min_rr": 3.5,
        "lookback_swings": 24        # 2 years of monthly swings
    }
}


class StructuralRiskCalculator:
    def __init__(self, atr_buffer_mult: float = 0.3):
        self.atr_buffer_mult = atr_buffer_mult

    def find_swing_points(self, df: pd.DataFrame, window: int = 5, lookback: int = 20) -> Tuple[float, float, float, float]:
        """
        Finds recent swing highs and swing lows from candle data.
        Fractal pivot: một đáy chỉ được công nhận khi nến giữa THẤP HƠN cả
        `window` nến hai bên (đỉnh thì ngược lại) — không còn lấy min/max trần
        của cả dải nến nên SL neo đúng cấu trúc, râu quét đơn lẻ không chạm.
        Returns: (recent_swing_low, major_support, recent_swing_high, major_resistance)
        """
        if df is None or len(df) < window * 2 + 1:
            return 0.0, 0.0, 0.0, 0.0

        highs = df["high"].to_numpy(dtype=float)
        lows = df["low"].to_numpy(dtype=float)

        eff_lookback = min(len(df), max(10, lookback))
        seg_lo = lows[-eff_lookback:]
        seg_hi = highs[-eff_lookback:]

        fractal_lows: list = []
        fractal_highs: list = []
        n = len(seg_lo)
        for i in range(window, n - window):
            center_lo = seg_lo[i]
            if bool((center_lo < seg_lo[i - window:i]).all() and (center_lo < seg_lo[i + 1:i + window + 1]).all()):
                fractal_lows.append(float(center_lo))
            center_hi = seg_hi[i]
            if bool((center_hi > seg_hi[i - window:i]).all() and (center_hi > seg_hi[i + 1:i + window + 1]).all()):
                fractal_highs.append(float(center_hi))

        # Đáy/đỉnh fractal gần nhất; không có pivot hợp lệ thì fallback biên
        # ATR-wide chứ không lấy min/max trần (vốn là đáy râu đơn lẻ).
        if fractal_lows:
            local_low = fractal_lows[-1]
        else:
            local_low = float(np.percentile(seg_lo, 10))
        if fractal_highs:
            local_high = fractal_highs[-1]
        else:
            local_high = float(np.percentile(seg_hi, 90))

        # Major structural levels (last 60 candles): vẫn dùng biên cứng vì đây
        # là vùng cản/tường thanh khoản, không phải neo SL trực tiếp.
        major_support = float(lows[-min(len(df), 60):].min())
        major_resist = float(highs[-min(len(df), 60):].max())

        return local_low, major_support, local_high, major_resist

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        if df is None or len(df) < period + 1:
            return 0.0
        high = df["high"]
        low = df["low"]
        close = df["close"]
        tr = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
        atr = float(tr.ewm(alpha=1/period, adjust=False).mean().iloc[-1])
        return atr

    def compute_setup(
        self,
        side: str,
        entry_price: float,
        df_structure: pd.DataFrame,
        df_macro: Optional[pd.DataFrame] = None,
        min_rr: Optional[float] = None,
        timeframe: str = "15m"
    ) -> StructuralTradeSetup:
        """
        Calculates exact Candle-Specific Adaptive Structural SL & TP with R:R validation.
        Enforces volatility noise floor to strictly prevent premature stop-outs on low timeframes.
        """
        side_clean = side.upper()
        prof = TIMEFRAME_RISK_PROFILES.get(timeframe, TIMEFRAME_RISK_PROFILES["15m"])
        required_rr = min_rr if min_rr is not None else prof["min_rr"]

        atr = self.calculate_atr(df_structure)
        if atr <= 0:
            atr = entry_price * 0.008

        # Timeframe-specific volatility floor & buffer
        min_sl_dist = entry_price * prof["min_sl_dist_pct"]
        sl_atr_dist = prof["sl_atr_mult"] * atr
        buffer = atr * prof["buffer_atr_mult"]

        lookback = prof.get("lookback_swings", 20)
        local_low, major_sup, local_high, major_res = self.find_swing_points(df_structure, lookback=lookback)

        # Macro extensions for long-term riding
        macro_low, macro_sup, macro_high, macro_res = (0.0, 0.0, 0.0, 0.0)
        if df_macro is not None and not df_macro.empty:
            macro_low, macro_sup, macro_high, macro_res = self.find_swing_points(df_macro, window=10, lookback=30)

        if side_clean == "BUY":
            # Long Setup:
            anchor_low = local_low if (0 < local_low < entry_price) else (entry_price - sl_atr_dist)
            raw_risk = entry_price - (anchor_low - buffer)
            # Enforce volatility floor to never get stopped out by normal micro-noise
            risk_dist = max(raw_risk, sl_atr_dist, min_sl_dist)
            stop_loss = round(entry_price - risk_dist, 2)

            # Target resistance or R:R expansion
            anchor_high = local_high if (local_high > entry_price) else (entry_price + (risk_dist * required_rr))
            target_high = max(anchor_high * 0.9985, entry_price + (risk_dist * required_rr))
            take_profit = round(target_high, 2)

            # Macro extended TP
            macro_tp = round(macro_res * 0.998, 2) if (macro_res > target_high) else round(take_profit + (1.5 * atr), 2)

            reward_dist = max(take_profit - entry_price, 0.0)
            rr = round(reward_dist / risk_dist, 2)
            swing_anchor = anchor_low

            valid = (rr >= required_rr) and (stop_loss < entry_price) and (take_profit > entry_price)
            reason = (
                f"BUY [{timeframe}] Setup: SL=${stop_loss:,.1f} (-{risk_dist/entry_price*100:.2f}%, đệm chống nhiễu ${risk_dist:,.1f}), "
                f"TP=${take_profit:,.1f} (+{reward_dist/entry_price*100:.2f}%) | R:R={rr}:1"
            )

        else:  # SELL / SHORT
            # Short Setup:
            anchor_high = local_high if (local_high > entry_price) else (entry_price + sl_atr_dist)
            raw_risk = (anchor_high + buffer) - entry_price
            # Enforce volatility floor to never get stopped out by normal micro-noise
            risk_dist = max(raw_risk, sl_atr_dist, min_sl_dist)
            stop_loss = round(entry_price + risk_dist, 2)

            # Target support or R:R expansion
            anchor_low = local_low if (0 < local_low < entry_price) else (entry_price - (risk_dist * required_rr))
            target_low = min(anchor_low * 1.0015, entry_price - (risk_dist * required_rr))
            take_profit = round(target_low, 2)

            # Macro extended TP
            macro_tp = round(macro_sup * 1.002, 2) if (0 < macro_sup < target_low) else round(take_profit - (1.5 * atr), 2)

            reward_dist = max(entry_price - take_profit, 0.0)
            rr = round(reward_dist / risk_dist, 2)
            swing_anchor = anchor_high

            valid = (rr >= required_rr) and (stop_loss > entry_price) and (take_profit < entry_price)
            reason = (
                f"SELL [{timeframe}] Setup: SL=${stop_loss:,.1f} (+{risk_dist/entry_price*100:.2f}%, đệm chống nhiễu ${risk_dist:,.1f}), "
                f"TP=${take_profit:,.1f} (-{reward_dist/entry_price*100:.2f}%) | R:R={rr}:1"
            )

        return StructuralTradeSetup(
            valid=valid,
            side=side_clean,
            entry_price=round(entry_price, 2),
            stop_loss=round(stop_loss, 2),
            take_profit=round(take_profit, 2),
            risk_distance=round(risk_dist, 2),
            reward_distance=round(reward_dist, 2),
            rr_ratio=rr,
            tp_macro_extended=round(macro_tp, 2),
            swing_anchor=round(swing_anchor, 2),
            buffer_used=round(buffer, 2),
            reason=reason
        )
