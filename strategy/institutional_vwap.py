"""
Institutional Anchored VWAP (Volume Weighted Average Price) & Standard Deviation Bands
The benchmark metric used by institutional hedge funds and market makers:
- VWAP: True volume-weighted equilibrium price
- Upper Band +1σ, +2σ: Overextended expensive territory (Take Profit / Short fade zone)
- Lower Band -1σ, -2σ: Undervalued discount territory (Take Profit / Long bounce zone)
"""
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import pandas as pd
import numpy as np


@dataclass
class VWAPBandResult:
    vwap: float
    upper_band_1: float   # +1 Standard Deviation
    upper_band_2: float   # +2 Standard Deviation (Extreme High)
    lower_band_1: float   # -1 Standard Deviation
    lower_band_2: float   # -2 Standard Deviation (Extreme Low)
    current_deviation: float  # How many sigmas current price is away from VWAP
    valuation_status: str     # 'DISCOUNT_CHEAP', 'EQUILIBRIUM_FAIR', 'PREMIUM_EXPENSIVE'
    rationale: str


class InstitutionalVWAPEngine:
    def __init__(self):
        pass

    def calculate(self, df: pd.DataFrame, current_price: float) -> Optional[VWAPBandResult]:
        """Calculates Session VWAP and Standard Deviation Bands from candle data"""
        if df is None or len(df) < 15:
            return None

        # Typical price = (High + Low + Close) / 3
        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        volumes = df["volume"].values

        typical_price = (highs + lows + closes) / 3.0
        tp_v = typical_price * volumes
        cum_tp_v = np.cumsum(tp_v)
        cum_v = np.cumsum(volumes)

        # Avoid zero division
        cum_v = np.where(cum_v == 0, 1e-9, cum_v)
        vwap_series = cum_tp_v / cum_v
        current_vwap = vwap_series[-1]

        # Standard Deviation calculation: sqrt( sum( v * (tp - vwap)^2 ) / sum(v) )
        squared_diff = (typical_price - vwap_series) ** 2
        variance = np.cumsum(volumes * squared_diff) / cum_v
        std_dev = np.sqrt(np.maximum(0, variance[-1]))

        upper_1 = current_vwap + (1.0 * std_dev)
        upper_2 = current_vwap + (2.0 * std_dev)
        lower_1 = current_vwap - (1.0 * std_dev)
        lower_2 = current_vwap - (2.0 * std_dev)

        # Calculate deviation in standard deviations
        sigma_dev = (current_price - current_vwap) / std_dev if std_dev > 0 else 0.0

        if sigma_dev <= -1.8:
            status = "DISCOUNT_CHEAP"
            rationale = f"📐 VÙNG CHIẾT KHẤU CAO (Giá lệch {sigma_dev:.1f}σ dưới VWAP ${current_vwap:,.1f}). Xác suất bật hồi Mean-Reversion cực lớn!"
        elif sigma_dev >= 1.8:
            status = "PREMIUM_EXPENSIVE"
            rationale = f"📐 VÙNG ĐỊNH GIÁ CAO (Giá lệch +{sigma_dev:.1f}σ trên VWAP ${current_vwap:,.1f}). Không FOMO Mua, ưu tiên canh chốt hoặc Short!"
        elif abs(sigma_dev) <= 0.8:
            status = "EQUILIBRIUM_FAIR"
            rationale = f"📐 GIÁ TRỊ CÂN BẰNG (Quanh VWAP ${current_vwap:,.1f} lệch {sigma_dev:.1f}σ). Thị trường đang tôn trọng giá trị dòng tiền."
        else:
            status = "TRENDING_EXPANSION"
            rationale = f"Giá đang mở rộng theo xu hướng ({sigma_dev:+.1f}σ so với VWAP ${current_vwap:,.1f})."

        return VWAPBandResult(
            vwap=round(float(current_vwap), 2),
            upper_band_1=round(float(upper_1), 2),
            upper_band_2=round(float(upper_2), 2),
            lower_band_1=round(float(lower_1), 2),
            lower_band_2=round(float(lower_2), 2),
            current_deviation=round(float(sigma_dev), 2),
            valuation_status=status,
            rationale=rationale
        )
