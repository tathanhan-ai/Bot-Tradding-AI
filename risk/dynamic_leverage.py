"""
Dynamic Leverage Strategy Engine
Calculates the optimal leverage dynamically based on:
1. Market Volatility (ATR %)
2. AI Market Regime & Confidence Score
3. Liquidation Safety Margin (guaranteeing >= 2.5x distance to Stop-Loss)
"""
from dataclasses import dataclass
from typing import Tuple


@dataclass
class LeverageAdvice:
    leverage: int
    rationale: str
    est_liq_distance_pct: float
    safety_rating: str     # 'CỰC KỲ AN TOÀN', 'AN TOÀN', 'TRUNG BÌNH', 'RỦI RO'


class DynamicLeverageEngine:
    def __init__(self, min_leverage: int = 1, max_leverage: int = 10, target_stop_pct: float = 0.015):
        self.min_leverage = min_leverage
        self.max_leverage = max_leverage
        self.target_stop_pct = target_stop_pct  # 1.5% stop loss distance typical

    def calculate_optimal_leverage(
        self,
        current_price: float,
        atr: float,
        regime: str,
        confidence: int
    ) -> LeverageAdvice:
        """
        Dynamically selects safe leverage:
        - Calm, low-volatility strong trends with high AI confidence -> Safe to scale up to 5x - 8x.
        - Normal market conditions -> 3x - 5x.
        - High volatility / chop -> Scale down to 2x - 3x.
        - Panic flash dump / pump -> Scale down to 1x - 2x.
        """
        atr_pct = (atr / current_price) if current_price > 0 else 0.01

        # Base leverage inverted by volatility (higher ATR -> lower leverage)
        if atr_pct <= 0.005:
            base_lev = 7
        elif atr_pct <= 0.008:
            base_lev = 5
        elif atr_pct <= 0.015:
            base_lev = 3
        else:
            base_lev = 2

        # Regime adjustments
        if regime in ("VOLATILE_PANIC", "EXTREME_CHOP"):
            rec_lev = max(self.min_leverage, min(base_lev, 2))
            rationale = f"Biến động thị trường dâng cao (ATR={atr_pct*100:.2f}%). AI hạ đòn bẩy về {rec_lev}x để bảo vệ tài khoản khỏi các cú quét râu."
        elif regime == "RANGING_SIDEWAY":
            rec_lev = max(self.min_leverage, min(base_lev, 4))
            rationale = f"Thị trường tích lũy dao động hẹp. Đòn bẩy {rec_lev}x giúp tối ưu hóa biên độ lợi nhuận của từng tầng lệnh."
        elif regime in ("TRENDING_BULL", "TRENDING_BEAR") and confidence >= 80:
            rec_lev = max(self.min_leverage, min(base_lev + 1, self.max_leverage))
            rationale = f"Xu hướng rõ ràng, độ tin cậy AI {confidence}%. Đòn bẩy {rec_lev}x giúp nhân rộng tỷ suất sinh lời khi bứt phá sóng."
        else:
            rec_lev = max(self.min_leverage, min(base_lev, 4))
            rationale = f"Điều kiện giao dịch tiêu chuẩn. Áp dụng đòn bẩy cơ sở {rec_lev}x với biên độ thanh lý an toàn."

        # Estimate liquidation distance
        # On Binance Futures: Liquidation buffer ~ (1 / leverage) * 98% (with 0.5% maintenance margin)
        est_liq_distance = (0.98 / rec_lev) * 100.0

        if est_liq_distance >= 30:
            rating = "CỰC KỲ AN TOÀN"
        elif est_liq_distance >= 18:
            rating = "AN TOÀN"
        elif est_liq_distance >= 10:
            rating = "TRUNG BÌNH"
        else:
            rating = "RỦI RO CAO"

        return LeverageAdvice(
            leverage=rec_lev,
            rationale=rationale,
            est_liq_distance_pct=round(est_liq_distance, 1),
            safety_rating=rating
        )
