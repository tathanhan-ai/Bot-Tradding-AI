"""
Freqtrade-Grade Time-Decaying Minimal ROI Table
Modeled after https://github.com/freqtrade/freqtrade/blob/develop/freqtrade/strategy/interface.py

Key Mechanism:
Allows strategies to exit trades at lower profit targets as trade duration increases.
- Fresh trade (high momentum expected): hold for large target (+3.5% to +5.0%).
- Stagnant trade (momentum faded, ranging chop): decay profit target down to +0.4%
  to take any gain above fees, unlock capital, and eliminate chop risk!
"""
import time
from typing import Dict, Tuple, Optional


class FreqtradeROIEngine:
    """
    Time-Decaying Minimal ROI Exit Coordinator
    """
    # Timeframe-tailored ROI curves: { minutes_in_trade: min_profit_ratio }
    ROI_TABLES = {
        "1m": {
            0: 0.015,     # 0-3m: Target 1.5%
            3: 0.009,     # 3-8m: Target 0.9%
            8: 0.005,     # 8-15m: Target 0.5%
            15: 0.0035    # >15m: Target 0.35% (release margin after fees)
        },
        "5m": {
            0: 0.025,     # 0-10m: Target 2.5%
            10: 0.015,    # 10-25m: Target 1.5%
            25: 0.008,    # 25-45m: Target 0.8%
            45: 0.004     # >45m: Target 0.4%
        },
        "15m": {
            0: 0.035,     # 0-15m: Target 3.5%
            15: 0.018,    # 15-45m: Target 1.8%
            45: 0.010,    # 45-90m: Target 1.0%
            90: 0.004     # >90m: Target 0.4%
        },
        "1h": {
            0: 0.050,     # 0-60m: Target 5.0%
            60: 0.030,    # 1-3h: Target 3.0%
            180: 0.015,   # 3-6h: Target 1.5%
            360: 0.006    # >6h: Target 0.6%
        }
    }

    def __init__(self, default_timeframe: str = "15m"):
        self.default_timeframe = default_timeframe

    def get_target_roi(self, trade_duration_seconds: float, timeframe: str = "15m") -> float:
        """
        Determines the active minimum ROI required for the given trade age.
        """
        tf = timeframe if timeframe in self.ROI_TABLES else self.default_timeframe
        table = self.ROI_TABLES.get(tf, self.ROI_TABLES["15m"])
        mins = trade_duration_seconds / 60.0

        # Sort table thresholds descending: e.g. [90, 45, 15, 0]
        sorted_thresholds = sorted(table.keys(), reverse=True)
        for threshold_min in sorted_thresholds:
            if mins >= threshold_min:
                return table[threshold_min]

        return table.get(0, 0.035)

    def evaluate_roi_exit(
        self,
        pos: dict,
        current_price: float,
        current_time: Optional[float] = None
    ) -> Tuple[bool, str, float, float]:
        """
        Evaluates whether the position meets Freqtrade's Minimal ROI exit condition.
        Returns: (should_exit, reason_str, current_profit_pct, target_roi_pct)
        """
        now = current_time or time.time()
        open_ts = pos.get("open_timestamp")
        if not open_ts:
            # Fallback to now if not set
            open_ts = now

        duration_sec = max(0.0, now - open_ts)
        pos_tf = pos.get("timeframe", self.default_timeframe)
        target_roi = self.get_target_roi(duration_sec, pos_tf)

        direction = pos.get("direction", 1)
        entry_price = pos.get("entry_price", current_price)

        if entry_price <= 0:
            return False, "", 0.0, target_roi * 100.0

        # Price return percentage (excluding leverage)
        profit_ratio = (current_price - entry_price) / entry_price * direction
        profit_pct = round(profit_ratio * 100.0, 2)
        target_roi_pct = round(target_roi * 100.0, 2)

        # Check if profit ratio clears the time-decaying target
        if profit_ratio >= target_roi:
            mins = int(duration_sec // 60)
            secs = int(duration_sec % 60)
            reason = (
                f"CHỐT LỜI FREQTRADE MINIMAL ROI 💰 (+{profit_pct:+.2f}% đạt mốc yêu cầu +{target_roi_pct}% "
                f"sau {mins}p{secs}s nắm giữ)"
            )
            return True, reason, profit_pct, target_roi_pct

        return False, "", profit_pct, target_roi_pct
