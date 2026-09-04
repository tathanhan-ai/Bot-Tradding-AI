"""
Hummingbot-Grade Avellaneda-Stoikov Inventory Skew & Order Refresh Engine
Modeled after https://github.com/hummingbot/hummingbot pure_market_making & avellaneda_market_making

Key Mechanisms:
1. Avellaneda-Stoikov Reservation Price:
   r(s, q) = s - (q * gamma * sigma^2)
   - When Long (q > 0), shifts price DOWNWARD to discourage buying more and encourage exiting inventory.
   - When Short (q < 0), shifts price UPWARD to discourage shorting more and encourage covering.
   - When Flat (q = 0), reservation price equals fair mid-price.
2. Order Refresh Tolerance (Anti-Spam Filter):
   - Keeps pending limit orders in the book if price drift <= tolerance (e.g. 0.08%).
   - Preserves queue priority in Binance order book and eliminates unnecessary cancel/replace churn.
"""
import time
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class InventorySkewStatus:
    current_position_side: str    # "FLAT", "LONG", "SHORT"
    inventory_ratio_q: float      # [-1.0, 1.0]
    reservation_price: float      # Skewed target price
    price_skew_offset: float      # Offset from mid price in USDT
    skew_direction: str           # "DISCOURAGE_BUY", "DISCOURAGE_SELL", "NEUTRAL"
    volatility_sigma: float       # Current normalized volatility


class HummingbotInventorySkewEngine:
    def __init__(self, risk_aversion_gamma: float = 0.15, refresh_tolerance_pct: float = 0.08):
        self.gamma = risk_aversion_gamma
        self.refresh_tolerance_pct = refresh_tolerance_pct

    def calculate_reservation_price(
        self,
        mid_price: float,
        current_position: Optional[dict],
        atr: float,
        total_balance: float = 1000.0
    ) -> InventorySkewStatus:
        if not current_position or mid_price <= 0:
            return InventorySkewStatus(
                current_position_side="FLAT",
                inventory_ratio_q=0.0,
                reservation_price=mid_price,
                price_skew_offset=0.0,
                skew_direction="NEUTRAL",
                volatility_sigma=round(atr / max(mid_price, 1.0), 4)
            )

        direction = current_position.get("direction", 0)
        margin = current_position.get("margin", 100.0)
        # Normalized inventory ratio q in [-1.0, 1.0] relative to account balance
        q = (margin / max(total_balance, 1.0)) * direction
        q = max(-1.0, min(1.0, q))

        sigma = atr / mid_price  # Normalized volatility
        # Avellaneda-Stoikov reservation price shift: offset = q * gamma * sigma * mid_price * 10
        # Multiplied by 10 to scale meaningfully for futures spreads
        offset = round(q * self.gamma * (sigma * 10.0) * mid_price, 2)
        reservation_price = round(mid_price - offset, 2)

        if direction == 1:
            side_str = "LONG"
            skew_dir = "DISCOURAGE_BUY (Hạ giá đón để tránh đọng vị thế Long)"
        elif direction == -1:
            side_str = "SHORT"
            skew_dir = "DISCOURAGE_SELL (Nâng giá đón để tránh đọng vị thế Short)"
        else:
            side_str = "FLAT"
            skew_dir = "NEUTRAL"

        return InventorySkewStatus(
            current_position_side=side_str,
            inventory_ratio_q=round(q, 3),
            reservation_price=reservation_price,
            price_skew_offset=offset,
            skew_direction=skew_dir,
            volatility_sigma=round(sigma, 4)
        )

    def should_refresh_order(self, current_order_price: float, optimal_new_price: float) -> Tuple[bool, float]:
        """
        Determines whether a pending order should be cancelled and replaced.
        Returns (should_refresh, drift_pct).
        """
        if current_order_price <= 0:
            return True, 0.0

        drift_pct = abs(optimal_new_price - current_order_price) / current_order_price * 100.0
        # If price drift is smaller than tolerance, KEEP order in queue!
        if drift_pct <= self.refresh_tolerance_pct:
            return False, round(drift_pct, 3)
        return True, round(drift_pct, 3)
