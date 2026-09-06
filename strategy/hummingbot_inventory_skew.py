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
    prospective_bid_spread: float = 0.0  # Dynamic Avellaneda half spread in USDT
    prospective_bid_price: float = 0.0   # Prospective maker bid
    prospective_ask_price: float = 0.0   # Prospective maker ask


class HummingbotInventorySkewEngine:
    def __init__(self, risk_aversion_gamma: float = 0.15, refresh_tolerance_pct: float = 0.08):
        self.gamma = risk_aversion_gamma
        self.refresh_tolerance_pct = refresh_tolerance_pct

    def calculate_reservation_price(
        self,
        mid_price: float,
        current_position: Optional[dict],
        atr: float,
        total_balance: float = 1000.0,
        hedge_positions: Optional[dict] = None
    ) -> InventorySkewStatus:
        atr = float(atr or 0.0)
        sigma = (atr / mid_price) if mid_price > 0 else 0.0
        # Prospective Avellaneda spread when flat: half spread = 0.5 * gamma * sigma * 10 * mid_price
        prospective_half = round(0.5 * self.gamma * (sigma * 10.0) * mid_price, 2)
        prospective_bid = round(mid_price - prospective_half, 2)
        prospective_ask = round(mid_price + prospective_half, 2)

        # Aggregate net inventory across current_position and hedge_positions (e.g. Grid legs)
        net_dir_margin = 0.0
        if current_position:
            net_dir_margin += current_position.get("direction", 0) * current_position.get("margin", 100.0)
        if hedge_positions and isinstance(hedge_positions, dict):
            for hpos in hedge_positions.values():
                net_dir_margin += hpos.get("direction", 0) * hpos.get("margin", 0.0)

        if mid_price <= 0 or net_dir_margin == 0.0:
            return InventorySkewStatus(
                current_position_side="FLAT",
                inventory_ratio_q=0.0,
                reservation_price=mid_price,
                price_skew_offset=0.0,
                skew_direction=f"NEUTRAL (Biên đón Avellaneda: ${prospective_bid:,.1f} - ${prospective_ask:,.1f} | ±${prospective_half:,.2f})",
                volatility_sigma=round(sigma, 4),
                prospective_bid_spread=prospective_half,
                prospective_bid_price=prospective_bid,
                prospective_ask_price=prospective_ask
            )

        # Normalized inventory ratio q in [-1.0, 1.0] relative to account balance
        q = net_dir_margin / max(total_balance, 1.0)
        q = max(-1.0, min(1.0, q))

        # Avellaneda-Stoikov reservation price shift: offset = q * gamma * sigma * mid_price * 10
        offset = round(q * self.gamma * (sigma * 10.0) * mid_price, 2)
        reservation_price = round(mid_price - offset, 2)

        if q > 0:
            side_str = "LONG"
            skew_dir = "DISCOURAGE_BUY (Hạ giá đón để tránh đọng vị thế Long)"
        else:
            side_str = "SHORT"
            skew_dir = "DISCOURAGE_SELL (Nâng giá đón để tránh đọng vị thế Short)"

        return InventorySkewStatus(
            current_position_side=side_str,
            inventory_ratio_q=round(q, 3),
            reservation_price=reservation_price,
            price_skew_offset=offset,
            skew_direction=skew_dir,
            volatility_sigma=round(sigma, 4),
            prospective_bid_spread=prospective_half,
            prospective_bid_price=prospective_bid,
            prospective_ask_price=prospective_ask
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
