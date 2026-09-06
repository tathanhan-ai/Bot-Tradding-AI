"""Deterministic, range-only grid planning shared by paper and Testnet execution."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class GridLeg:
    side: str
    entry_price: float
    stop_loss: float
    take_profit: float
    level: int


@dataclass(frozen=True)
class GridPlan:
    model: str
    reason: str
    lower: float = 0.0
    center: float = 0.0
    upper: float = 0.0
    legs: Tuple[GridLeg, ...] = ()

    @property
    def is_tradeable(self) -> bool:
        return self.model != "NO_TRADE" and bool(self.legs)

    def as_dict(self) -> Dict[str, object]:
        return {"model": self.model, "reason": self.reason, "lower": self.lower,
                "center": self.center, "upper": self.upper,
                "legs": [asdict(leg) for leg in self.legs]}


class GridPlanner:
    """Choose a bounded range plan; never manufacture a grid in an unclear regime."""

    MAX_LEVELS_PER_SIDE = 5
    MIN_LEVELS_PER_SIDE = 2

    @classmethod
    def plan_hedge_grid(
        cls, *, price: float, best_bid: float, best_ask: float, support: float,
        resistance: float, vwap: float, atr: float, regime: str,
        recommended_strategy: str, adx: float, hurst: float,
    ) -> GridPlan:
        values = (price, best_bid, best_ask, support, resistance, vwap, atr, adx, hurst)
        if any(not math.isfinite(float(value)) for value in values) or min(price, best_bid, best_ask, support, resistance, vwap, atr) <= 0:
            return GridPlan("NO_TRADE", "Grid inputs are incomplete or invalid")
        acceptable_regimes = ("RANGING_SIDEWAY", "SIDEWAY_GRID", "CHOPPY", "NEUTRAL", "EQUILIBRIUM_FAIR", "RANGING", "")
        acceptable_strategies = ("TWO_WAY_RANGE", "GRID_BOT", "SIDEWAY_GRID", "MEAN_REVERSION_GRID", "SMC_ORDER_BLOCK", "AUTO", "POST_ONLY", "GRID", "")
        if (regime and regime not in acceptable_regimes) or (recommended_strategy and recommended_strategy not in acceptable_strategies):
            return GridPlan("NO_TRADE", f"Regime {regime} / {recommended_strategy} is not a two-way range")
        if adx >= 28.0:
            return GridPlan("NO_TRADE", f"ADX {adx:.1f} indicates directional expansion")
        if hurst > 0.62:
            return GridPlan("NO_TRADE", f"Hurst {hurst:.2f} indicates persistent trend")
        if not (support < vwap < resistance and support < price < resistance):
            support = min(support if support > 0 else price - 2.5 * atr, price - 2.0 * atr, vwap - 1.5 * atr)
            resistance = max(resistance if resistance > 0 else price + 2.5 * atr, price + 2.0 * atr, vwap + 1.5 * atr)
        if abs(price - vwap) > atr * 0.75:
            return GridPlan("NO_TRADE", "Live price is too far from VWAP for a balanced hedge grid")

        width = resistance - support
        if width < atr * 2.0:
            return GridPlan("NO_TRADE", "Range is narrower than two ATR")
        # The number of levels is a consequence of measured range width, not a margin preset.
        side_levels = max(cls.MIN_LEVELS_PER_SIDE, min(cls.MAX_LEVELS_PER_SIDE, int(width / max(atr, 1e-9))))
        lower_span, upper_span = vwap - support, resistance - vwap
        if min(lower_span, upper_span) < atr * .6:
            return GridPlan("NO_TRADE", "One side of the range is too thin for a protective stop")
        legs: List[GridLeg] = []
        long_step, short_step = lower_span / (side_levels + 1), upper_span / (side_levels + 1)
        for level in range(1, side_levels + 1):
            # Begin at the outer range boundary: legs near VWAP cannot earn enough
            # to justify their protective stop and are deliberately omitted below.
            long_entry = support + long_step * level
            short_entry = resistance - short_step * level
            # Passive entries must never cross the current book.
            if long_entry >= best_ask or short_entry <= best_bid:
                return GridPlan("NO_TRADE", "Grid entry would cross the live book")
            long_stop, short_stop = support - .5 * atr, resistance + .5 * atr
            long_tp, short_tp = vwap, vwap
            if min((long_tp - long_entry) / max(long_entry - long_stop, 1e-9),
                   (short_entry - short_tp) / max(short_stop - short_entry, 1e-9)) < 1.0:
                continue
            legs.extend((
                GridLeg("BUY", round(long_entry, 8), round(long_stop, 8), round(long_tp, 8), level),
                GridLeg("SELL", round(short_entry, 8), round(short_stop, 8), round(short_tp, 8), level),
            ))
        if len(legs) < cls.MIN_LEVELS_PER_SIDE * 2:
            return GridPlan("NO_TRADE", "Range cannot support two protected pairs with reward/risk >= 1:1")
        return GridPlan("HEDGE_GRID", "Measured two-way range", support, vwap, resistance, tuple(legs))
