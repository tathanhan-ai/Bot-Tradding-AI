"""
Multi-Order Futures Grid Trading Strategy
Partitions capital into multiple balanced price levels to capture profits from market fluctuations.
"""
from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
import pandas as pd

from config.settings import StrategyConfig, DEFAULT_STRATEGY
from strategy.base import BaseStrategy


@dataclass
class GridLevel:
    level_id: int
    buy_price: float
    sell_price: float
    units: float
    allocated_margin: float
    status: str              # 'PENDING_BUY', 'BOUGHT_WAITING_SELL', 'PROFIT_TAKEN'
    pnl: float = 0.0


class FuturesGridBot:
    def __init__(
        self,
        symbol: str = "BTCUSDT",
        total_balance: float = 1000.0,
        leverage: int = 3,
        num_grids: int = 10,
        grid_spread_pct: float = 0.03  # 3% grid width (from -1.5% to +1.5%)
    ):
        self.symbol = symbol
        self.total_balance = total_balance
        self.leverage = leverage
        self.num_grids = num_grids
        self.grid_spread_pct = grid_spread_pct
        self.grids: List[GridLevel] = []
        self.active_center_price: float = 0.0
        self.total_grid_profit: float = 0.0

    def generate_grid(self, current_price: float):
        """
        Builds a balanced grid of buy & sell levels around current market price.
        """
        self.active_center_price = current_price
        lower_bound = current_price * (1.0 - (self.grid_spread_pct / 2.0))
        upper_bound = current_price * (1.0 + (self.grid_spread_pct / 2.0))
        
        price_step = (upper_bound - lower_bound) / self.num_grids
        capital_per_grid = (self.total_balance * 0.7) / self.num_grids  # Use 70% capital for active grid
        
        self.grids.clear()
        for i in range(self.num_grids):
            buy_p = lower_bound + (i * price_step)
            sell_p = buy_p + price_step
            notional = capital_per_grid * self.leverage
            units = notional / buy_p

            self.grids.append(
                GridLevel(
                    level_id=i + 1,
                    buy_price=round(buy_p, 2),
                    sell_price=round(sell_p, 2),
                    units=round(units, 4),
                    allocated_margin=round(capital_per_grid, 2),
                    status="PENDING_BUY" if buy_p < current_price else "BOUGHT_WAITING_SELL"
                )
            )

    def on_tick(self, current_price: float) -> Optional[dict]:
        """
        Check if any grid order fills on the current price tick.
        """
        for g in self.grids:
            # Check Buy Fill
            if g.status == "PENDING_BUY" and current_price <= g.buy_price:
                g.status = "BOUGHT_WAITING_SELL"
                return {
                    "action": "GRID_BUY_FILL",
                    "level": g.level_id,
                    "price": g.buy_price,
                    "units": g.units,
                    "target_sell": g.sell_price
                }
            
            # Check Sell Fill (Take Profit)
            elif g.status == "BOUGHT_WAITING_SELL" and current_price >= g.sell_price:
                profit = (g.sell_price - g.buy_price) * g.units
                self.total_grid_profit += profit
                g.status = "PENDING_BUY"  # Re-arm level for next wave
                return {
                    "action": "GRID_PROFIT_TAKEN",
                    "level": g.level_id,
                    "sell_price": g.sell_price,
                    "profit": round(profit, 2),
                    "total_profit": round(self.total_grid_profit, 2)
                }
        return None
