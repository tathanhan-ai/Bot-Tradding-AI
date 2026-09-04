"""
Jesse-Grade Quantitative Expectancy & Kelly Criterion Position Sizing Engine
Modeled after https://github.com/jesse-ai/jesse metrics and risk management

Key Formulas:
1. Trade Expectancy:
   E = (WinRate * AvgWin) - (LossRate * AvgLoss)
   Positive E proves statistical edge. Negative E signals market regime shift and forces capital preservation.
2. Half-Kelly Criterion Position Sizing:
   K = (W * (R + 1) - 1) / R
   Half_Kelly = max(0.01, min(0.15, K * 0.5))
   Dynamically sizes account margin allocation based on statistical edge.
3. Profit Factor & Performance Ratios:
   Profit Factor = Total Wins / Total Losses
"""
import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class JesseMetrics:
    total_trades: int = 0
    win_rate_pct: float = 0.0
    expectancy_usdt: float = 0.0     # Expected net USDT gain per trade
    profit_factor: float = 0.0
    avg_win_usdt: float = 0.0
    avg_loss_usdt: float = 0.0
    win_loss_ratio: float = 1.0      # R = AvgWin / AvgLoss
    kelly_fraction_pct: float = 5.0  # Dynamic optimal margin allocation %
    max_consecutive_losses: int = 0
    current_consecutive_losses: int = 0
    edge_status: str = "ESTABLISHING" # 'POSITIVE_EDGE 🟢', 'NEGATIVE_EDGE 🔴', 'ESTABLISHING ⚪'


class JesseExpectancyEngine:
    """
    Jesse Expectancy & Kelly Capital Allocator
    """
    def __init__(self, default_margin_pct: float = 10.0):
        self.default_margin_pct = default_margin_pct
        self.trade_pnls: List[float] = []
        self.current_consecutive_losses: int = 0
        self.max_consecutive_losses: int = 0

    def record_trade(self, net_pnl: float):
        self.trade_pnls.append(net_pnl)
        if net_pnl <= 0:
            self.current_consecutive_losses += 1
            if self.current_consecutive_losses > self.max_consecutive_losses:
                self.max_consecutive_losses = self.current_consecutive_losses
        else:
            self.current_consecutive_losses = 0

    def compute_metrics(self) -> JesseMetrics:
        if not self.trade_pnls:
            return JesseMetrics(
                total_trades=0,
                win_rate_pct=50.0,
                expectancy_usdt=0.0,
                profit_factor=1.0,
                kelly_fraction_pct=self.default_margin_pct,
                edge_status="ESTABLISHING ⚪"
            )

        wins = [p for p in self.trade_pnls if p > 0]
        losses = [abs(p) for p in self.trade_pnls if p <= 0]

        total = len(self.trade_pnls)
        num_wins = len(wins)
        num_losses = len(losses)
        win_rate = num_wins / total if total > 0 else 0.5
        loss_rate = 1.0 - win_rate

        avg_win = sum(wins) / num_wins if num_wins > 0 else 1.0
        avg_loss = sum(losses) / num_losses if num_losses > 0 else 1.0

        # Expectancy: E = (W * AvgWin) - (L * AvgLoss)
        expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)
        r_ratio = (avg_win / avg_loss) if avg_loss > 0 else 1.0

        total_loss_sum = sum(losses)
        profit_factor = (sum(wins) / total_loss_sum) if total_loss_sum > 0 else (99.0 if sum(wins) > 0 else 1.0)

        # Kelly Criterion: K = (W * (R + 1) - 1) / R
        if r_ratio > 0:
            k_raw = (win_rate * (r_ratio + 1.0) - 1.0) / r_ratio
        else:
            k_raw = 0.0

        # Half-Kelly for safety, clamped between 2.0% and 20.0% of balance
        if expectancy <= 0 or k_raw <= 0:
            kelly_pct = 3.0  # Defensive minimal allocation on negative expectancy
            edge_status = "NEGATIVE_EDGE 🔴"
        else:
            kelly_pct = max(2.0, min(20.0, round(k_raw * 0.5 * 100.0, 1)))
            edge_status = "POSITIVE_EDGE 🟢"

        return JesseMetrics(
            total_trades=total,
            win_rate_pct=round(win_rate * 100.0, 1),
            expectancy_usdt=round(expectancy, 2),
            profit_factor=round(profit_factor, 2),
            avg_win_usdt=round(avg_win, 2),
            avg_loss_usdt=round(avg_loss, 2),
            win_loss_ratio=round(r_ratio, 2),
            kelly_fraction_pct=kelly_pct,
            max_consecutive_losses=self.max_consecutive_losses,
            current_consecutive_losses=self.current_consecutive_losses,
            edge_status=edge_status
        )

    def calculate_optimal_margin(self, current_balance: float) -> float:
        """Returns optimal margin in USDT for next trade using Half-Kelly"""
        metrics = self.compute_metrics()
        margin = current_balance * (metrics.kelly_fraction_pct / 100.0)
        return max(10.0, round(margin, 2))
