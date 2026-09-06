# -*- coding: utf-8 -*-
"""
Portfolio Capital Allocation Framework - Dual-Horizon Capital Partitioning
Partitions account capital into independent sleeves to guarantee stable allocation:
1. Short-Term Tactical Sleeve (Scalp/Intraday 1m-15m): 40% of total balance
2. Long-Term Strategic Sleeve (Swing/Macro 30m-1d): 45% of total balance
3. Emergency Cash Reserve Buffer: 15% of total balance (unencumbered margin call protection)
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


@dataclass
class AllocationResult:
    allowed: bool
    horizon: str
    requested_margin: float
    allocated_margin: float
    sleeve_budget: float
    sleeve_used: float
    sleeve_available: float
    reserve_buffer: float
    rationale: str
    is_throttled: bool = False


class PortfolioCapitalAllocator:
    def __init__(
        self,
        short_term_ratio: float = 0.40,
        long_term_ratio: float = 0.45,
        reserve_ratio: float = 0.15
    ):
        total = short_term_ratio + long_term_ratio + reserve_ratio
        self.short_term_ratio = short_term_ratio / total
        self.long_term_ratio = long_term_ratio / total
        self.reserve_ratio = reserve_ratio / total

    @staticmethod
    def get_horizon(timeframe: str) -> str:
        norm_tf = str(timeframe or "15m").strip().lower()
        if norm_tf in ("1m", "3m", "5m", "15m"):
            return "SHORT_TERM"
        return "LONG_TERM"

    def calculate_sleeve_budgets(self, total_equity: float) -> Dict[str, float]:
        eq = max(100.0, float(total_equity))
        return {
            "short_term_budget": round(eq * self.short_term_ratio, 2),
            "long_term_budget": round(eq * self.long_term_ratio, 2),
            "reserve_buffer": round(eq * self.reserve_ratio, 2),
            "deployable_capital": round(eq * (self.short_term_ratio + self.long_term_ratio), 2),
            "total_equity": round(eq, 2)
        }

    def compute_sleeve_utilization(
        self,
        active_positions: List[Dict[str, Any]],
        pending_orders: List[Any]
    ) -> Dict[str, float]:
        short_used = 0.0
        long_used = 0.0

        for pos in (active_positions or []):
            m = float(pos.get("margin", 0.0) or pos.get("initial_margin", 0.0) or 0.0)
            if m <= 0 and pos.get("entry_price", 0.0) > 0 and pos.get("units", 0.0) > 0:
                lev = max(1, int(pos.get("leverage", 5)))
                m = (float(pos["units"]) * float(pos["entry_price"])) / lev
            tf = pos.get("timeframe", pos.get("metadata", {}).get("timeframe", "15m"))
            hz = pos.get("horizon", pos.get("metadata", {}).get("horizon", self.get_horizon(tf)))
            if hz == "SHORT_TERM":
                short_used += m
            else:
                long_used += m

        for order in (pending_orders or []):
            m = 0.0
            if hasattr(order, "margin"):
                m = float(order.margin)
            elif isinstance(order, dict):
                m = float(order.get("margin", 0.0))
            tf = getattr(order, "timeframe", getattr(order, "metadata", {}).get("timeframe", "15m") if hasattr(order, "metadata") else "15m")
            hz = getattr(order, "horizon", getattr(order, "metadata", {}).get("horizon", self.get_horizon(tf)) if hasattr(order, "metadata") else self.get_horizon(tf))
            if hz == "SHORT_TERM":
                short_used += m
            else:
                long_used += m

        return {
            "short_term_used": round(short_used, 2),
            "long_term_used": round(long_used, 2),
            "total_used": round(short_used + long_used, 2)
        }

    def evaluate_allocation(
        self,
        horizon: str,
        requested_margin: float,
        total_equity: float,
        active_positions: Optional[List[Dict[str, Any]]] = None,
        pending_orders: Optional[List[Any]] = None,
        min_trade_margin: float = 30.0
    ) -> AllocationResult:
        budgets = self.calculate_sleeve_budgets(total_equity)
        util = self.compute_sleeve_utilization(active_positions or [], pending_orders or [])

        eff_hz = "SHORT_TERM" if horizon == "SHORT_TERM" else "LONG_TERM"
        sleeve_cap = budgets["short_term_budget"] if eff_hz == "SHORT_TERM" else budgets["long_term_budget"]
        sleeve_used = util["short_term_used"] if eff_hz == "SHORT_TERM" else util["long_term_used"]
        sleeve_avail = max(0.0, sleeve_cap - sleeve_used)

        max_total_allowed = max(0.0, budgets["deployable_capital"] - util["total_used"])
        effective_limit = min(sleeve_avail, max_total_allowed)

        req_m = max(0.0, float(requested_margin))
        if effective_limit < min_trade_margin:
            return AllocationResult(
                allowed=False,
                horizon=eff_hz,
                requested_margin=req_m,
                allocated_margin=0.0,
                sleeve_budget=sleeve_cap,
                sleeve_used=sleeve_used,
                sleeve_available=sleeve_avail,
                reserve_buffer=budgets["reserve_buffer"],
                rationale=f"Ngan von {eff_hz} da het han muc (Da dung: ${sleeve_used:.1f}/${sleeve_cap:.1f}, Kha dung: ${sleeve_avail:.1f} < toi thieu ${min_trade_margin:.1f}). Dam bao 15% quy du tru an toan (${budgets['reserve_buffer']:.1f}).",
                is_throttled=True
            )

        if req_m <= effective_limit:
            allocated = req_m
            throttled = False
            rationale = f"Phan bo thanh cong ${allocated:.1f} tu ngan von {eff_hz} (Han muc: ${sleeve_cap:.1f}, Kha dung: ${sleeve_avail:.1f})."
        else:
            allocated = round(effective_limit, 0)
            throttled = True
            rationale = f"Yeu cau ${req_m:.1f} vuot han muc con lai (${effective_limit:.1f}). Tu dong dieu tiet xuong ${allocated:.1f} de giu vung ty trong danh muc {eff_hz}."

        return AllocationResult(
            allowed=True,
            horizon=eff_hz,
            requested_margin=req_m,
            allocated_margin=allocated,
            sleeve_budget=sleeve_cap,
            sleeve_used=sleeve_used,
            sleeve_available=sleeve_avail,
            reserve_buffer=budgets["reserve_buffer"],
            rationale=rationale,
            is_throttled=throttled
        )

    def export_state(
        self,
        total_equity: float,
        active_positions: Optional[List[Dict[str, Any]]] = None,
        pending_orders: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        budgets = self.calculate_sleeve_budgets(total_equity)
        util = self.compute_sleeve_utilization(active_positions or [], pending_orders or [])
        return {
            "total_equity": budgets["total_equity"],
            "reserve_buffer": budgets["reserve_buffer"],
            "reserve_ratio_pct": round(self.reserve_ratio * 100, 1),
            "short_term": {
                "horizon": "SHORT_TERM",
                "target_ratio_pct": round(self.short_term_ratio * 100, 1),
                "budget_usdt": budgets["short_term_budget"],
                "used_usdt": util["short_term_used"],
                "available_usdt": max(0.0, round(budgets["short_term_budget"] - util["short_term_used"], 2)),
                "utilization_pct": round((util["short_term_used"] / max(1.0, budgets["short_term_budget"])) * 100, 1)
            },
            "long_term": {
                "horizon": "LONG_TERM",
                "target_ratio_pct": round(self.long_term_ratio * 100, 1),
                "budget_usdt": budgets["long_term_budget"],
                "used_usdt": util["long_term_used"],
                "available_usdt": max(0.0, round(budgets["long_term_budget"] - util["long_term_used"], 2)),
                "utilization_pct": round((util["long_term_used"] / max(1.0, budgets["long_term_budget"])) * 100, 1)
            }
        }
