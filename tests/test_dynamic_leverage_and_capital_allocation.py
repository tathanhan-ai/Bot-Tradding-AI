# -*- coding: utf-8 -*-
import unittest
from risk.dynamic_leverage import DynamicLeverageEngine, LeverageAdvice
from risk.capital_allocator import PortfolioCapitalAllocator, AllocationResult
from risk.ai_order_researcher import AIOrderResearcher, AIOrderResearchResult


class TestDynamicLeverageAndCapitalAllocation(unittest.TestCase):
    def setUp(self):
        self.lev_engine = DynamicLeverageEngine(min_leverage=1, max_leverage=10, min_clearance_ratio=3.0)
        self.allocator = PortfolioCapitalAllocator(short_term_ratio=0.40, long_term_ratio=0.45, reserve_ratio=0.15)
        self.researcher = AIOrderResearcher()

    def test_dynamic_leverage_volatility_scaling(self):
        # Low volatility (ATR 0.2% of price)
        adv_low = self.lev_engine.calculate_optimal_leverage(
            current_price=80000.0,
            atr=160.0, # 0.2%
            regime="TRENDING_BULL",
            confidence=85,
            horizon="SHORT_TERM"
        )
        self.assertGreaterEqual(adv_low.leverage, 7)
        self.assertEqual(adv_low.horizon, "SHORT_TERM")

        # High volatility (ATR 2.5% of price)
        adv_high = self.lev_engine.calculate_optimal_leverage(
            current_price=80000.0,
            atr=2000.0, # 2.5%
            regime="VOLATILE_PANIC",
            confidence=70,
            horizon="SHORT_TERM"
        )
        self.assertLessEqual(adv_high.leverage, 2)
        self.assertIn("Biến động", adv_high.rationale)

    def test_dynamic_leverage_horizon_partitioning(self):
        # Same volatility, compare short-term vs long-term
        adv_short = self.lev_engine.calculate_optimal_leverage(
            current_price=80000.0,
            atr=350.0,
            horizon="SHORT_TERM",
            timeframe="5m"
        )
        adv_long = self.lev_engine.calculate_optimal_leverage(
            current_price=80000.0,
            atr=350.0,
            horizon="LONG_TERM",
            timeframe="1h"
        )
        # Long term swing should be conservatively capped at <= 4x
        self.assertLessEqual(adv_long.leverage, 4)
        self.assertGreaterEqual(adv_short.leverage, adv_long.leverage)
        self.assertEqual(adv_long.horizon, "LONG_TERM")
        self.assertEqual(adv_short.horizon, "SHORT_TERM")

    def test_dynamic_leverage_vpin_toxic_flow_downscaling(self):
        # Elevated VPIN (0.75)
        adv_vpin_elev = self.lev_engine.calculate_optimal_leverage(
            current_price=80000.0,
            atr=200.0,
            vpin=0.75
        )
        self.assertLessEqual(adv_vpin_elev.leverage, 3)
        self.assertEqual(adv_vpin_elev.vpin_impact, "ELEVATED_VPIN_CAP")

        # Critical VPIN (0.92)
        adv_vpin_crit = self.lev_engine.calculate_optimal_leverage(
            current_price=80000.0,
            atr=200.0,
            vpin=0.92
        )
        self.assertLessEqual(adv_vpin_crit.leverage, 2)
        self.assertEqual(adv_vpin_crit.vpin_impact, "CRITICAL_TOXIC_MIN")

    def test_dynamic_leverage_drawdown_penalty(self):
        adv_dd = self.lev_engine.calculate_optimal_leverage(
            current_price=80000.0,
            atr=200.0,
            current_drawdown_pct=5.5
        )
        self.assertEqual(adv_dd.leverage, 1)
        self.assertEqual(adv_dd.drawdown_impact, "DEFENSIVE_LOCK")

    def test_dynamic_leverage_liquidation_clearance_guarantee(self):
        # If SL is 2.5%, clearance ratio 3.0 requires liquidation distance >= 7.5%
        adv = self.lev_engine.calculate_optimal_leverage(
            current_price=80000.0,
            atr=400.0,
            sl_distance_pct=0.035 # 3.5% SL requires >= 10.5% clearance
        )
        self.assertGreaterEqual(adv.est_liq_distance_pct, 10.5)
        self.assertGreaterEqual(adv.liq_clearance_ratio, 3.0)

    def test_capital_allocator_sleeve_budgets(self):
        budgets = self.allocator.calculate_sleeve_budgets(total_equity=10000.0)
        self.assertEqual(budgets["short_term_budget"], 4000.0)
        self.assertEqual(budgets["long_term_budget"], 4500.0)
        self.assertEqual(budgets["reserve_buffer"], 1500.0)
        self.assertEqual(budgets["deployable_capital"], 8500.0)

    def test_capital_allocator_allocation_and_throttling(self):
        # 1. Normal short term allocation within limits
        res1 = self.allocator.evaluate_allocation(
            horizon="SHORT_TERM",
            requested_margin=500.0,
            total_equity=5000.0
        )
        self.assertTrue(res1.allowed)
        self.assertFalse(res1.is_throttled)
        self.assertEqual(res1.allocated_margin, 500.0)

        # 2. Short term requested margin exceeds available sleeve budget ($2000 total budget, $1800 used)
        active_pos = [{"margin": 1800.0, "horizon": "SHORT_TERM"}]
        res2 = self.allocator.evaluate_allocation(
            horizon="SHORT_TERM",
            requested_margin=400.0,
            total_equity=5000.0,
            active_positions=active_pos
        )
        # Sleeve cap 2000 - 1800 = 200 available. Should throttle to 200
        self.assertTrue(res2.allowed)
        self.assertTrue(res2.is_throttled)
        self.assertEqual(res2.allocated_margin, 200.0)

        # 3. Long term sleeve is completely independent and still has full $2250 available
        res3 = self.allocator.evaluate_allocation(
            horizon="LONG_TERM",
            requested_margin=1000.0,
            total_equity=5000.0,
            active_positions=active_pos
        )
        self.assertTrue(res3.allowed)
        self.assertFalse(res3.is_throttled)
        self.assertEqual(res3.allocated_margin, 1000.0)

    def test_vpin_defensive_small_capital_throttling(self):
        from dataclasses import dataclass
        @dataclass
        class MockHFT:
            vpin: float = 0.82
            toxicity_regime: str = "TOXIC_INFORMED"
            is_toxic_flow: bool = True
            market_resilience_pct: float = 65.0
            lob_imbalance_20: float = 0.25

        mock_hft = MockHFT()
        indicators = {"atr": 500.0, "rsi": 55.0, "adx": 25.0}

        res = self.researcher.research(
            current_price=80000.0,
            best_bid=79999.8,
            best_ask=80000.0,
            spread=0.2,
            indicators=indicators,
            ai_verdict=None,
            ensemble_result=None,
            ai_cro=None,
            current_balance=5000.0,
            active_timeframe="5m",
            visual_hft_metrics=mock_hft
        )

        # Confirm VPIN defensive throttling triggered
        self.assertTrue(res.is_vpin_throttled)
        self.assertIn("Điều Tiết Vốn Nhỏ An Toàn", res.research_rationale)
        self.assertLessEqual(res.optimal_leverage, 3)
        self.assertEqual(res.recommended_type, "POST_ONLY")
        self.assertEqual(res.strategy_horizon, "SHORT_TERM")


if __name__ == "__main__":
    unittest.main()
