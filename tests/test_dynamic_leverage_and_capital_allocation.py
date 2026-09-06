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

    def test_monthly_governor_capital_preservation_research_throttling(self):
        from risk.monthly_target_governor import MonthlyGovernorStatus
        gov_status = MonthlyGovernorStatus(
            enabled=True,
            base_target_pct=10.0,
            carried_deficit_pct=0.0,
            effective_target_pct=10.0,
            current_month_str="2026-09",
            month_start_balance=5000.0,
            month_realized_pnl=600.0,
            current_pnl_pct=12.0,
            progress_ratio=1.2,
            day_of_month=5,
            days_in_month=30,
            month_time_progress_pct=16.7,
            regime="TARGET_ACHIEVED",
            protection_mode="CAPITAL_PRESERVATION",
            size_multiplier=0.50,
            max_leverage_cap=3,
            min_ai_confidence=75,
            min_risk_reward_ratio=2.0,
            rationale="Đã đạt mục tiêu tháng",
            updated_at="12:00:00",
            target_pnl_usdt=500.0,
            target_daily_pnl_usdt=0.0,
            remaining_days=26,
            reserve_ratio_recommended=0.30
        )

        from dataclasses import dataclass
        @dataclass
        class MockFlow:
            delta_momentum: str = "STRONG_BUY_PRESSURE"
        mock_flow = MockFlow()

        indicators = {"atr": 400.0, "rsi": 50.0, "adx": 28.0}
        res = self.researcher.research(
            current_price=80000.0,
            best_bid=79999.0,
            best_ask=80001.0,
            spread=2.0,
            indicators=indicators,
            ai_verdict=None,
            ensemble_result=None,
            ai_cro=None,
            current_balance=5000.0,
            active_timeframe="15m",
            order_flow_verdict=mock_flow,
            monthly_governor_status=gov_status
        )

        # In Capital Preservation, leverage strictly <= 3x, size multiplier 0.5x, Maker order
        self.assertLessEqual(res.optimal_leverage, 3)
        self.assertEqual(res.recommended_type, "POST_ONLY")
        self.assertEqual(res.monthly_regime, "TARGET_ACHIEVED")
        self.assertEqual(res.monthly_size_multiplier, 0.50)
        self.assertGreaterEqual(res.rr_ratio, 2.0)
        # Reserve buffer in sleeve allocation should be 30% of $5000 = $1500
        self.assertEqual(res.sleeve_allocation["reserve_buffer"], 1500.0)

    def test_monthly_governor_deficit_catchup_rr_expansion(self):
        from risk.monthly_target_governor import MonthlyGovernorStatus
        from dataclasses import dataclass
        @dataclass
        class MockFlow:
            delta_momentum: str = "STRONG_BUY_PRESSURE"
        mock_flow = MockFlow()

        gov_status = MonthlyGovernorStatus(
            enabled=True,
            base_target_pct=10.0,
            carried_deficit_pct=3.5,
            effective_target_pct=13.5,
            current_month_str="2026-09",
            month_start_balance=5000.0,
            month_realized_pnl=0.0,
            current_pnl_pct=0.0,
            progress_ratio=0.0,
            day_of_month=2,
            days_in_month=30,
            month_time_progress_pct=6.7,
            regime="DEFICIT_CATCHUP",
            protection_mode="ADAPTIVE_CATCHUP",
            size_multiplier=0.85,
            max_leverage_cap=4,
            min_ai_confidence=65,
            min_risk_reward_ratio=2.5,
            rationale="Bù thiếu hụt tháng trước",
            updated_at="12:00:00",
            target_pnl_usdt=675.0,
            target_daily_pnl_usdt=23.28,
            remaining_days=29,
            reserve_ratio_recommended=0.15
        )

        indicators = {"atr": 400.0, "rsi": 50.0, "adx": 28.0}
        res = self.researcher.research(
            current_price=80000.0,
            best_bid=79999.0,
            best_ask=80001.0,
            spread=2.0,
            indicators=indicators,
            ai_verdict=None,
            ensemble_result=None,
            ai_cro=None,
            current_balance=5000.0,
            active_timeframe="15m",
            order_flow_verdict=mock_flow,
            monthly_governor_status=gov_status
        )

        self.assertLessEqual(res.optimal_leverage, 4)
        self.assertGreaterEqual(res.rr_ratio, 2.5)
        self.assertEqual(res.monthly_regime, "DEFICIT_CATCHUP")
        self.assertEqual(res.monthly_size_multiplier, 0.85)

    def test_carver_volatility_target_monthly_calibration(self):
        from strategy.carver_systematic_engine import VolatilityTargeter
        vt = VolatilityTargeter(annual_vol_target_pct=0.25)

        # Baseline: $5,000 with default 25% annual vol -> ~ $65.4 / day
        base_target = vt.get_daily_cash_vol_target(5000.0)
        self.assertAlmostEqual(base_target, (5000.0 * 0.25) / 19.10497, delta=1.0)

        # Monthly target 10%: implied annual return = 120%, implied annual vol capped at 40%
        scaled_target = vt.get_daily_cash_vol_target(5000.0, monthly_target_pct=10.0)
        self.assertGreater(scaled_target, base_target)

        # Target Achieved governor multiplier: 0.50x cuts daily cash vol in half
        achieved_target = vt.get_daily_cash_vol_target(5000.0, monthly_target_pct=10.0, governor_multiplier=0.50)
        self.assertAlmostEqual(achieved_target, scaled_target * 0.50, delta=0.5)

    def test_capital_allocator_dynamic_reserve_override(self):
        # Normal 15% reserve on $5,000 = $750
        budgets_normal = self.allocator.calculate_sleeve_budgets(5000.0)
        self.assertEqual(budgets_normal["reserve_buffer"], 750.0)
        self.assertEqual(budgets_normal["short_term_budget"], 2000.0)
        self.assertEqual(budgets_normal["long_term_budget"], 2250.0)

        # Boosted 30% reserve (Capital Preservation) on $5,000 = $1,500
        budgets_boosted = self.allocator.calculate_sleeve_budgets(5000.0, reserve_ratio_override=0.30)
        self.assertEqual(budgets_boosted["reserve_buffer"], 1500.0)
        self.assertEqual(budgets_boosted["deployable_capital"], 3500.0)
        # Short-term is 40/85 of 3500 = 1647.06, Long-term is 45/85 of 3500 = 1852.94
        self.assertAlmostEqual(budgets_boosted["short_term_budget"] + budgets_boosted["long_term_budget"], 3500.0, delta=0.5)


if __name__ == "__main__":
    unittest.main()
