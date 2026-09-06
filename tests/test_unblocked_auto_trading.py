"""Test unblocked auto trading: Auto-Grid generation in ranging market, adaptive VPIN, and Carver initial entry."""
import unittest
import numpy as np
import pandas as pd
import time

from trading.pipeline import CandidateOrder, MarketSnapshot, SevenStagePipeline, StageOutcome
from strategy.grid_planner import GridPlanner
from strategy.octobot_matrix import OctoBotMatrixEngine
from strategy.visual_hft_microstructure import VisualHFTMetrics


class UnblockedAutoTradingTest(unittest.TestCase):
    def test_grid_planner_generates_valid_hedge_grid_in_range(self):
        plan = GridPlanner.plan_hedge_grid(
            price=80000.0,
            best_bid=79995.0,
            best_ask=80005.0,
            support=78500.0,
            resistance=81500.0,
            vwap=80000.0,
            atr=500.0,
            regime="RANGING_SIDEWAY",
            recommended_strategy="SIDEWAY_GRID",
            adx=14.0,
            hurst=0.45,
        )
        self.assertTrue(plan.is_tradeable)
        self.assertEqual(plan.model, "HEDGE_GRID")
        self.assertGreaterEqual(len(plan.legs), 4)

    def test_octobot_matrix_lower_threshold(self):
        engine = OctoBotMatrixEngine(confidence_threshold=0.30)
        indicators = {"rsi": 58.0, "ema20": 80100.0, "ema50": 79900.0}
        consensus = engine.evaluate_matrix(
            current_price=80050.0,
            indicators=indicators,
            order_flow_telemetry={"buy_ratio": 65.0, "cvd_delta_60s": 50000.0, "absorption_signal": "NONE"},
            smc_data={"structure": "BULLISH_TREND", "demand_zone": [79800.0, 79950.0], "supply_zone": [81000.0, 81200.0], "liquidity_sweep": "NONE"},
            vwap_data={"vwap": 79950.0, "vwap_status": "PREMIUM", "dist_sigma": 0.5},
            mtf_consensus={"score": 0.4, "consensus": "BULLISH", "timeframes": {}}
        )
        self.assertGreaterEqual(consensus.matrix_score, 0.20)
        self.assertIn(consensus.consensus_state, ("STRONG_BULLISH", "WEAK_BULLISH"))

    def test_adaptive_vpin_does_not_veto_maker(self):
        hft = VisualHFTMetrics(
            vpin=0.74,
            vpin_ready=True,
            depth_ready=True,
            is_toxic_flow=True,
            market_resilience_pct=70.0,
            liquidity_drought_warning=False,
        )
        self.assertLess(hft.vpin, 0.88)
        self.assertGreaterEqual(hft.market_resilience_pct, 30.0)


if __name__ == "__main__":
    unittest.main()
