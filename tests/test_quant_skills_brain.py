import unittest
import numpy as np
import pandas as pd
from strategy.quant_skills_brain import QuantSkillsBrain

class TestQuantSkillsBrain(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        n = 100
        trend = np.linspace(80000, 85000, n)
        noise = np.random.randn(n) * 30
        prices = trend + noise
        self.df_trend = pd.DataFrame({
            "open": prices,
            "high": prices + 25.0,
            "low": prices - 25.0,
            "close": prices + 5.0
        })

    def test_hurst_exponent_trending(self):
        res = QuantSkillsBrain.calculate_hurst_exponent(self.df_trend["close"])
        self.assertGreaterEqual(res.hurst_exponent, 0.05)
        self.assertLessEqual(res.hurst_exponent, 0.95)
        self.assertIn(res.regime_type, ["MEAN_REVERTING", "RANDOM_WALK", "TRENDING"])

    def test_garman_klass_volatility(self):
        res = QuantSkillsBrain.calculate_garman_klass_volatility(self.df_trend)
        self.assertGreater(res.garman_klass_bar, 0.0)
        self.assertGreater(res.garman_klass_annualized, 0.0)
        self.assertIn(res.volatility_regime, ["LOW_COMPRESSION", "NORMAL", "HIGH_VOLATILITY", "EXTREME_EXPANSION"])

    def test_order_book_imbalance(self):
        bids = [(80000, 3.0), (79990, 2.0)]
        asks = [(80010, 1.0), (80020, 1.0)]
        res = QuantSkillsBrain.calculate_order_book_imbalance(bids, asks)
        self.assertAlmostEqual(res["obi"], (5.0 - 2.0) / 7.0, places=2)
        self.assertIn("BUY", res["status"])

    def test_chandelier_exit(self):
        stop_long = QuantSkillsBrain.calculate_chandelier_exit(self.df_trend, direction=1, entry_price=84000.0)
        self.assertGreater(stop_long, 0.0)
        stop_short = QuantSkillsBrain.calculate_chandelier_exit(self.df_trend, direction=-1, entry_price=84000.0)
        self.assertGreater(stop_short, 0.0)

    def test_four_stage_exit_plan(self):
        plan = QuantSkillsBrain.generate_four_stage_exit_plan(80000.0, 79000.0, direction=1)
        self.assertEqual(len(plan), 4)
        self.assertEqual(plan[0].size_pct, 0.25)
        self.assertGreater(plan[1].target_price, plan[0].target_price)

if __name__ == "__main__":
    unittest.main()