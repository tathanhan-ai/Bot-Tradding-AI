import unittest
import pandas as pd
import numpy as np
from strategy.smart_money_concepts import SmartMoneyEngine
from strategy.institutional_vwap import InstitutionalVWAPEngine
from strategy.ensemble_strategy import MeanReversionSubEngine, LiquiditySweepSubEngine, EnsembleCoordinator
from strategy.hummingbot_inventory_skew import HummingbotInventorySkewEngine

class TestRealtimeSMCEnsembleSkew(unittest.TestCase):
    def setUp(self):
        # Generate synthetic 15m candle DataFrame
        np.random.seed(42)
        n = 50
        base_price = 60000.0
        returns = np.random.normal(0, 0.002, n)
        closes = base_price * np.cumprod(1 + returns)
        highs = closes * (1 + np.random.uniform(0.001, 0.005, n))
        lows = closes * (1 - np.random.uniform(0.001, 0.005, n))
        opens = (closes + np.roll(closes, 1)) / 2.0
        opens[0] = closes[0]
        volumes = np.random.uniform(10, 100, n)

        self.df = pd.DataFrame({
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        })
        self.current_price = float(closes[-1])

    def test_smc_realtime_distance_and_testing(self):
        engine = SmartMoneyEngine()
        res = engine.analyze(self.df, self.current_price)
        self.assertIsNotNone(res)
        self.assertIsInstance(res.dist_demand_pct, float)
        self.assertIsInstance(res.dist_supply_pct, float)
        self.assertIn(res.is_testing_ob, [True, False])
        self.assertIn(res.tested_ob_type, ['DEMAND', 'SUPPLY', 'NONE'])

        # Test price inside Demand zone
        if res.nearest_demand_zone:
            inside_demand = (res.nearest_demand_zone[0] + res.nearest_demand_zone[1]) / 2.0
            res_inside = engine.analyze(self.df, inside_demand)
            self.assertTrue(res_inside.is_testing_ob)
            self.assertEqual(res_inside.tested_ob_type, 'DEMAND')

    def test_vwap_realtime_distance(self):
        engine = InstitutionalVWAPEngine()
        vwap_res = engine.calculate(self.df, self.current_price)
        self.assertIsNotNone(vwap_res)
        self.assertIsInstance(vwap_res.dist_vwap_pct, float)
        self.assertIsInstance(vwap_res.dist_vwap_usd, float)
        self.assertEqual(vwap_res.dist_vwap_usd, round(self.current_price - vwap_res.vwap, 2))

    def test_mean_reversion_continuous_scoring(self):
        engine = MeanReversionSubEngine()
        # Normal mid price should give a responsive continuous score, not just a hardcoded 0.0
        vote_mid = engine.evaluate(self.df, current_price=self.current_price)
        self.assertIsInstance(vote_mid.score, float)
        self.assertIn('%B', vote_mid.rationale)

        # Extreme high price should give negative mean reversion score
        vote_high = engine.evaluate(self.df, current_price=self.current_price * 1.05)
        self.assertEqual(vote_high.direction, -1)
        self.assertLessEqual(vote_high.score, -50.0)

        # Extreme low price should give positive mean reversion score
        vote_low = engine.evaluate(self.df, current_price=self.current_price * 0.95)
        self.assertEqual(vote_low.direction, 1)
        self.assertGreaterEqual(vote_low.score, 50.0)

    def test_liquidity_sweep_dynamic_absorption(self):
        engine = LiquiditySweepSubEngine()
        vote = engine.evaluate(self.df, current_price=self.current_price)
        self.assertIsInstance(vote.score, float)
        self.assertIn('Râu', vote.rationale)

    def test_hummingbot_prospective_spread_when_flat(self):
        engine = HummingbotInventorySkewEngine()
        status_flat = engine.calculate_reservation_price(
            mid_price=65000.0, current_position=None, atr=400.0, total_balance=1000.0
        )
        self.assertEqual(status_flat.current_position_side, 'FLAT')
        self.assertEqual(status_flat.inventory_ratio_q, 0.0)
        self.assertGreater(status_flat.prospective_bid_spread, 0.0)
        self.assertLess(status_flat.prospective_bid_price, 65000.0)
        self.assertGreater(status_flat.prospective_ask_price, 65000.0)
        self.assertIn('Biên đón Avellaneda', status_flat.skew_direction)

    def test_hummingbot_net_inventory_with_hedge_positions(self):
        engine = HummingbotInventorySkewEngine()
        # Single flat position, but grid legs active: 2 Long legs (margin  each) and 1 Short leg (margin )
        # Net = +
        hedge_positions = {
            'grid_1': {'direction': 1, 'margin': 50.0},
            'grid_2': {'direction': 1, 'margin': 50.0},
            'grid_3': {'direction': -1, 'margin': 30.0},
        }
        status = engine.calculate_reservation_price(
            mid_price=65000.0, current_position=None, atr=400.0, total_balance=1000.0,
            hedge_positions=hedge_positions
        )
        self.assertEqual(status.current_position_side, 'LONG')
        self.assertEqual(status.inventory_ratio_q, 0.07)  # 70 / 1000
        self.assertGreater(status.price_skew_offset, 0.0)
        self.assertLess(status.reservation_price, 65000.0)
        self.assertIn('DISCOURAGE_BUY', status.skew_direction)

if __name__ == '__main__':
    unittest.main()
