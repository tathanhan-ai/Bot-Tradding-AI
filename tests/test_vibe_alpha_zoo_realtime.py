# -*- coding: utf-8 -*-
import unittest
import pandas as pd
import numpy as np

from strategy.vibe_alpha_zoo import VibeAlphaZooEngine, VibeAlphaMetrics


class VibeAlphaZooRealtimeTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        n = 50
        base = 80000.0
        self.df = pd.DataFrame({
            'open': [base + i * 5 for i in range(n)],
            'high': [base + i * 5 + 10 for i in range(n)],
            'low': [base + i * 5 - 10 for i in range(n)],
            'close': [base + i * 5 + 2 for i in range(n)],
            'volume': [100.0 for _ in range(n)]
        })
        self.engine = VibeAlphaZooEngine()

    def test_alpha_zoo_incorporates_live_current_price_tick(self):
        last_closed = float(self.df['close'].iloc[-1])
        
        # Test 1: Realtime surge (+0.5% jump)
        surge_price = last_closed * 1.005
        surge_metrics = self.engine.evaluate(self.df, surge_price, surge_price - 1, surge_price + 1, 2, lob_imbalance=0.7)
        self.assertIsInstance(surge_metrics, VibeAlphaMetrics)
        self.assertGreater(surge_metrics.momentum_zscore, 0)
        self.assertGreater(surge_metrics.ema_slope, 0)

        # Test 2: Realtime drop (-0.5% dump)
        dump_price = last_closed * 0.995
        dump_metrics = self.engine.evaluate(self.df, dump_price, dump_price - 1, dump_price + 1, 2, lob_imbalance=-0.7)
        self.assertLess(dump_metrics.momentum_zscore, surge_metrics.momentum_zscore)
        self.assertLess(dump_metrics.composite_alpha_score, surge_metrics.composite_alpha_score)

    def test_all_12_factors_are_populated_and_valid(self):
        p = float(self.df['close'].iloc[-1])
        m = self.engine.evaluate(self.df, p, p-1, p+1, 2, lob_imbalance=0.2, vpin=0.25)
        
        factors = [
            m.ema_slope, m.ema_acceleration, m.momentum_zscore,
            m.kyles_lambda, m.amihud_illiquidity, m.volume_force_ratio,
            m.yang_zhang_vol, m.ou_half_life_bars, m.squeeze_intensity,
            m.tail_risk_skew, m.absorption_ratio, m.composite_alpha_score
        ]
        self.assertEqual(len(factors), 12)
        self.assertTrue(all(np.isfinite(f) for f in factors))
        self.assertIn(m.alpha_regime, ['STRONG_BULLISH_MOMENTUM', 'BULLISH', 'NEUTRAL', 'BEARISH', 'STRONG_BEARISH_MOMENTUM'])
        self.assertTrue(len(m.dominant_factor) > 0)


if __name__ == '__main__':
    unittest.main()
