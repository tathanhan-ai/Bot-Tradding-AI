# -*- coding: utf-8 -*-
import unittest
import time
from strategy.visual_hft_microstructure import VisualHFTMicrostructureEngine, VisualHFTMetrics


class TestVisualHFTMicrostructure(unittest.TestCase):
    def setUp(self):
        self.engine = VisualHFTMicrostructureEngine(
            bucket_size_btc=5.0,
            num_buckets=10,
            toxic_vpin_threshold=0.65,
            min_resilience_threshold=35.0
        )

    def test_volume_bucket_and_vpin(self):
        # Feed 100 BTC of finely interleaved buys and sells (balanced healthy market)
        now = time.time()
        for i in range(100):
            # 1.0 BTC per trade, alternating buy and sell
            is_buyer_maker = (i % 2 == 0)
            self.engine.update_trade(price=65000.0, qty=1.0, is_buyer_maker=is_buyer_maker, timestamp=now + i * 0.1)
        
        m = self.engine.get_metrics()
        self.assertTrue(len(self.engine.completed_buckets) >= 5)
        self.assertLess(m.vpin, 0.40)
        self.assertEqual(m.toxicity_regime, "CLEAN")
        print(f"\nVPIN (Balanced healthy flow): {m.vpin:.3f} | Regime: {m.toxicity_regime}")

    def test_toxic_flow_detection(self):
        engine_toxic = VisualHFTMicrostructureEngine(bucket_size_btc=5.0, num_buckets=10)
        now = time.time()
        # Feed 100% aggressive one-sided sells (toxic flow)
        for i in range(10):
            engine_toxic.update_trade(price=65000.0, qty=5.0, is_buyer_maker=True, timestamp=now + i)
        
        m = engine_toxic.get_metrics()
        self.assertTrue(m.is_toxic_flow)
        self.assertEqual(m.toxicity_regime, "TOXIC_INFORMED")
        self.assertEqual(m.execution_safety_status, "CRITICAL_TOXIC_AVOID")
        print(f"Toxic Flow VPIN: {m.vpin:.3f} | Safety: {m.execution_safety_status}")

    def test_20_level_weighted_lob_imbalance(self):
        # 20 bids, 20 asks
        bids = [[65000.0 - i * 5, 2.0 + (20 - i) * 0.5] for i in range(20)] # Heavy bid wall near top
        asks = [[65001.0 + i * 5, 0.5] for i in range(20)] # Light asks
        
        m = self.engine.update_order_book(bids=bids, asks=asks)
        self.assertGreater(m.lob_imbalance_20, 0.3)
        self.assertIn("BUY_PRESSURE", m.book_pressure)
        self.assertGreater(m.bid_depth_usdt, m.ask_depth_usdt)
        print(f"20-Level Weighted Imbalance: {m.lob_imbalance_20:+.3f} | Pressure: {m.book_pressure}")

    def test_market_resilience(self):
        bids = [[65000.0 - i * 5, 1.0] for i in range(20)]
        asks = [[65001.0 + i * 5, 1.0] for i in range(20)]
        m = self.engine.update_order_book(bids=bids, asks=asks)
        self.assertGreater(m.market_resilience_pct, 0.0)
        print(f"Market Resilience: {m.market_resilience_pct:.1f}%")


if __name__ == "__main__":
    unittest.main()
