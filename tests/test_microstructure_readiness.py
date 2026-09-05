from concurrent.futures import ThreadPoolExecutor
import math
import time
import unittest
from unittest.mock import patch

from strategy.order_flow_cvd import OrderFlowCVDEngine
from strategy.visual_hft_microstructure import VisualHFTMicrostructureEngine


def depth():
    return [[60000 - i, 2] for i in range(20)], [[60001 + i, 2] for i in range(20)]


class MicrostructureReadinessTest(unittest.TestCase):
    def test_warmup_and_exact_balanced_bucket_completion(self):
        engine = VisualHFTMicrostructureEngine(bucket_size_btc=2)
        initial = engine.get_metrics()
        self.assertFalse(initial.vpin_ready)
        self.assertFalse(initial.depth_ready)
        self.assertEqual(initial.toxicity_regime, "WARMUP")
        self.assertEqual(initial.execution_safety_status, "WARMUP")
        self.assertEqual(initial.market_resilience_pct, 0)
        for n in range(5):
            engine.update_trade(60000, 1, False)
            metrics = engine.update_trade(60000, 1, True)
            self.assertEqual(metrics.completed_bucket_count, n + 1)
            self.assertEqual(engine.current_bucket.total_volume, 0)
        self.assertTrue(metrics.vpin_ready)
        self.assertEqual(metrics.vpin, 0)
        self.assertEqual(metrics.execution_safety_status, "WARMUP")
        metrics = engine.update_order_book(*depth())
        self.assertTrue(metrics.depth_ready)
        self.assertEqual(metrics.execution_safety_status, "SAFE")
        self.assertFalse(initial.vpin_ready)  # Returned snapshots do not mutate underneath a consumer.

    def test_configured_fractional_buckets_and_toxic_flow(self):
        engine = VisualHFTMicrostructureEngine(bucket_size_btc=.1)
        metrics = engine.update_trade(60000, .5, True)
        self.assertEqual(engine.bucket_size, .1)
        self.assertEqual(metrics.completed_bucket_count, 5)
        self.assertEqual(metrics.vpin, 1)
        self.assertTrue(metrics.is_toxic_flow)
        self.assertEqual(metrics.execution_safety_status, "CRITICAL_TOXIC_AVOID")
        engine = VisualHFTMicrostructureEngine(bucket_size_btc=.1)
        engine.update_trade(60000, 10, True)
        self.assertEqual(engine.bucket_counter, 101)
        self.assertEqual(len(engine.completed_buckets), engine.num_buckets)

    def test_invalid_trade_and_depth_never_publish_safety(self):
        engine = VisualHFTMicrostructureEngine(bucket_size_btc=1)
        for invalid in (float("inf"), float("nan"), -1, 0, None, True):
            engine.update_trade(60000, invalid, False)
        self.assertEqual(engine.trade_count, 0)
        self.assertEqual(engine.get_metrics().completed_bucket_count, 0)
        for bids, asks in (([[60000, 1]], [[60001, 1]]), ([[60000, float("nan")]] * 20, [[60001, 1]] * 20),
                           ([[60000, 1]] * 20, [[60001, 1]] * 20)):
            self.assertFalse(engine.update_order_book(bids, asks).depth_ready)
            self.assertEqual(engine.get_metrics().execution_safety_status, "WARMUP")
        self.assertTrue(engine.update_order_book(*depth()).depth_ready)
        self.assertFalse(engine.update_order_book([], []).depth_ready)

    def test_cvd_uses_exchange_event_age_seconds_or_milliseconds(self):
        now = 1_800_000_000
        engine = OrderFlowCVDEngine()
        with patch("strategy.order_flow_cvd.time.time", return_value=now):
            self.assertFalse(engine.add_trade(60000, 1, False, (now - 61) * 1000))
            self.assertFalse(engine.add_trade(60000, 1, False, (now + 3) * 1000))
            self.assertTrue(engine.add_trade(60000, 1, False, now - 59))
            self.assertTrue(engine.add_trade(60000, 2, True, (now - 59) * 1000))
            self.assertEqual(engine.evaluate(60000).cvd_delta_60s, -1)
            self.assertEqual(engine.last_trade_timestamp_ms, (now - 59) * 1000)
        with patch("strategy.order_flow_cvd.time.time", return_value=now + 2):
            result = engine.evaluate(60000)
            self.assertEqual(result.cvd_delta_60s, 0)
            self.assertEqual(result.current_cvd, -1)

    def test_cvd_invalid_and_out_of_order_payloads(self):
        now = 1_800_000_000
        engine = OrderFlowCVDEngine()
        with patch("strategy.order_flow_cvd.time.time", return_value=now):
            for invalid in (float("nan"), float("inf"), -1, 0, None, True):
                self.assertFalse(engine.add_trade(60000, invalid, False, now))
            self.assertFalse(engine.add_trade(60000, 1, "false", now))
            self.assertEqual(engine.cumulative_delta, 0)
            engine.add_trade(100, 1, False, now - 50)
            engine.add_trade(110, 1, False, now - 10)
            engine.add_trade(100, 1, False, now - 30)
            verdict = engine.evaluate(110)
            self.assertEqual(verdict.delta_momentum, "STRONG_BUY_PRESSURE")
            self.assertEqual(verdict.absorption_signal, "NONE")
            self.assertEqual(verdict.cvd_delta_60s, 3)

    def test_busy_window_is_not_truncated_by_legacy_tick_cap(self):
        now = 1_800_000_000
        engine = OrderFlowCVDEngine(max_ticks=2)
        with patch("strategy.order_flow_cvd.time.time", return_value=now):
            for age in (30, 20, 10):
                engine.add_trade(60000, 1, False, now - age)
            self.assertEqual(engine.evaluate(60000).cvd_delta_60s, 3)
        with patch("strategy.order_flow_cvd.time.time", return_value=now + 61):
            engine.add_trade(60000, 1, False, now + 61)
            self.assertEqual(len(engine.ticks), 1)

    def test_concurrent_market_writes_and_snapshot_reads_are_consistent(self):
        engine = VisualHFTMicrostructureEngine(bucket_size_btc=2)
        cvd = OrderFlowCVDEngine()
        engine.update_order_book(*depth())

        def writer():
            for i in range(500):
                engine.update_trade(60000, 1, i % 2 == 0)
                cvd.add_trade(60000, 1, i % 2 == 0, time.time())

        def reader():
            for _ in range(500):
                metrics = engine.get_metrics()
                self.assertEqual(metrics.vpin_ready, metrics.completed_bucket_count >= 5)
                self.assertTrue(math.isfinite(cvd.evaluate(60000).current_cvd))

        with ThreadPoolExecutor(max_workers=2) as pool:
            for future in (pool.submit(writer), pool.submit(reader)):
                future.result()
        self.assertEqual(cvd.evaluate(60000).cvd_delta_60s, 0)


if __name__ == "__main__":
    unittest.main()
