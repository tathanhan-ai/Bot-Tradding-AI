import tempfile
import time
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

from data.persistent_storage import PersistentStorageManager
from trading.pipeline import CandidateOrder
from ui.server import LiveTradingState

class TestAllOrderTypesPipeline(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="order-types-test-")
        root = Path(self.tmp.name)
        self.state = LiveTradingState(storage=PersistentStorageManager(root / "db.sqlite", root / "backup.json"))
        self.state.is_running = True
        self.state.vibe_swarm.enabled = False
        self.state.monthly_governor.enabled = False
        self.state.decision_mode = "DETERMINISTIC_ONLY"
        self.state.order_manager.orders = []

        now = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("min")
        for tf, rule in (("1m", "min"), ("5m", "5min"), ("15m", "15min"), ("1h", "h")):
            end = now.floor(rule) - pd.Timedelta(rule if rule[0].isdigit() else "1" + rule)
            idx = pd.date_range(end=end, periods=250, freq=rule)
            n = np.arange(250)
            price = 80000 + n * 1.0 + 130 * np.sin(n / 8)
            self.state.data_map[tf] = pd.DataFrame(dict(open=price-5, high=price+30, low=price-30, close=price, volume=np.full(250, 10.0), quote_volume=price*10), index=idx)
        self.state.live_price = float(self.state.data_map["1m"].close.iloc[-1])
        p = self.state.live_price
        self.state.latest_l2_bids = [[p - .1 - n, 4.0] for n in range(20)]
        self.state.latest_l2_asks = [[p + .1 + n, 4.0] for n in range(20)]
        self.state.visual_hft.update_order_book(self.state.latest_l2_bids, self.state.latest_l2_asks)
        for _ in range(6):
            self.state.visual_hft.update_trade(p, 5.0, False)
            self.state.visual_hft.update_trade(p, 5.0, True)
        self.state.market_source_times = dict(depth=time.time(), agg_trade=time.time(), kline_1m=time.time())
        self.state.fee_engine.update_book(p-.1, p+.1)

    def tearDown(self):
        self.tmp.cleanup()

    def test_post_only_order_passes_and_queues(self):
        p = self.state.live_price
        cand = self.state.build_candidate_order(
            direction=-1,
            order_type="POST_ONLY",
            entry_price=p + 5.0,
            stop_loss=p + 500.0,
            take_profit=p - 1000.0,
            source="auto"
        )
        cand.margin = 100.0
        cand.quantity = 0.005
        cand.leverage = 4
        res = self.state.submit_candidate(cand)
        self.assertIn(res["status"], ("pending", "open", "active"))
        self.assertTrue(any(o.order_type == "POST_ONLY" for o in self.state.order_manager.pending_orders))

    def test_limit_order_passes_and_queues(self):
        p = self.state.live_price
        cand = self.state.build_candidate_order(
            direction=-1,
            order_type="LIMIT",
            entry_price=p + 10.0,
            stop_loss=p + 500.0,
            take_profit=p - 1000.0,
            source="auto"
        )
        cand.margin = 100.0
        cand.quantity = 0.005
        cand.leverage = 4
        res = self.state.submit_candidate(cand)
        self.assertIn(res["status"], ("pending", "open", "active"))
        self.assertTrue(any(o.order_type == "LIMIT" for o in self.state.order_manager.pending_orders))

    def test_scale_ratio_dca_ladder_passes_and_creates_3_tiers(self):
        p = self.state.live_price
        cand = self.state.build_candidate_order(
            direction=-1,
            order_type="SCALE_RATIO",
            entry_price=p + 10.0,
            stop_loss=p + 600.0,
            take_profit=p - 1200.0,
            source="auto"
        )
        cand.margin = 200.0
        cand.quantity = 0.010
        cand.leverage = 4
        cand.metadata["group_type"] = "SCALE"
        res = self.state.submit_candidate(cand)
        self.assertIn(res["status"], ("pending", "open", "active"))
        scale_orders = [o for o in self.state.order_manager.orders if o.order_type == "SCALE_RATIO"]
        self.assertEqual(len(scale_orders), 3)
        self.assertEqual([o.scale_level for o in scale_orders], [1, 2, 3])

    def _prime_momentum(self):
        # CONDITIONAL/TWAP thuoc nhom Taker: can momentum (OctoBot thuan + Alpha duong) de khong bi ha ve POST_ONLY.
        # Luu y: pipeline goi analyze_snapshot() lai ben trong va ghi de consensus, nen prime xong
        # phai khoa analyze_snapshot (no-op) khi submit de giu dung gia tri momentum mong muon.
        s = self.state
        for _ in range(30):
            s.jesse_engine.record_trade(10)
        s.market_source_times = dict(depth=time.time(), agg_trade=time.time(), kline_1m=time.time())
        s.freqtrade_protections.max_drawdown_guard.peak_balance = s.current_balance
        s.freqtrade_protections.max_drawdown_guard.pause_until = 0.0
        s.analyze_snapshot(s.build_market_snapshot())
        s.octobot_consensus.is_tradable = True
        s.octobot_consensus.recommended_direction = -1
        s.octobot_setup = None
        s.vibe_alpha_zoo.latest_metrics.composite_alpha_score = -60.0
        s.order_research.carver_output.update(raw_forecast=-3.0, daily_price_vol_pct=0.03)
        s.market_source_times = dict(depth=time.time(), agg_trade=time.time(), kline_1m=time.time())

    def test_conditional_stop_limit_order_passes_and_queues(self):
        self._prime_momentum()
        p = self.state.live_price
        cand = self.state.build_candidate_order(
            direction=-1,
            order_type="CONDITIONAL",
            entry_price=p - 200.0,
            stop_loss=p + 400.0,
            take_profit=p - 1000.0,
            source="auto"
        )
        cand.margin = 100.0
        cand.quantity = 0.005
        cand.leverage = 4
        cand.metadata.update({
            "trigger_price": p - 190.0,
            "trigger_condition": "BELOW"
        })
        from unittest.mock import patch as _patch
        # Khoa analyze_snapshot de giu momentum da prime (pipeline goi lai ben trong se ghi de)
        with _patch.object(self.state, "analyze_snapshot"):
            res = self.state.submit_candidate(cand)
        self.assertIn(res["status"], ("pending", "open", "active"))
        cond_orders = [o for o in self.state.order_manager.pending_orders if o.order_type == "CONDITIONAL"]
        self.assertTrue(len(cond_orders) > 0)
        self.assertEqual(cond_orders[0].trigger_price, p - 190.0)

    def test_trailing_stop_order_passes(self):
        p = self.state.live_price
        cand = self.state.build_candidate_order(
            direction=-1,
            order_type="TRAILING_STOP",
            entry_price=p,
            stop_loss=p + 500.0,
            take_profit=p - 1000.0,
            source="auto"
        )
        cand.margin = 100.0
        cand.quantity = 0.005
        cand.leverage = 4
        cand.metadata["callback_pct"] = 0.8
        res = self.state.submit_candidate(cand)
        self.assertIn(res["status"], ("pending", "open", "active", "filled"))

    def test_twap_order_passes_and_slices(self):
        self._prime_momentum()
        p = self.state.live_price
        cand = self.state.build_candidate_order(
            direction=-1,
            order_type="TWAP",
            entry_price=p,
            stop_loss=p + 500.0,
            take_profit=p - 1000.0,
            source="auto"
        )
        cand.margin = 200.0
        cand.quantity = 0.010
        cand.leverage = 4
        cand.metadata["twap_slices"] = 5
        from unittest.mock import patch as _patch
        # Khoa analyze_snapshot de giu momentum da prime (pipeline goi lai ben trong se ghi de)
        with _patch.object(self.state, "analyze_snapshot"):
            res = self.state.submit_candidate(cand)
        self.assertIn(res["status"], ("pending", "open", "active", "filled"))
        twap_orders = [o for o in self.state.order_manager.orders if o.order_type == "TWAP_SLICE"]
        self.assertEqual(len(twap_orders), 5)

    def test_market_order_passes_and_fills(self):
        p = self.state.live_price
        cand = self.state.build_candidate_order(
            direction=-1,
            order_type="MARKET",
            entry_price=p,
            stop_loss=p + 500.0,
            take_profit=p - 1000.0,
            source="auto"
        )
        cand.margin = 100.0
        cand.quantity = 0.005
        cand.leverage = 4
        res = self.state.submit_candidate(cand)
        self.assertIn(res["status"], ("filled", "open", "active", "pending"))

    def test_taker_types_degrade_to_post_only_without_momentum(self):
        # Khong prime momentum: CONDITIONAL/TWAP thieu dong luc phai duoc ha ve POST_ONLY an toan
        p = self.state.live_price
        for otype in ("CONDITIONAL", "TWAP"):
            cand = self.state.build_candidate_order(
                direction=-1, order_type=otype, entry_price=p,
                stop_loss=p + 500.0, take_profit=p - 1000.0, source="auto"
            )
            cand.margin = 100.0
            cand.quantity = 0.005
            cand.leverage = 4
            if otype == "CONDITIONAL":
                cand.metadata.update({"trigger_price": p - 190.0, "trigger_condition": "BELOW"})
            else:
                cand.metadata["twap_slices"] = 5
            self.state.order_manager.orders = []
            res = self.state.submit_candidate(cand)
            self.assertIn(res["status"], ("pending", "open", "active", "filled", "rejected"))
            if res["status"] != "rejected":
                self.assertEqual(self.state.order_manager.orders[0].order_type, "POST_ONLY")

    def test_automated_decision_cycle_dispatches_recommended_order(self):
        s = self.state
        snap = s.build_market_snapshot()
        s.analyze_snapshot(snap)
        self.assertIsNotNone(s.order_research)
        self.assertTrue(s.order_research.ready)

        carver_side = s.order_research.carver_action
        if carver_side in ("BUY", "SELL"):
            s.order_research.recommended_side = carver_side
            if carver_side == "SELL":
                s.order_research.optimal_price = s.live_price + 10.0
                s.order_research.structural_sl = s.live_price + 500.0
                s.order_research.structural_tp = s.live_price - 1000.0
            else:
                s.order_research.optimal_price = s.live_price - 10.0
                s.order_research.structural_sl = s.live_price - 500.0
                s.order_research.structural_tp = s.live_price + 1000.0

        s.evaluate_ensemble_automated_decision(s.live_price)
        self.assertTrue(len(s.order_manager.orders) > 0, "Automated decision cycle must place orders")
        first_order = s.order_manager.orders[0]
        self.assertIn(first_order.order_type, ("POST_ONLY", "LIMIT", "SCALE_RATIO", "MARKET", "CONDITIONAL", "TRAILING_STOP", "TWAP", "TWAP_SLICE"))

if __name__ == "__main__":
    unittest.main()
