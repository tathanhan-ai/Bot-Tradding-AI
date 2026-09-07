import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from data.persistent_storage import PersistentStorageManager
from trading.pipeline import CandidateOrder
from ui.server import LiveTradingState


def make_state():
    tmp = tempfile.TemporaryDirectory(prefix="ordertype-fix-")
    root = Path(tmp.name)
    s = LiveTradingState(storage=PersistentStorageManager(root / "db.sqlite", root / "backup.json"))
    s.vibe_swarm.enabled = False
    s.monthly_governor.enabled = False
    s.decision_mode = "DETERMINISTIC_ONLY"
    now = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("min")
    for tf, rule in (("1m", "min"), ("5m", "5min"), ("15m", "15min"), ("1h", "h")):
        end = now.floor(rule) - pd.Timedelta(rule if rule[0].isdigit() else "1" + rule)
        idx = pd.date_range(end=end, periods=250, freq=rule)
        n = np.arange(250)
        price = 80000 + n * 1.0 + 130 * np.sin(n / 8)
        s.data_map[tf] = pd.DataFrame(dict(open=price - 5, high=price + 30, low=price - 30,
                                            close=price, volume=np.full(250, 10.0), quote_volume=price * 10), index=idx)
    s.live_price = float(s.data_map["1m"].close.iloc[-1])
    p = s.live_price
    s.latest_l2_bids = [[p - .1 - n, 4.0] for n in range(20)]
    s.latest_l2_asks = [[p + .1 + n, 4.0] for n in range(20)]
    s.visual_hft.update_order_book(s.latest_l2_bids, s.latest_l2_asks)
    for _ in range(6):
        s.visual_hft.update_trade(p, 5.0, False)
        s.visual_hft.update_trade(p, 5.0, True)
    s.market_source_times = dict(depth=time.time(), agg_trade=time.time(), kline_1m=time.time())
    s.fee_engine.update_book(p - .1, p + .1)
    s._tmp = tmp
    return s


class OrderTypeFixTests(unittest.TestCase):
    def test_scale_ratio_maker_guard_rejects_crossing_spread(self):
        s = make_state()
        try:
            p = s.live_price
            # DCA BUY dat cham spread (gia >= best ask) -> phai bi tu choi nhu POST_ONLY
            order, reason = s.order_manager.place_order(
                symbol="BTCUSDT", order_type="SCALE_RATIO", side="BUY",
                price=p + 50.0, margin=200.0, leverage=4,
                best_bid=p - 0.1, best_ask=p + 0.1,
            )
            self.assertIsNone(order)
            self.assertIn("SCALE_RATIO", reason)
        finally:
            s._tmp.cleanup()

    def test_scale_ratio_maker_guard_passes_passive_ladder(self):
        s = make_state()
        try:
            p = s.live_price
            order, reason = s.order_manager.place_order(
                symbol="BTCUSDT", order_type="SCALE_RATIO", side="BUY",
                price=p - 100.0, margin=200.0, leverage=4,
                best_bid=p - 0.1, best_ask=p + 0.1,
            )
            self.assertIsNotNone(order)
            scale_orders = [o for o in s.order_manager.orders if o.order_type == "SCALE_RATIO"]
            self.assertEqual(len(scale_orders), 3)
        finally:
            s._tmp.cleanup()

    def test_veto_streak_downgrades_scale_to_post_only(self):
        s = make_state()
        try:
            now = time.time()
            s.veto_streak["SCALE_RATIO:1"] = {"count": 3, "last_veto_ts": now}
            s.market_source_times = dict(depth=time.time(), agg_trade=time.time(), kline_1m=time.time())
            s.analyze_snapshot(s.build_market_snapshot())
            s.order_research.recommended_side = "BUY"
            s.order_research.recommended_type = "SCALE_RATIO"
            s.order_research.optimal_price = s.live_price - 50
            s.order_research.structural_sl = s.live_price - 500
            s.order_research.structural_tp = s.live_price + 1000
            s.order_research.win_probability = 75
            s.order_research.optimal_margin = 0
            with patch.object(s, "evaluate_candidate_pipeline") as ev, \
                 patch.object(s, "queue_approved_candidate", return_value={"status": "pending"}) as q:
                from trading.pipeline import PipelineDecision, DecisionTrace, StageOutcome
                cand = s.build_candidate_order(1, "POST_ONLY", s.live_price, s.live_price - 500,
                                               s.live_price + 1000, "auto")
                trace = DecisionTrace(cand.order_id, "snap")
                trace.add("Stage 1 / Test", StageOutcome.pass_("ok"))
                ev.return_value = PipelineDecision(candidate=cand, trace=trace)
                # Chi kiem tra cand_type bi ep ve POST_ONLY truoc khi vao pipeline
                s.evaluate_ensemble_automated_decision(s.live_price)
                called_candidate = ev.call_args[0][0]
                self.assertEqual(called_candidate.order_type, "POST_ONLY")
        finally:
            s._tmp.cleanup()

    def test_veto_streak_cooldown_skips_cycle(self):
        s = make_state()
        try:
            now = time.time()
            s.veto_streak["SCALE_RATIO:1"] = {"count": 5, "last_veto_ts": now}
            s.market_source_times = dict(depth=time.time(), agg_trade=time.time(), kline_1m=time.time())
            s.analyze_snapshot(s.build_market_snapshot())
            s.order_research.recommended_side = "BUY"
            s.order_research.recommended_type = "SCALE_RATIO"
            with patch.object(s, "evaluate_candidate_pipeline") as ev:
                s.evaluate_ensemble_automated_decision(s.live_price)
                ev.assert_not_called()
        finally:
            s._tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
