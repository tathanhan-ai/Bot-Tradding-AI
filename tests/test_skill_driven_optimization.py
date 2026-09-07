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


def make_test_state():
    tmp = tempfile.TemporaryDirectory(prefix="skill-opt-")
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
        price = 60000 + n * 1.0 + 130 * np.sin(n / 8)
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


def prime(s, alpha=30, direction=1, jesse_wins=0):
    for i in range(jesse_wins):
        s.jesse_engine.record_trade(10 if i % 4 else -5)
    s.market_source_times = dict(depth=time.time(), agg_trade=time.time(), kline_1m=time.time())
    s.freqtrade_protections.max_drawdown_guard.peak_balance = s.current_balance
    s.freqtrade_protections.max_drawdown_guard.pause_until = 0.0
    s.analyze_snapshot(s.build_market_snapshot())
    s.octobot_consensus.is_tradable = True
    s.octobot_consensus.recommended_direction = direction
    s.octobot_setup = None
    s.vibe_alpha_zoo.latest_metrics.composite_alpha_score = alpha
    s.order_research.carver_output.update(raw_forecast=3.0, daily_price_vol_pct=0.03)
    s.market_source_times = dict(depth=time.time(), agg_trade=time.time(), kline_1m=time.time())


class SkillDrivenOptimizationTests(unittest.TestCase):
    def test_3_loss_cooldown_requires_30min(self):
        s = make_test_state()
        try:
            s.jesse_engine.record_trade(-5)
            s.jesse_engine.record_trade(-5)
            s.jesse_engine.record_trade(-5)
            self.assertGreaterEqual(s.jesse_engine.compute_metrics().current_consecutive_losses, 3)
            s.trades.append({"closed_at_ts": time.time() - 60.0, "pnl": -5})
            order = CandidateOrder(order_id="t1", symbol="BTCUSDT", direction=1, order_type="POST_ONLY",
                                   entry_price=s.live_price, stop_loss=s.live_price - 500,
                                   take_profit=s.live_price + 1000, quantity=0.01, source="auto")
            s.market_source_times = dict(depth=time.time(), agg_trade=time.time(), kline_1m=time.time())
            with patch.object(s, "analyze_snapshot"):
                decision = s.evaluate_candidate_pipeline(order)
            self.assertFalse(decision.approved)
            veto_reasons = " ".join(e.reason for e in decision.trace.entries if e.verdict == "VETO")
            self.assertIn("cooldown", veto_reasons.lower())
        finally:
            s._tmp.cleanup()

    def test_weak_combo_15m_long_veto_on_counter_evidence(self):
        s = make_test_state()
        try:
            prime(s, alpha=30, direction=1, jesse_wins=30)
            # Bang chung nguoc chieu ro rang: OctoBot SHORT + Alpha am manh
            s.vibe_alpha_zoo.latest_metrics.composite_alpha_score = -50.0
            s.octobot_consensus.is_tradable = True
            s.octobot_consensus.recommended_direction = -1
            order = CandidateOrder(order_id="t3", symbol="BTCUSDT", direction=1, order_type="POST_ONLY",
                                   entry_price=s.live_price, stop_loss=s.live_price - 500,
                                   take_profit=s.live_price + 1000, quantity=0.01, source="auto")
            order.metadata["timeframe"] = "15m"
            s.market_source_times = dict(depth=time.time(), agg_trade=time.time(), kline_1m=time.time())
            with patch.object(s, "analyze_snapshot"):
                decision = s.evaluate_candidate_pipeline(order)
            self.assertFalse(decision.approved)
            self.assertTrue(any("Weak attributed combo" in e.reason
                                for e in decision.trace.entries if e.verdict == "VETO"))
        finally:
            s._tmp.cleanup()

    def test_weak_combo_passes_without_counter_evidence(self):
        s = make_test_state()
        try:
            prime(s, alpha=0, direction=1, jesse_wins=30)
            # Combo yeu (15m-LONG) nhung KHONG co bang chung nguoc: OctoBot thuan + Alpha trung tinh
            s.vibe_alpha_zoo.latest_metrics.composite_alpha_score = 0.0
            s.octobot_consensus.is_tradable = True
            s.octobot_consensus.recommended_direction = 1
            order = CandidateOrder(order_id="t5", symbol="BTCUSDT", direction=1, order_type="LIMIT",
                                   entry_price=s.live_price, stop_loss=s.live_price - 500,
                                   take_profit=s.live_price + 1000, quantity=0.01, source="auto")
            order.metadata["timeframe"] = "15m"
            s.market_source_times = dict(depth=time.time(), agg_trade=time.time(), kline_1m=time.time())
            with patch.object(s, "analyze_snapshot"):
                decision = s.evaluate_candidate_pipeline(order)
            self.assertFalse(any("Weak attributed combo" in e.reason
                                 for e in decision.trace.entries if e.verdict == "VETO"))
        finally:
            s._tmp.cleanup()

    def test_strong_combo_short_not_filtered(self):
        s = make_test_state()
        try:
            prime(s, alpha=-60, direction=-1, jesse_wins=30)
            # 15m-SHORT la combo manh (PF=17.42) -> khong bi veto boi filter
            s.vibe_alpha_zoo.latest_metrics.composite_alpha_score = -60.0
            s.octobot_consensus.is_tradable = True
            s.octobot_consensus.recommended_direction = -1
            order = CandidateOrder(order_id="t4", symbol="BTCUSDT", direction=-1, order_type="POST_ONLY",
                                   entry_price=s.live_price, stop_loss=s.live_price + 500,
                                   take_profit=s.live_price - 1000, quantity=0.01, source="auto")
            order.metadata["timeframe"] = "15m"
            s.market_source_times = dict(depth=time.time(), agg_trade=time.time(), kline_1m=time.time())
            with patch.object(s, "analyze_snapshot"):
                decision = s.evaluate_candidate_pipeline(order)
            self.assertFalse(any("Weak attributed combo" in e.reason
                                 for e in decision.trace.entries if e.verdict == "VETO"))
        finally:
            s._tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
