import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from data.persistent_storage import PersistentStorageManager
from strategy.ensemble_strategy import EnsembleCoordinator
from trading.pipeline import CandidateOrder
from ui.server import LiveTradingState


def make_state():
    tmp = tempfile.TemporaryDirectory(prefix="layer-opt-")
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


def prime(s, alpha=-60, direction=-1, jesse_wins=30):
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
    s.order_research.carver_output.update(raw_forecast=-3.0, daily_price_vol_pct=0.03)
    s.market_source_times = dict(depth=time.time(), agg_trade=time.time(), kline_1m=time.time())


class LayerOptimizationTests(unittest.TestCase):
    def test_ensemble_weights_normalized(self):
        coord = EnsembleCoordinator()
        now = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("min")
        data_map = {}
        for tf, rule in (("15m", "15min"), ("1h", "h")):
            idx = pd.date_range(end=now, periods=100, freq=rule)
            n = np.arange(100)
            price = 80000 + n * 2.0
            data_map[tf] = pd.DataFrame(dict(open=price - 5, high=price + 30, low=price - 30,
                                              close=price, volume=np.full(100, 10.0)), index=idx)
        for tf in ("1m", "4h"):
            res = coord.evaluate_ensemble(data_map, 80200.0, "RANGING_SIDEWAY", active_timeframe=tf)
            total = sum(v.weight for v in res.votes)
            self.assertAlmostEqual(total, 1.0, places=6)

    def test_ensemble_weak_zone_verdict(self):
        coord = EnsembleCoordinator()
        now = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("min")
        idx = pd.date_range(end=now, periods=100, freq="15min")
        n = np.arange(100)
        price = 80000 + n * 0.5 + 20 * np.sin(n / 3)
        data_map = {"15m": pd.DataFrame(dict(open=price - 5, high=price + 30, low=price - 30,
                                              close=price, volume=np.full(100, 10.0)), index=idx),
                    "1h": pd.DataFrame(dict(open=price - 5, high=price + 30, low=price - 30,
                                             close=price, volume=np.full(100, 10.0)), index=idx)}
        res = coord.evaluate_ensemble(data_map, float(price[-1]), "RANGING_SIDEWAY", active_timeframe="15m")
        self.assertIn(res.consensus_verdict, ("STRONG_LONG", "STRONG_SHORT", "WEAK_LONG", "WEAK_SHORT", "SIDEWAY_GRID"))
        if res.consensus_verdict == "SIDEWAY_GRID":
            self.assertLessEqual(res.confidence, 75)

    def test_veto_streak_downgrades_conditional(self):
        s = make_state()
        try:
            prime(s)
            s.veto_streak["CONDITIONAL:-1"] = {"count": 3, "last_veto_ts": time.time()}
            s.order_research.recommended_side = "SELL"
            s.order_research.recommended_type = "CONDITIONAL"
            s.order_research.optimal_price = s.live_price + 50
            s.order_research.structural_sl = s.live_price + 500
            s.order_research.structural_tp = s.live_price - 1000
            s.order_research.win_probability = 75
            s.order_research.optimal_margin = 0
            from trading.pipeline import PipelineDecision, DecisionTrace, StageOutcome
            with patch.object(s, "evaluate_candidate_pipeline") as ev, \
                 patch.object(s, "queue_approved_candidate", return_value={"status": "pending"}):
                cand = s.build_candidate_order(-1, "POST_ONLY", s.live_price, s.live_price + 500,
                                               s.live_price - 1000, "auto")
                trace = DecisionTrace(cand.order_id, "snap")
                trace.add("Stage 1 / Test", StageOutcome.pass_("ok"))
                ev.return_value = PipelineDecision(candidate=cand, trace=trace)
                s.evaluate_ensemble_automated_decision(s.live_price)
                called = ev.call_args[0][0]
                self.assertEqual(called.order_type, "POST_ONLY")
        finally:
            s._tmp.cleanup()

    def test_memory_downgrade_under_jesse_lock(self):
        s = make_state()
        try:
            prime(s, jesse_wins=0)
            for _ in range(4):
                s.jesse_engine.record_trade(-5)
            self.assertGreaterEqual(s.jesse_engine.compute_metrics().current_consecutive_losses, 3)
            # Ghi 1 loss de memory co gi so sanh
            s.trade_memory.record_trade_outcome(999, -1, s.live_price, s.indicators,
                                                {"absorption_signal": "BEAR_ABSORPTION"}, {"structure": "RANGING"},
                                                {"vwap_status": getattr(s.ai_verdict, "vwap_status", "EQUILIBRIUM_FAIR")},
                                                -10, "STOP_LOSS")
            order = CandidateOrder(order_id="mem1", symbol="BTCUSDT", direction=-1, order_type="POST_ONLY",
                                   entry_price=s.live_price, stop_loss=s.live_price + 500,
                                   take_profit=s.live_price - 1000, quantity=0.01, source="auto")
            order.metadata["timeframe"] = "15m"
            s.market_source_times = dict(depth=time.time(), agg_trade=time.time(), kline_1m=time.time())
            with patch.object(s, "analyze_snapshot"):
                decision = s.evaluate_candidate_pipeline(order)
            # Duoi khoa kep: khong duoc veto han vi memory (phai probation hoac duyet/veto ly do khac)
            mem_vetoes = [e for e in decision.trace.entries
                          if e.verdict == "VETO" and "Stage 5" in e.stage and "MEMORY VETO" in e.reason]
            self.assertEqual(mem_vetoes, [])
        finally:
            s._tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
