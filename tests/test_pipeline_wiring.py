"""Actual stage integration; synthetic candles are fixtures, not performance evidence."""
import tempfile
import time
import io
import json
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd

from data.persistent_storage import PersistentStorageManager
from trading.pipeline import resample_closed_candles
from ui.server import LiveTradingState


class PipelineWiringTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir="D:/Codex", prefix="pipeline-test-")
        root = Path(self.tmp.name)
        self.state = LiveTradingState(storage=PersistentStorageManager(root / "db.sqlite", root / "backup.json"))
        self.state.vibe_swarm.enabled = False
        self.state.monthly_governor.enabled = False
        self.state.decision_mode = "DETERMINISTIC_ONLY"
        now = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("min")
        for tf, rule in (("1m", "min"), ("5m", "5min"), ("15m", "15min"), ("1h", "h")):
            end = now.floor(rule) - pd.Timedelta(rule if rule[0].isdigit() else "1" + rule)
            idx = pd.date_range(end=end, periods=250, freq=rule)
            n = np.arange(250)
            price = 60000 + n * 1.0 + 130 * np.sin(n / 8)
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

    def test_actual_alpha_modules_share_snapshot_without_contract_errors(self):
        s = self.state
        candidate = s.build_candidate_order(0, "AUTO", s.live_price, 0, 0, "auto")
        decision = s.evaluate_candidate_pipeline(candidate)
        failures = [(e.stage, e.reason) for e in decision.trace.entries if "gate error" in e.reason]
        self.assertEqual(failures, [])
        self.assertIs(decision.candidate, candidate)
        self.assertTrue(any("Deterministic Guard" in e.stage for e in decision.trace.entries), [(e.stage, e.reason) for e in decision.trace.entries])
        self.assertEqual(s.ai_verdict.regime, s.deterministic_validation.corrected_regime)
        self.assertIn("daily_price_vol_pct", s.order_research.carver_output)
        self.assertLess(s.order_research.carver_output["daily_price_vol_pct"], 1)

    def test_manual_data_and_freqtrade_veto_never_calls_alpha_or_council(self):
        s = self.state
        candidate = s.build_candidate_order(1, "MARKET", s.live_price, s.live_price-500, s.live_price+1000, "manual")
        with patch.object(s, "analyze_snapshot") as alpha, patch.object(s.vibe_swarm, "evaluate_council") as council:
            s.market_source_times["depth"] -= 20
            decision = s.evaluate_candidate_pipeline(candidate)
            self.assertFalse(decision.approved)
            alpha.assert_not_called()
            council.assert_not_called()
            s.market_source_times["depth"] = time.time()
            with patch.object(s.freqtrade_protections, "validate_new_trade", return_value=(False, "MaxDD", None)):
                decision = s.evaluate_candidate_pipeline(candidate)
            self.assertIn("Freqtrade", decision.trace.entries[-1].stage)
            alpha.assert_not_called()

    def test_resample_excludes_partial_buckets_and_lookahead(self):
        frame = self.state.data_map["1m"].iloc[-31:].copy()
        # Exact bucket boundary makes the first full 15m candle available.
        frame.index = pd.date_range("2026-01-01", periods=31, freq="min")
        now = pd.Timestamp("2026-01-01 00:30").timestamp()
        result = resample_closed_candles(frame, "15m", now)
        self.assertEqual(len(result), 2)
        self.assertEqual(result.iloc[0].volume, 150)
        future = frame.copy()
        future.iloc[-1, future.columns.get_loc("close")] = 999999
        pd.testing.assert_frame_equal(result, resample_closed_candles(future, "15m", now))
        self.assertTrue(resample_closed_candles(frame.drop(frame.index[2]), "15m", now).index[0] > frame.index[0])

    def prime_directional_setup(self, alpha=30):
        s = self.state
        s.analyze_snapshot(s.build_market_snapshot())
        s.octobot_consensus.is_tradable = True
        s.octobot_consensus.recommended_direction = 1
        s.octobot_setup = None
        s.vibe_alpha_zoo.latest_metrics.composite_alpha_score = alpha
        s.order_research.carver_output.update(raw_forecast=3.0, daily_price_vol_pct=0.03)
        s.order_research.recommended_side = "BUY"
        s.order_research.structural_sl = s.live_price - 500
        s.order_research.structural_tp = s.live_price + 1000

    def test_sized_candidate_probation_cost_cap_and_loss_memory_veto(self):
        s = self.state
        self.prime_directional_setup()
        def candidate():
            return s.build_candidate_order(1, "MARKET", s.live_price, s.live_price-500, s.live_price+1000, "manual")
        with patch.object(s, "analyze_snapshot"):
            decision = s.evaluate_candidate_pipeline(candidate())
        self.assertTrue(decision.approved, [(e.stage, e.reason, e.details) for e in decision.trace.entries])
        self.assertTrue(any("AI Council" in e.stage for e in decision.trace.entries))
        self.assertGreater(decision.candidate.quantity, 0)
        self.assertLessEqual(decision.candidate.quantity * (500 + s.live_price * .0014), s.current_balance * .0025)
        # The same adverse ENTRY context must be recognized after sizing.
        s.order_flow_verdict = s.order_flow_engine.evaluate(s.live_price)
        with patch.object(type(s.order_flow_verdict), "absorption_signal", new_callable=property, fget=lambda _: "BEAR_ABSORPTION"):
            s.trade_memory.record_trade_outcome(1, 1, s.live_price, s.indicators, {"absorption_signal": "BEAR_ABSORPTION"}, {}, {"vwap_status": s.ai_verdict.vwap_status}, -10, "STOP_LOSS")
            with patch.object(s, "analyze_snapshot"):
                rejected = s.evaluate_candidate_pipeline(candidate())
        self.assertFalse(rejected.approved)
        self.assertIn("Trade Memory", rejected.trace.entries[-1].stage)
        self.assertGreater(rejected.candidate.quantity, 0)

    def test_alpha_zoo_changes_actual_carver_target_and_inertia_blocks(self):
        s = self.state
        self.prime_directional_setup(alpha=0)
        # Remove probation only for this calculation test using an actual positive trade sample.
        for i in range(30):
            s.jesse_engine.record_trade(10 if i % 4 else -5)
        def decide():
            s.market_source_times = dict(depth=time.time(), agg_trade=time.time(), kline_1m=time.time())
            return s.evaluate_candidate_pipeline(s.build_candidate_order(1, "LIMIT", s.live_price, s.live_price-500, s.live_price+1000, "manual"))
        with patch.object(s, "analyze_snapshot"):
            first = decide()
            s.vibe_alpha_zoo.latest_metrics.composite_alpha_score = 60
            second = decide()
        self.assertTrue(first.approved and second.approved)
        self.assertGreater(second.candidate.metadata["carver"]["optimal_contracts"], first.candidate.metadata["carver"]["optimal_contracts"])
        target = second.candidate.metadata["carver"]["optimal_contracts"]
        s.current_position = dict(direction=1, units=target, entry_price=s.live_price, stop_loss=s.live_price-500, margin=target*s.live_price/3, unrealized_pnl=0)
        with patch.object(s, "analyze_snapshot"):
            hold = decide()
        self.assertFalse(hold.approved)
        self.assertIn("inertia", hold.trace.entries[-1].reason)

    def test_carver_can_reduce_held_inventory_without_reverse_entry(self):
        s = self.state
        self.prime_directional_setup()
        s.order_research.recommended_type = "MARKET"
        s.order_research.optimal_price = s.live_price
        s.current_position = dict(direction=1, units=1.0, entry_price=s.live_price,
                                  stop_loss=s.live_price-500, margin=20000, unrealized_pnl=0)
        with patch.object(s, "analyze_snapshot"):
            decision = s.evaluate_candidate_pipeline(s.build_candidate_order(0, "AUTO", s.live_price, 0, 0, "auto"))
        self.assertTrue(decision.approved, [(e.stage, e.reason) for e in decision.trace.entries])
        self.assertTrue(decision.candidate.metadata["reduce_only"])
        self.assertEqual(decision.candidate.direction, -1)
        self.assertLessEqual(decision.candidate.quantity, 1.0)
        with patch.object(s.execution, "request_close", return_value=True) as close, patch.object(s.execution, "queue") as queue:
            s.queue_approved_candidate(decision.candidate)
        close.assert_called_once()
        queue.assert_not_called()

    def test_snapshot_and_dashboard_do_not_reuse_stale_flow(self):
        s = self.state
        s.order_flow_verdict = SimpleNamespace(current_cvd=-9999)
        s.order_flow_engine.add_trade(s.live_price, 3.0, False, time.time())
        snapshot = s.build_market_snapshot()
        self.assertEqual(snapshot.cvd, snapshot.context["flow"].current_cvd)
        self.assertNotEqual(snapshot.cvd, -9999)
        self.assertEqual(snapshot.price, (snapshot.bids[0][0] + snapshot.asks[0][0]) / 2)
        self.assertEqual(s.get_state_dict()["order_flow"]["current_cvd"], snapshot.cvd)
        payload = s.get_state_dict()
        self.assertTrue(payload["visual_hft"]["vpin_ready"])
        self.assertEqual(payload["visual_hft"]["completed_bucket_count"], snapshot.context["hft"].completed_bucket_count)
        self.assertFalse(payload["order_research"]["ready"])
        self.assertEqual(payload["order_research"]["win_probability"], 0)

    def test_testnet_chart_does_not_mix_mainnet_or_overwrite_ohlcv(self):
        from ui import server
        s = self.state
        s.binance_api.is_live_enabled = True
        s.binance_api.is_testnet = True
        expected = float(s.data_map["1m"].iloc[-1]["close"])
        with patch.object(server, "state", s), patch("urllib.request.urlopen") as request:
            response = server.get_klines(interval="1m")
        request.assert_not_called()
        self.assertEqual(response["exchange"], "binance")
        self.assertEqual(response["candles"][-1]["close"], expected)
        self.assertEqual(response["candles"][-1]["time"], int(s.data_map["1m"].index[-1].value // 1_000_000_000))

    def test_dashboard_streams_exchange_forming_candle_without_polluting_closed_history(self):
        s = self.state
        closed_tail = float(s.data_map["15m"].iloc[-1].close)
        s.chart_klines["15m"] = {
            "time": 1_700_000_000, "open": 60_000.0, "high": 60_100.0, "low": 59_900.0,
            "close": 60_050.0, "volume": 12.34, "quote_volume": 740_000.0,
            "is_closed": False, "event_time": 1_700_000_060_000,
        }

        chart_candle = s.get_state_dict()["chart_klines"]["15m"]

        self.assertEqual(chart_candle["volume"], 12.34)
        self.assertFalse(chart_candle["is_closed"])
        self.assertEqual(float(s.data_map["15m"].iloc[-1].close), closed_tail)

    def test_every_frame_consumed_by_ensemble_is_data_gated(self):
        s = self.state
        s.data_map["4h"] = s.data_map["1h"].copy()  # Deliberately wrong cadence.
        snapshot = s.build_market_snapshot()
        self.assertIn("4h", snapshot.context["required_frames"])
        with patch.object(s, "analyze_snapshot") as alpha:
            decision = s.evaluate_candidate_pipeline(s.build_candidate_order(0, "AUTO", s.live_price, 0, 0, "auto"))
        self.assertFalse(decision.approved)
        self.assertIn("4h", decision.trace.entries[-1].reason)
        alpha.assert_not_called()

    def test_paper_tick_keeps_tp2_after_tp1_is_complete(self):
        s = self.state
        s.current_position = dict(direction=1, units=.06, entry_price=60000, stop_loss=59000,
            take_profit=65000, margin=1200, orders=[], partial_tp_done=True,
            tp_stage_initial_units=.1, tp_stage_filled={"tp1": .04},
            staged_take_profits={"tp1":61000, "tp1_ratio":.4, "tp2":62000, "tp2_ratio":.35})
        with patch.object(s.execution, "request_close", return_value=True) as close:
            s.process_execution_tick(62000)
        self.assertEqual(close.call_args.kwargs["tp_stage"], "tp2")
        self.assertAlmostEqual(close.call_args.args[0], .035/.06)

    def test_grid_fills_keep_hedge_sides_separate_and_close_per_side(self):
        s = self.state
        long, _ = s.order_manager.place_order("BTCUSDT", "POST_ONLY", "BUY", s.live_price - 20, 60, 3,
                                              quantity=.003, stop_loss=s.live_price - 500, take_profit=s.live_price + 100,
                                              group_type="GRID", position_side="LONG", client_order_id="grid-long")
        short, _ = s.order_manager.place_order("BTCUSDT", "POST_ONLY", "SELL", s.live_price + 20, 60, 3,
                                               quantity=.003, stop_loss=s.live_price + 500, take_profit=s.live_price - 100,
                                               group_type="GRID", position_side="SHORT", client_order_id="grid-short")
        s.execution.apply_entry(long, .003, long.price)
        s.execution.apply_entry(short, .003, short.price)
        self.assertIsNone(s.current_position)
        self.assertEqual(set(s.hedge_positions), {"LONG", "SHORT"})
        self.assertEqual(s.hedge_positions["LONG"]["direction"], 1)
        self.assertEqual(s.hedge_positions["SHORT"]["direction"], -1)
        self.assertTrue(s.execution.request_close(1.0, "GRID_TEST_CLOSE", position_side="LONG"))
        self.assertNotIn("LONG", s.hedge_positions)
        self.assertIn("SHORT", s.hedge_positions)


if __name__ == "__main__":
    unittest.main()
