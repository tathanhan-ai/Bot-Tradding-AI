"""Synthetic TEST-ONLY fixtures verify contracts, never trading performance."""
from copy import deepcopy
import json
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch


from trading.replay import (RecordedExecution, ReplayDataError, ReplayStorage,
                            _feed, _state_class, depth_vwap, performance_metrics,
                            replay_clock, run_replay, snapshot_from_event)


def synthetic_event(timestamp=1767268800.0, price=50000.0):
    """Fabricated data ONLY for invariants; never an OOS performance dataset."""
    frames = {}
    for tf, duration in (("1m", 60), ("5m", 300), ("15m", 900), ("1h", 3600)):
        end = math.floor(timestamp / duration) * duration
        frames[tf] = [{"time": end - duration * (240 - i), "open": price,
                       "high": price + 100, "low": price - 100, "close": price,
                       "volume": 100, "is_closed": True} for i in range(240)]
    trades = [{"id": f"{timestamp}:{i}", "timestamp": timestamp - .6 + i * .01,
               "price": price, "quantity": 1, "is_buyer_maker": bool(i % 2)} for i in range(60)]
    return {"snapshot_id": f"test-{timestamp}", "captured_at": timestamp,
            "symbol": "BTCUSDT", "exchange": "binance", "price": price,
            "source_times": {"depth": timestamp, "agg_trade": trades[-1]["timestamp"],
                             "kline_1m": math.floor(timestamp / 60) * 60},
            "bids": [[price - 1 - i, 1] for i in range(20)],
            "asks": [[price + 1 + i, 1] for i in range(20)], "frames": frames,
            "agg_trades": trades, "provenance": {"kind": "test_only"}}


class ReplayContractsTest(unittest.TestCase):
    def test_l2_missing_cannot_be_replaced_with_top_of_book(self):
        event = synthetic_event()
        event["bids"], event["asks"] = event["bids"][:1], event["asks"][:1]
        with self.assertRaisesRegex(ReplayDataError, "20-level"):
            snapshot_from_event(event)

    def test_any_future_or_forming_candle_is_rejected(self):
        event = synthetic_event()
        event["frames"]["15m"][-1]["time"] += 900
        with self.assertRaisesRegex(ReplayDataError, "look-ahead"):
            snapshot_from_event(event)
        event = synthetic_event()
        event["frames"]["1h"][0]["is_closed"] = False
        with self.assertRaisesRegex(ReplayDataError, "closed-candle"):
            snapshot_from_event(event)

    def test_stale_source_wrong_exchange_and_gap_rejected(self):
        for mutation, expected in ((lambda e: e["source_times"].update(depth=e["captured_at"] - 4), "depth age"),
                                   (lambda e: e.update(exchange="mexc"), "exchange=mexc"),
                                   (lambda e: e["frames"]["1m"].pop(-10), "gap")):
            event = synthetic_event()
            mutation(event)
            with self.assertRaisesRegex(ReplayDataError, expected):
                snapshot_from_event(event)

    def test_depth_vwap_consumes_real_levels_and_never_extrapolates(self):
        self.assertAlmostEqual(depth_vwap([[101, 1], [102, 2]], 2), 101.5)
        with self.assertRaisesRegex(ReplayDataError, "insufficient"):
            depth_vwap([[101, 1]], 2)

    def test_metrics_are_net_with_defined_zero_trade_case(self):
        metrics = performance_metrics(1000, 1010, [1000, 900, 1010],
                                      [{"pnl": 20, "fee": 2}, {"pnl": -10, "fee": 1}],
                                      [{"notional": 2000, "slippage_usdt": 2}])
        self.assertAlmostEqual(metrics["net_return_pct"], 1)
        self.assertEqual(metrics["max_drawdown_pct"], 10)
        self.assertEqual(metrics["profit_factor"], 2)
        self.assertEqual(metrics["turnover_multiple"], 2)
        self.assertEqual(metrics["slippage_bps"], 10)
        zero = performance_metrics(1000, 1000, [1000], [], [])
        self.assertIsNone(zero["profit_factor"])
        json.dumps(zero, allow_nan=False)

    def test_rewritten_closed_history_and_reverse_event_order_fail(self):
        events = [synthetic_event(), synthetic_event(1767269700)]
        events[1]["frames"]["1h"][0]["volume"] = 101
        with self.assertRaisesRegex(ReplayDataError, "rewritten"):
            run_replay(events)
        with self.assertRaisesRegex(ReplayDataError, "chronological"):
            run_replay([synthetic_event(1767269700), synthetic_event()])

    def test_future_and_missing_raw_trades_fail_before_analysis(self):
        state_type = _state_class()
        event = synthetic_event()
        clock = SimpleNamespace(value=event["captured_at"])
        with replay_clock(clock):
            state = state_type(storage=ReplayStorage())
            bad = deepcopy(event)
            bad["agg_trades"] = []
            with self.assertRaisesRegex(ReplayDataError, "raw aggTrade missing"):
                _feed(state, snapshot_from_event(bad), bad, set(), clock)
            bad = deepcopy(event)
            bad["agg_trades"][0]["timestamp"] = event["captured_at"] + 1
            with self.assertRaisesRegex(ReplayDataError, "future aggTrade"):
                _feed(state, snapshot_from_event(bad), bad, set(), clock)

    def test_offline_replay_calls_actual_pipeline_for_is_and_oos(self):
        state_type = _state_class()
        original = state_type.evaluate_candidate_pipeline
        calls = []

        def record(state, *args, **kwargs):
            result = original(state, *args, **kwargs)
            calls.append(result)
            return result

        events = [synthetic_event(1767268800 + i * 900) for i in range(4)]
        with patch.object(state_type, "evaluate_candidate_pipeline", record):
            report = run_replay(events, split_fraction=.5)
        self.assertEqual(len(calls), 4)
        self.assertEqual(report["status"], "TEST_ONLY_NOT_PERFORMANCE_EVIDENCE")
        self.assertFalse(report["comparison"]["promotion_allowed"])
        self.assertFalse(report["comparison"]["tuning_performed"])
        self.assertEqual(report["results"]["out_of_sample"]["pipeline"]["equity"][0], 5000)
        self.assertTrue(all(call.trace.entries[0].stage == "Stage 0 / Market Snapshot" for call in calls))
        # Flat test fixture may legitimately veto. Never invent fills to improve it.
        self.assertTrue(all(call.trace.entries[0].verdict == "PASS" for call in calls))
        json.dumps(report, allow_nan=False)

    def test_shared_execution_prices_fees_and_feedback_use_replay_clock(self):
        state_type = _state_class()
        event = synthetic_event()
        clock = SimpleNamespace(value=event["captured_at"])
        with replay_clock(clock):
            state = state_type(storage=ReplayStorage())
            _feed(state, snapshot_from_event(event), event, set(), clock)
            adapter = RecordedExecution(state)
            order = SimpleNamespace(direction=1, order_type="MARKET", side="BUY", candidate_payload={},
                                    execution_group="test-only-entry", group_type="",
                                    client_order_id="test-only-entry", order_id="test-only-entry",
                                    applied_quantity=0, exchange_order_id="paper-test", parent_intent_id="test-only-entry",
                                    timeframe="15m", leverage=3, stop_loss=49000, take_profit=52000)
            state.execution.apply_entry(order, 1.5, event["price"])
            self.assertAlmostEqual(state.current_position["entry_price"], (50001 + 50002 * .5) / 1.5)
            self.assertEqual(state.current_position["open_timestamp"], event["captured_at"])
            state.execution.request_close(1.0, "TEST_ONLY")
            self.assertEqual(len(state.trades), 1)
            self.assertLess(state.current_balance, 5000)
            self.assertEqual(state.trades[0]["closed_at_ts"], event["captured_at"])
            self.assertTrue(state.trades[0]["exit_time"].startswith("2026-01-01"))
            self.assertEqual(state.monthly_governor.current_month_str, "2026-01")
            self.assertEqual(len(adapter.fills), 2)
            self.assertAlmostEqual(state.current_balance - 5000, state.trades[0]["pnl"])
            self.assertTrue(state.storage.load_execution_runtime()["processed_execution_ids"])


if __name__ == "__main__":
    unittest.main()
