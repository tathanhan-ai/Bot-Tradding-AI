"""Bootstrap/reconnect candle backfill with mocked REST, never an exchange request."""
from copy import deepcopy
import threading
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from trading.pipeline import MarketSnapshot, candle_end
from ui.server import LiveTradingState


class ClosedHistoryRepairTests(unittest.TestCase):
    def setUp(self):
        self.now = pd.Timestamp("2026-09-05 02:12:30").timestamp()
        self.frames = {}
        for tf, freq in (("1m", "min"), ("5m", "5min"), ("15m", "15min"), ("1h", "h")):
            end = pd.Timestamp(self.now, unit="s").floor(freq)
            index = pd.date_range(end=end, periods=251, freq=freq)[:-1]
            self.frames[tf] = pd.DataFrame({"open": 60000., "high": 60010., "low": 59990.,
                "close": 60001., "volume": 12., "quote_volume": 720012.}, index=index)
        self.state = SimpleNamespace(
            data_map=deepcopy(self.frames), market_source_times={"depth": self.now, "agg_trade": self.now, "kline_1m": self.now},
            market_lock=threading.RLock(), last_history_repair_at=0., symbol="BTCUSDT", ws_engine=object(),
            fetcher=SimpleNamespace(base_url="https://demo-fapi.binance.com", fetch_klines=Mock()),
        )
        self.state.build_market_snapshot = self.snapshot
        self.state.repair_closed_history = MethodType(LiveTradingState.repair_closed_history, self.state)

    def snapshot(self):
        return MarketSnapshot("repair", "BTCUSDT", "binance", self.now, 60000.,
            dict(self.state.market_source_times), [[59999. - i, 1.] for i in range(20)], [[60001. + i, 1.] for i in range(20)],
            frames=deepcopy(self.state.data_map), context={"required_frames": ["1m", "5m", "15m", "1h"]}, environment="testnet")

    def test_gap_repaired_from_correct_rest_host_preserves_new_ws_candle(self):
        s = self.state
        full = self.frames["1m"]
        missing = full.index[-2]  # Bootstrap had 02:09, then received 02:11.
        s.data_map["1m"] = full.drop(index=missing)
        self.assertIn("gap in 1m closed candles", s.build_market_snapshot().freshness_issues(self.now))

        def fetch(symbol, tf, limit, use_cache):
            self.assertEqual(s.fetcher.base_url, "https://demo-fapi.binance.com")
            self.assertEqual((symbol, tf, limit, use_cache), ("BTCUSDT", "1m", 250, False))
            # The live callback runs while REST is in flight and provides a newer row.
            with s.market_lock:
                newer = full.iloc[[-1]].copy()
                newer["volume"] = 17.
                s.data_map["1m"] = pd.concat([s.data_map["1m"], newer])
            forming = full.iloc[[-1]].copy()
            forming.index = pd.DatetimeIndex([candle_end(full.index[-1], "1m")])
            return pd.concat([full, forming])

        s.fetcher.fetch_klines.side_effect = fetch
        self.assertEqual(s.repair_closed_history(self.now), ["1m"])
        self.assertIn(missing, s.data_map["1m"].index)
        self.assertEqual(s.data_map["1m"].iloc[-1].volume, 17.)
        self.assertEqual(s.data_map["1m"].index[-1], full.index[-1])
        self.assertEqual(s.build_market_snapshot().freshness_issues(self.now), [])

    def test_new_closed_ws_bar_after_request_start_is_not_filtered_out(self):
        s = self.state
        s.data_map["1m"] = self.frames["1m"].drop(self.frames["1m"].index[-2])
        finish = pd.Timestamp("2026-09-05 02:13:01").timestamp()
        new_open = pd.Timestamp("2026-09-05 02:12")
        def fetch(*args, **kwargs):
            with s.market_lock:
                new_bar = self.frames["1m"].iloc[[-1]].copy()
                new_bar.index = pd.DatetimeIndex([new_open])
                new_bar["volume"] = 23.
                s.data_map["1m"] = pd.concat([s.data_map["1m"], new_bar])
            return self.frames["1m"]  # REST snapshot predates the WS close.
        s.fetcher.fetch_klines.side_effect = fetch
        with patch("ui.server.time.time", side_effect=[self.now, finish]):
            self.assertEqual(s.repair_closed_history(), ["1m"])
        self.assertEqual(s.data_map["1m"].index[-1], new_open)
        self.assertEqual(s.data_map["1m"].iloc[-1].volume, 23.)
        self.assertEqual(s.market_source_times["kline_1m"], finish)

    def test_failed_rest_keeps_veto_and_throttles_retry(self):
        s = self.state
        s.data_map["1m"] = self.frames["1m"].drop(self.frames["1m"].index[-2])
        before = s.data_map["1m"].copy()
        s.fetcher.fetch_klines.side_effect = RuntimeError("REST unavailable")
        with patch("builtins.print"):
            self.assertEqual(s.repair_closed_history(self.now), [])
            self.assertEqual(s.repair_closed_history(self.now + 14), [])
        self.assertEqual(s.fetcher.fetch_klines.call_count, 1)
        pd.testing.assert_frame_equal(s.data_map["1m"], before)
        self.assertIn("gap in 1m closed candles", s.build_market_snapshot().freshness_issues(self.now))
        s.fetcher.fetch_klines.side_effect = None
        s.fetcher.fetch_klines.return_value = self.frames["1m"]
        self.assertEqual(s.repair_closed_history(self.now + 15), ["1m"])
        self.assertEqual(s.fetcher.fetch_klines.call_count, 2)

    def test_invalid_geometry_is_vetoed_then_replaced_by_real_rest_row(self):
        s = self.state
        bad = s.data_map["1m"].index[-2]
        s.data_map["1m"].loc[bad, "high"] = 1
        self.assertIn("invalid 1m OHLCV", s.build_market_snapshot().freshness_issues(self.now))
        s.fetcher.fetch_klines.return_value = self.frames["1m"]
        self.assertEqual(s.repair_closed_history(self.now), ["1m"])
        self.assertEqual(s.data_map["1m"].loc[bad, "high"], 60010)
        self.assertEqual(s.build_market_snapshot().freshness_issues(self.now), [])

    def test_stale_required_frames_repaired_without_alpha(self):
        s = self.state
        s.data_map["1m"] = self.frames["1m"].iloc[:-3]
        s.data_map["5m"] = self.frames["5m"].iloc[:-2]
        s.market_source_times["kline_1m"] = self.now - 190
        s.fetcher.fetch_klines.side_effect = lambda symbol, tf, limit, use_cache: self.frames[tf]
        s.analyze_snapshot = Mock()
        self.assertEqual(set(s.repair_closed_history(self.now)), {"1m", "5m"})
        self.assertEqual(s.build_market_snapshot().freshness_issues(self.now), [])
        s.analyze_snapshot.assert_not_called()

    def test_environment_changed_during_rest_response_is_discarded(self):
        s = self.state
        s.data_map["1m"] = self.frames["1m"].drop(self.frames["1m"].index[-2])
        before = s.data_map["1m"].copy()
        def fetch(*args, **kwargs):
            s.fetcher.base_url = "https://fapi.binance.com"
            return self.frames["1m"]
        s.fetcher.fetch_klines.side_effect = fetch
        self.assertEqual(s.repair_closed_history(self.now), [])
        pd.testing.assert_frame_equal(s.data_map["1m"], before)

    def test_decision_cycle_calls_repair_before_entry_pipeline(self):
        calls = []
        s = SimpleNamespace(decision_lock=threading.RLock(), live_price=60000.,
            risk_manager=SimpleNamespace(sync_day=lambda: calls.append("sync")),
            reconcile_binance_testnet_orders=lambda: calls.append("reconcile"),
            process_execution_tick=lambda price: calls.append("exits"),
            repair_closed_history=lambda: calls.append("repair"),
            evaluate_ensemble_automated_decision=lambda price: calls.append("pipeline"))
        LiveTradingState.run_decision_cycle(s)
        self.assertEqual(calls, ["sync", "reconcile", "exits", "repair", "pipeline"])


if __name__ == "__main__":
    unittest.main()
