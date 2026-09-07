import unittest
import tempfile
import time
from pathlib import Path
import numpy as np
import pandas as pd
from unittest.mock import MagicMock

from data.persistent_storage import PersistentStorageManager
from risk.ai_order_researcher import AIOrderResearcher
from trading.pipeline import CandidateOrder
from ui.server import LiveTradingState


class TestFixRecent10TradesFlaws(unittest.TestCase):
    def setUp(self):
        self.researcher = AIOrderResearcher()
        self.tmp = tempfile.TemporaryDirectory(prefix="flaws-test-")
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

    def test_5m_timeframe_leakage_blocked(self):
        confluence = MagicMock()
        confluence.tactical_timeframe = "5m"
        confluence.radar_score = 45.0
        confluence.confluence_score = 45.0

        res = self.researcher.research(
            current_price=80000.0,
            best_bid=79999.0,
            best_ask=80001.0,
            spread=2.0,
            indicators={"atr": 100.0, "rsi": 50.0, "adx": 25.0},
            ai_verdict=None,
            ensemble_result=None,
            ai_cro=None,
            active_timeframe="15m",
            candle_confluence=confluence,
            current_balance=5000.0
        )
        self.assertEqual(res.active_timeframe, "15m")
        self.assertNotIn("5M", res.strategy_horizon)

    def test_overbought_rsi_anti_fomo(self):
        indicators = {
            "atr": 100.0,
            "rsi": 66.0,  # Overbought like trade #37
            "adx": 30.0,
            "resistance": 80100.0,
            "support": 79500.0
        }
        res = self.researcher.research(
            current_price=80000.0,
            best_bid=79999.0,
            best_ask=80001.0,
            spread=2.0,
            indicators=indicators,
            ai_verdict=None,
            ensemble_result=None,
            ai_cro=None,
            active_timeframe="15m",
            current_balance=5000.0
        )
        # Should not buy market at peak: either SIDEWAY or POST_ONLY with deep discount
        if res.recommended_side == "BUY":
            self.assertNotEqual(res.recommended_type, "MARKET")
            self.assertLess(res.optimal_price, 80000.0 - 30.0)
        else:
            self.assertIn(res.recommended_side, ("SIDEWAY", "SELL"))

    def test_oversold_rsi_anti_fomo(self):
        indicators = {
            "atr": 100.0,
            "rsi": 32.0,  # Oversold
            "adx": 30.0,
            "resistance": 80500.0,
            "support": 79900.0
        }
        res = self.researcher.research(
            current_price=80000.0,
            best_bid=79999.0,
            best_ask=80001.0,
            spread=2.0,
            indicators=indicators,
            ai_verdict=None,
            ensemble_result=None,
            ai_cro=None,
            active_timeframe="15m",
            current_balance=5000.0
        )
        if res.recommended_side == "SELL":
            self.assertNotEqual(res.recommended_type, "MARKET")
            self.assertGreater(res.optimal_price, 80000.0 + 30.0)
        else:
            self.assertIn(res.recommended_side, ("SIDEWAY", "BUY"))

    def test_staged_tp_not_created_for_small_orders(self):
        order = CandidateOrder(
            order_id="test-small-order",
            symbol="BTCUSDT",
            direction=1,
            order_type="POST_ONLY",
            entry_price=self.state.live_price,
            stop_loss=self.state.live_price - 500.0,
            take_profit=self.state.live_price + 1200.0,
            quantity=0.005,  # Small position < 0.020 BTC
            leverage=5,
            source="auto"
        )
        decision = self.state.evaluate_candidate_pipeline(order)
        self.assertEqual(decision.candidate.staged_take_profits, {})

    def test_carver_loss_locking_guard_blocks_dumping_negative_position(self):
        # Position in unrealized loss (-300 price move)
        p = self.state.live_price
        self.state.current_position = {
            "direction": 1,
            "units": 0.02,
            "entry_price": p + 300.0,
            "stop_loss": p - 500.0,
            "take_profit": p + 1000.0,
            "margin": 320.0
        }
        order = CandidateOrder(
            order_id="test-carver-rebalance",
            symbol="BTCUSDT",
            direction=-1,
            order_type="POST_ONLY",
            entry_price=p,
            stop_loss=p - 500.0,
            take_profit=p + 1000.0,
            quantity=0.01,
            source="auto"
        )
        self.state.carver_output = {
            "daily_cash_vol_target": 100.0,
            "instrument_value_vol": 10.0,
            "contracts_to_execute": -0.01
        }
        decision = self.state.evaluate_candidate_pipeline(order)
        self.assertFalse(decision.approved)
        last_stage = decision.trace.entries[-1]
        self.assertIn("losing position", last_stage.reason)


if __name__ == "__main__":
    unittest.main()
