import unittest
import tempfile
import time
from pathlib import Path
import numpy as np
import pandas as pd

from data.persistent_storage import PersistentStorageManager
from risk.ai_order_researcher import AIOrderResearcher, AIOrderResearchResult
from trading.pipeline import CandidateOrder
from ui.server import LiveTradingState


class TestCarverHoldAndFeeSafe(unittest.TestCase):
    def setUp(self):
        self.researcher = AIOrderResearcher()
        self.tmp = tempfile.TemporaryDirectory(prefix="carver-hold-")
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

    def test_carver_rebalance_blocked_under_minimum_hold_time(self):
        p = self.state.live_price
        # Position opened only 60 seconds ago
        self.state.current_position = {
            "direction": 1,
            "units": 0.02,
            "entry_price": p - 300.0,  # in profit
            "stop_loss": p - 800.0,
            "take_profit": p + 1000.0,
            "margin": 320.0,
            "unrealized_pnl": 6.0,
            "open_timestamp": time.time() - 60.0  # 60s old < 900s
        }
        order = CandidateOrder(
            order_id="test-carver-rebalance-young",
            symbol="BTCUSDT",
            direction=-1,
            order_type="POST_ONLY",
            entry_price=p,
            stop_loss=p - 500.0,
            take_profit=p + 1000.0,
            quantity=0.01,
            source="auto"
        )
        decision = self.state.evaluate_candidate_pipeline(order)
        self.assertFalse(decision.approved)
        last_stage = decision.trace.entries[-1]
        self.assertIn("minimum hold time", last_stage.reason)

    def test_carver_rebalance_blocked_on_losing_position(self):
        p = self.state.live_price
        # Position opened 1200 seconds ago (mature), but in loss
        self.state.current_position = {
            "direction": 1,
            "units": 0.02,
            "entry_price": p + 100.0,  # in loss by 100 points
            "stop_loss": p - 800.0,
            "take_profit": p + 1000.0,
            "margin": 320.0,
            "unrealized_pnl": -2.0,
            "open_timestamp": time.time() - 1200.0
        }
        order = CandidateOrder(
            order_id="test-carver-rebalance-losing",
            symbol="BTCUSDT",
            direction=-1,
            order_type="POST_ONLY",
            entry_price=p,
            stop_loss=p - 500.0,
            take_profit=p + 1000.0,
            quantity=0.01,
            source="auto"
        )
        decision = self.state.evaluate_candidate_pipeline(order)
        self.assertFalse(decision.approved)
        last_stage = decision.trace.entries[-1]
        self.assertIn("losing position", last_stage.reason)

    def test_carver_rebalance_blocked_on_tiny_profit_below_fee_safe_threshold(self):
        p = self.state.live_price
        # Position opened 1200 seconds ago, profit only +20 points (below 0.75 * ATR)
        self.state.indicators["atr"] = 150.0  # 0.75 * 150 = 112.5 points needed
        self.state.current_position = {
            "direction": 1,
            "units": 0.02,
            "entry_price": p - 20.0,  # profit +20 points
            "stop_loss": p - 800.0,
            "take_profit": p + 1000.0,
            "margin": 320.0,
            "unrealized_pnl": 0.40,
            "open_timestamp": time.time() - 1200.0
        }
        order = CandidateOrder(
            order_id="test-carver-rebalance-tiny-profit",
            symbol="BTCUSDT",
            direction=-1,
            order_type="POST_ONLY",
            entry_price=p,
            stop_loss=p - 500.0,
            take_profit=p + 1000.0,
            quantity=0.01,
            source="auto"
        )
        decision = self.state.evaluate_candidate_pipeline(order)
        self.assertFalse(decision.approved)
        last_stage = decision.trace.entries[-1]
        self.assertIn("fee-safe threshold", last_stage.reason)

    def test_carver_rebalance_allowed_when_mature_and_profitable(self):
        p = self.state.live_price
        # Position opened 1200 seconds ago, profit +300 points (well above 0.75 * ATR)
        self.state.indicators["atr"] = 150.0
        self.state.current_position = {
            "direction": 1,
            "units": 0.02,
            "entry_price": p - 300.0,  # profit +300 points
            "stop_loss": p - 800.0,
            "take_profit": p + 1000.0,
            "margin": 320.0,
            "unrealized_pnl": 6.0,
            "open_timestamp": time.time() - 1200.0
        }
        order = CandidateOrder(
            order_id="test-carver-rebalance-allowed",
            symbol="BTCUSDT",
            direction=-1,
            order_type="POST_ONLY",
            entry_price=p,
            stop_loss=p - 500.0,
            take_profit=p + 1000.0,
            quantity=0.01,
            source="auto"
        )
        self.state.order_research = AIOrderResearchResult(
            recommended_type="POST_ONLY",
            recommended_side="BUY",
            optimal_price=p,
            optimal_margin=300.0,
            optimal_leverage=3,
            structural_sl=p - 500.0,
            structural_tp=p + 1000.0,
            carver_output={
                "daily_price_vol_pct": 0.02,
                "raw_forecast": -10.0,
                "capped_forecast": -10.0
            }
        )
        decision = self.state.evaluate_candidate_pipeline(order)
        self.assertTrue(decision.approved)

    def test_adaptive_counter_proposal_enforces_minimum_sl_floor(self):
        candidate = CandidateOrder(
            order_id="test-order-sl-floor",
            symbol="BTCUSDT",
            direction=1,
            order_type="MARKET",
            entry_price=80000.0,
            stop_loss=79955.0,  # dangerously tight (45 points)
            take_profit=80500.0,
            leverage=4,
            quantity=0.01,
            margin=200.0
        )
        df_structure = pd.DataFrame([
            {"open": 79990, "high": 80010, "low": 79960, "close": 80000, "volume": 10}
        ])
        indicators = {"atr": 100.0, "rsi": 50.0}

        alt = self.researcher.build_adaptive_counter_proposal(
            original_candidate=candidate,
            veto_stage="Stage 3 / AI Council",
            veto_reason="Risk: Stop loss is too tight",
            current_price=80000.0,
            best_bid=79995.0,
            best_ask=80000.0,
            indicators=indicators,
            df_structure=df_structure,
            current_balance=5000.0
        )
        self.assertIsNotNone(alt)
        sl_distance = abs(alt.entry_price - alt.stop_loss)
        self.assertGreaterEqual(sl_distance, 250.0)


if __name__ == "__main__":
    unittest.main()
