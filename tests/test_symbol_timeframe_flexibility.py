# -*- coding: utf-8 -*-
import unittest
import asyncio
from types import SimpleNamespace
from unittest.mock import patch, AsyncMock
import pandas as pd
import numpy as np
import time

from ui.server import (
    LiveTradingState, get_symbol_spec, SYMBOL_SPECIFICATIONS,
    set_symbol, set_timeframe
)
from security import Role
from trading.pipeline import MarketSnapshot, CandidateOrder
from risk.ai_order_researcher import AIOrderResearcher


class TestSymbolTimeframeFlexibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create minimal state
        cls.state = LiveTradingState(symbol="BTCUSDT", balance=5000.0)
        cls.state.is_running = True

        # Build mock frames for BTCUSDT
        cls.btc_price = 80000.0
        dates = pd.date_range("2026-01-01", periods=100, freq="15min")
        df_15m = pd.DataFrame({
            "open": np.full(100, cls.btc_price),
            "high": np.full(100, cls.btc_price + 100.0),
            "low": np.full(100, cls.btc_price - 100.0),
            "close": np.full(100, cls.btc_price),
            "volume": np.full(100, 50.0),
        }, index=dates)
        cls.state.data_map["15m"] = df_15m

        dates_1m = pd.date_range("2026-01-01", periods=100, freq="1min")
        df_1m = pd.DataFrame({
            "open": np.full(100, cls.btc_price),
            "high": np.full(100, cls.btc_price + 20.0),
            "low": np.full(100, cls.btc_price - 20.0),
            "close": np.full(100, cls.btc_price),
            "volume": np.full(100, 10.0),
        }, index=dates_1m)
        cls.state.data_map["1m"] = df_1m

        dates_1h = pd.date_range("2026-01-01", periods=100, freq="1h")
        df_1h = pd.DataFrame({
            "open": np.full(100, cls.btc_price),
            "high": np.full(100, cls.btc_price + 400.0),
            "low": np.full(100, cls.btc_price - 400.0),
            "close": np.full(100, cls.btc_price),
            "volume": np.full(100, 200.0),
        }, index=dates_1h)
        cls.state.data_map["1h"] = df_1h

        cls.state.live_price = cls.btc_price
        cls.state.latest_l2_bids = [[cls.btc_price - .1 - n, 4.0] for n in range(20)]
        cls.state.latest_l2_asks = [[cls.btc_price + .1 + n, 4.0] for n in range(20)]
        cls.state.visual_hft.update_order_book(cls.state.latest_l2_bids, cls.state.latest_l2_asks)
        for _ in range(6):
            cls.state.visual_hft.update_trade(cls.btc_price, 15.0, False)
            cls.state.visual_hft.update_trade(cls.btc_price, 15.0, True)
        cls.state.market_source_times = dict(depth=time.time(), agg_trade=time.time(), kline_1m=time.time())
        cls.state.fee_engine.update_book(cls.btc_price - .1, cls.btc_price + .1)

        cls.researcher = AIOrderResearcher()

    def test_get_symbol_spec_supported_and_dynamic(self):
        btc_spec = get_symbol_spec("BTCUSDT", 80000.0)
        self.assertEqual(btc_spec["step_size"], 0.001)
        self.assertEqual(btc_spec["dca_min_qty"], 0.003)

        eth_spec = get_symbol_spec("ETHUSDT", 2500.0)
        self.assertEqual(eth_spec["step_size"], 0.01)
        self.assertEqual(eth_spec["dca_min_qty"], 0.03)

        sol_spec = get_symbol_spec("SOLUSDT", 150.0)
        self.assertEqual(sol_spec["step_size"], 0.1)
        self.assertEqual(sol_spec["dca_min_qty"], 0.3)

        doge_spec = get_symbol_spec("DOGEUSDT", 0.15)
        self.assertEqual(doge_spec["step_size"], 1.0)
        self.assertEqual(doge_spec["dca_min_qty"], 3.0)

        # Dynamic heuristic for custom token
        pepe_spec = get_symbol_spec("PEPEUSDT", 0.00001)
        self.assertEqual(pepe_spec["step_size"], 1.0)
        self.assertEqual(pepe_spec["min_notional"], 5.0)

    def test_timeframe_calibration_in_order_research(self):
        # Compare 1m (scalp) vs 15m (intraday) vs 1h (swing)
        res_1m = self.researcher.research(
            current_price=self.btc_price,
            best_bid=self.btc_price - 1.0,
            best_ask=self.btc_price + 1.0,
            spread=2.0,
            indicators={"atr": 50.0, "rsi": 50.0, "adx": 20.0},
            ai_verdict=None,
            ensemble_result=None,
            ai_cro=None,
            current_balance=5000.0,
            df_structure=self.state.data_map["1m"],
            df_macro=self.state.data_map["1h"],
            active_timeframe="1m"
        )
        self.assertEqual(res_1m.active_timeframe, "1m")
        self.assertLessEqual(res_1m.optimal_callback_pct, 0.65)
        self.assertEqual(res_1m.dca_ladder_step, 0.0025)
        self.assertIn("SCALP_1M", res_1m.execution_horizon)

        res_15m = self.researcher.research(
            current_price=self.btc_price,
            best_bid=self.btc_price - 1.0,
            best_ask=self.btc_price + 1.0,
            spread=2.0,
            indicators={"atr": 200.0, "rsi": 50.0, "adx": 20.0},
            ai_verdict=None,
            ensemble_result=None,
            ai_cro=None,
            current_balance=5000.0,
            df_structure=self.state.data_map["15m"],
            df_macro=self.state.data_map["1h"],
            active_timeframe="15m"
        )
        self.assertEqual(res_15m.active_timeframe, "15m")
        self.assertEqual(res_15m.dca_ladder_step, 0.005)
        self.assertIn("INTRADAY_15M", res_15m.execution_horizon)

        res_1h = self.researcher.research(
            current_price=self.btc_price,
            best_bid=self.btc_price - 1.0,
            best_ask=self.btc_price + 1.0,
            spread=2.0,
            indicators={"atr": 600.0, "rsi": 50.0, "adx": 20.0},
            ai_verdict=None,
            ensemble_result=None,
            ai_cro=None,
            current_balance=5000.0,
            df_structure=self.state.data_map["1h"],
            df_macro=self.state.data_map["1h"],
            active_timeframe="1h"
        )
        self.assertEqual(res_1h.active_timeframe, "1h")
        self.assertGreaterEqual(res_1h.optimal_callback_pct, 0.8)
        self.assertEqual(res_1h.dca_ladder_step, 0.010)
        self.assertIn("SWING_1H", res_1h.execution_horizon)

    def test_sizing_gate_supports_eth_and_sol_lot_sizes(self):
        # Verify that ETH, SOL, and DOGE lot sizes and precision adapt dynamically
        spec_eth = get_symbol_spec("ETHUSDT", 2500.0)
        self.assertEqual(spec_eth["step_size"], 0.01)
        self.assertEqual(spec_eth["min_qty"], 0.01)
        self.assertEqual(spec_eth["dca_min_qty"], 0.03)

        # ETH quantity 0.1234 correctly clamps to 0.12
        raw_qty = 0.1234
        clamped_eth = round(int((raw_qty + 1e-12) / spec_eth["step_size"]) * spec_eth["step_size"], 4)
        self.assertEqual(clamped_eth, 0.12)

        # SOL quantity 1.25 correctly clamps to 1.2
        spec_sol = get_symbol_spec("SOLUSDT", 150.0)
        self.assertEqual(spec_sol["step_size"], 0.1)
        clamped_sol = round(int((1.25 + 1e-12) / spec_sol["step_size"]) * spec_sol["step_size"], 2)
        self.assertEqual(clamped_sol, 1.2)

        # DOGE quantity 55.8 correctly clamps to 55.0
        spec_doge = get_symbol_spec("DOGEUSDT", 0.25)
        self.assertEqual(spec_doge["step_size"], 1.0)
        clamped_doge = int((55.8 + 1e-12) / spec_doge["step_size"]) * spec_doge["step_size"]
        self.assertEqual(clamped_doge, 55.0)

    def test_sizing_gate_dca_ladder_guard_adapts_to_symbol(self):
        # Verify DCA Ladder minimum size guard enforces symbol-specific minimums
        spec_btc = get_symbol_spec("BTCUSDT", 80000.0)
        spec_sol = get_symbol_spec("SOLUSDT", 150.0)
        spec_doge = get_symbol_spec("DOGEUSDT", 0.25)

        self.assertEqual(spec_btc["dca_min_qty"], 0.003)
        self.assertEqual(spec_sol["dca_min_qty"], 0.3)
        self.assertEqual(spec_doge["dca_min_qty"], 3.0)

        # For SOL: 0.15 is below dca_min_qty (0.3), so guard triggers
        self.assertLess(0.15, spec_sol["dca_min_qty"])
        self.assertGreaterEqual(0.5, spec_sol["dca_min_qty"])

    def test_set_timeframe_api(self):
        from ui.server import state
        # Ensure server singleton state has 1m data loaded
        state.data_map["1m"] = self.state.data_map["1m"]
        res = set_timeframe("1m")
        self.assertEqual(res["status"], "ok")
        self.assertEqual(res["active_timeframe"], "1m")

        res_invalid = set_timeframe("invalid_tf")
        self.assertEqual(res_invalid["status"], "error")

    def test_set_symbol_api_safety_and_switching(self):
        from ui.server import state
        session = SimpleNamespace(user="admin", role=Role.ADMIN)

        # 1. Reject invalid pair
        res_bad = asyncio.run(set_symbol("BTCBUSD_WRONG", session=session))
        self.assertEqual(res_bad["status"], "rejected")

        # 2. Reject if open position exists
        state.current_position = {"units": 0.01, "entry_price": 80000.0, "direction": 1}
        res_exp = asyncio.run(set_symbol("ETHUSDT", session=session))
        self.assertEqual(res_exp["status"], "rejected")
        self.assertIn("đang có vị thế", res_exp["reason"])
        state.current_position = None
        state.hedge_positions = {}
        saved_orders = list(state.order_manager.orders)
        state.order_manager.orders.clear()

        # 3. Successful symbol switch
        try:
            with patch.object(state, "restart_market_data", new_callable=AsyncMock) as mock_restart:
                res_ok = asyncio.run(set_symbol("ETHUSDT", session=session))
                self.assertEqual(res_ok["status"], "ok")
                self.assertEqual(state.symbol, "ETHUSDT")
                self.assertEqual(state.strategy_config.symbol, "ETHUSDT")
                self.assertEqual(state.fee_engine.symbol, "ETHUSDT")
                mock_restart.assert_awaited_once()

            # Switch back to BTCUSDT for other tests
            with patch.object(state, "restart_market_data", new_callable=AsyncMock):
                asyncio.run(set_symbol("BTCUSDT", session=session))
        finally:
            state.order_manager.orders.extend(saved_orders)


if __name__ == "__main__":
    unittest.main()
