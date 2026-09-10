"""User-data stream: fill ve tuc thi (<1s), khong doi reconcile 0.5s.

- Parser ORDER_TRADE_UPDATE / ACCOUNT_UPDATE dung dinh dang Binance.
- apply_user_fill: khop lenh local theo orderId/clientOrderId, apply fill
  ngay (khong doi tick reconcile).
- apply_user_account: cap nhat UPNL theo san ngay.
- listenKey: tao/gia han/URL dung endpoint.
"""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from data.binance_userdata_stream import (
    BinanceUserDataStream,
    parse_account_update,
    parse_order_update,
)
from trading.execution import ExecutionLifecycle


def make_state():
    api = SimpleNamespace(is_live_enabled=True, is_testnet=False)
    api.normalize_order_values = Mock(side_effect=lambda s, q, price=None, **k: (True, {"quantity": q, "price": price}))
    api.prepare_testnet_trading = Mock(return_value=(True, {"maxNotionalValue": "1000000"}))
    api.test_connection = Mock(return_value={"success": True, "can_trade": True, "wallet_balance": 100.0, "available_balance": 100.0})
    api.place_order_live = Mock(return_value=(True, {"orderId": "1", "status": "NEW"}))
    api.query_order = Mock(return_value=(False, {"code": -2013}))
    api.cancel_order = Mock(return_value=(False, {"code": -2011}))
    api._send_request = Mock(return_value=(True, []))
    from risk.order_manager import OrderQueueManager
    from risk.fee_and_spread_engine import SpreadFeeEngine
    from risk.risk_manager import FuturesRiskManager
    from config.settings import RiskConfig
    from strategy.memory_reasoning_engine import EpisodicTradeMemoryBank
    state = SimpleNamespace(
        active_exchange="binance", binance_api=api, symbol="BTCUSDT",
        live_price=78000.0, active_timeframe="15m",
        current_balance=100.0, initial_balance=100.0, total_fees=0,
        current_position=None, hedge_positions={}, trades=[], execution_blocker="",
        order_manager=OrderQueueManager(), trade_memory=EpisodicTradeMemoryBank(),
        last_decision_trace=None,
        fee_engine=SpreadFeeEngine(),
        risk_manager=FuturesRiskManager(RiskConfig(initial_balance=100.0)),
        exchange_wallet=100.0,
        freqtrade_protections=SimpleNamespace(
            max_drawdown_guard=SimpleNamespace(peak_balance=100.0),
            on_trade_closed=Mock(), export_state=Mock(return_value=None),
            restore_state=Mock()),
        jesse_engine=SimpleNamespace(record_trade=Mock(),
            compute_metrics=lambda: SimpleNamespace(current_consecutive_losses=0)),
        shadow_account=SimpleNamespace(
            analyze_trade_history=Mock(return_value=SimpleNamespace(
                total_trades=0, win_rate=0.0, profit_factor=0.0, expectancy=0.0))),
        storage=SimpleNamespace(save_execution_runtime=Mock(return_value=0),
                                load_execution_runtime=Mock(return_value=None)),
        build_market_snapshot=lambda: SimpleNamespace(
            freshness_issues=lambda: [], price=78000.0, environment="mainnet",
            context={"hft": SimpleNamespace(is_toxic_flow=False, depth_ready=True,
                                            vpin_ready=True, market_resilience_pct=100.0,
                                            liquidity_drought_warning=False)}),
    )
    state.fee_engine.update_book(77999.0, 78001.0)
    return state


class UserDataParserTests(unittest.TestCase):
    def test_parse_order_update(self):
        evt = parse_order_update({"e": "ORDER_TRADE_UPDATE", "o": {
            "s": "BTCUSDT", "i": 123, "c": "auto-1", "S": "BUY", "o": "LIMIT",
            "X": "FILLED", "z": "0.001", "q": "78.0", "ap": "78000.0"}})
        self.assertIsNotNone(evt)
        self.assertEqual(evt["order_id"], "123")
        self.assertEqual(evt["status"], "FILLED")
        self.assertAlmostEqual(evt["executed_qty"], 0.001)

    def test_parse_other_event_returns_none(self):
        self.assertIsNone(parse_order_update({"e": "ACCOUNT_UPDATE"}))
        self.assertIsNone(parse_account_update({"e": "ORDER_TRADE_UPDATE"}))

    def test_parse_account_update(self):
        evt = parse_account_update({"e": "ACCOUNT_UPDATE", "a": {
            "P": [{"s": "BTCUSDT", "pa": "0.001", "ep": "78000.0", "up": "-0.05"}],
            "B": [{"a": "USDT", "wb": "99.9"}]}})
        self.assertIsNotNone(evt)
        self.assertEqual(len(evt["positions"]), 1)
        self.assertAlmostEqual(evt["positions"][0]["unrealized_pnl"], -0.05)
        self.assertAlmostEqual(evt["balances"]["USDT"], 99.9)

    def test_account_update_skips_zero_positions(self):
        evt = parse_account_update({"e": "ACCOUNT_UPDATE", "a": {
            "P": [{"s": "BTCUSDT", "pa": "0", "ep": "0", "up": "0"}], "B": []}})
        self.assertEqual(evt["positions"], [])


class UserFillTests(unittest.TestCase):
    def test_fill_applies_immediately(self):
        state = make_state()
        life = ExecutionLifecycle(state)
        order, _ = state.order_manager.place_order(
            "BTCUSDT", "LIMIT", "BUY", 78000.0, 20.0, 5, quantity=0.001,
            stop_loss=77000.0, take_profit=79000.0, client_order_id="u-1",
            best_bid=77999.0, best_ask=78001.0)
        order.exchange_order_id = "999"
        order.exchange_status = "NEW"
        ok = life.apply_user_fill({"symbol": "BTCUSDT", "order_id": "999",
                                   "client_order_id": "u-1", "status": "FILLED",
                                   "executed_qty": 0.001, "cum_quote": 78.0, "avg_price": 78000.0})
        self.assertTrue(ok)
        self.assertIsNotNone(state.current_position)
        self.assertAlmostEqual(state.current_position["units"], 0.001)

    def test_unknown_fill_returns_false(self):
        state = make_state()
        life = ExecutionLifecycle(state)
        self.assertFalse(life.apply_user_fill({"symbol": "BTCUSDT", "order_id": "nope",
                                               "client_order_id": "nope", "status": "FILLED",
                                               "executed_qty": 0.001, "cum_quote": 78.0, "avg_price": 78000.0}))

    def test_account_update_syncs_upnl(self):
        state = make_state()
        life = ExecutionLifecycle(state)
        state.current_position = {"direction": 1, "units": 0.001, "entry_price": 78000.0,
                                  "margin": 15.6, "initial_margin": 15.6, "unrealized_pnl": -99.0}
        ok = life.apply_user_account({"positions": [{"symbol": "BTCUSDT", "amount": 0.001,
                                                     "entry_price": 78000.0, "unrealized_pnl": -0.05}],
                                      "balances": {"USDT": 99.9}})
        self.assertTrue(ok)
        self.assertAlmostEqual(state.current_position["unrealized_pnl"], -0.05)
        self.assertAlmostEqual(state.exchange_wallet, 99.9)


class ListenKeyTests(unittest.TestCase):
    def test_stream_url_hosts(self):
        main = SimpleNamespace(is_testnet=False)
        test = SimpleNamespace(is_testnet=True)
        from data.binance_api_manager import BinanceAPIManager
        self.assertIn("fstream.binance.com", BinanceAPIManager.user_data_stream_url(main, "KEY"))
        self.assertIn("stream.binancefuture.com", BinanceAPIManager.user_data_stream_url(test, "KEY"))

    def test_stream_start_requires_websockets_and_key(self):
        stream = BinanceUserDataStream(api_manager=Mock())
        stream.api.create_listen_key = Mock(return_value=(False, {"code": -1}))
        self.assertFalse(stream.start())
        self.assertFalse(stream.is_running)


if __name__ == "__main__":
    unittest.main()
