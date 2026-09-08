import json
import unittest
import urllib.parse
from unittest.mock import Mock, patch

from data.binance_api_manager import BinanceAPIManager


class BinanceExecutionTest(unittest.TestCase):
    def manager(self):
        return BinanceAPIManager(api_key="key", api_secret="secret", is_testnet=True, is_live_enabled=True)

    def test_market_order_requests_result_and_preserves_reduce_only(self):
        manager = BinanceAPIManager(api_key="key", api_secret="secret", is_testnet=True, is_live_enabled=True)
        captured = {}

        def send(method, endpoint, params, signed=True):
            captured.update(method=method, endpoint=endpoint, params=params, signed=signed)
            return True, {"orderId": 123, "status": "FILLED", "executedQty": "0.01"}

        manager._send_request = send
        ok, response = manager.place_order_live(
            symbol="BTCUSDT", side="SELL", order_type="MARKET", quantity=0.01,
            reduce_only=True, client_order_id="pipe-order-1",
        )

        self.assertTrue(ok)
        self.assertEqual(response["status"], "FILLED")
        self.assertEqual(captured["endpoint"], "/fapi/v1/order")
        self.assertEqual(captured["params"]["newOrderRespType"], "RESULT")
        self.assertEqual(captured["params"]["reduceOnly"], "true")
        self.assertEqual(captured["params"]["newClientOrderId"], "pipe-order-1")

    def test_post_only_maps_to_gtx_limit_and_invalid_order_is_fail_closed(self):
        manager = BinanceAPIManager(api_key="key", api_secret="secret", is_testnet=True, is_live_enabled=True)
        captured = {}
        manager._send_request = lambda method, endpoint, params, signed=True: (captured.update(params=params) or True, {"status": "NEW"})

        ok, _ = manager.place_order_live("BTCUSDT", "BUY", "POST_ONLY", 0.01, price=60_000)
        self.assertTrue(ok)
        self.assertEqual(captured["params"]["type"], "LIMIT")
        self.assertEqual(captured["params"]["timeInForce"], "GTX")

        ok, error = manager.place_order_live("BTCUSDT", "BUY", "TRAILING_STOP_MARKET", 0.01)
        self.assertFalse(ok)
        self.assertIn("callbackRate", error["msg"])

    def test_only_filled_exchange_responses_can_create_local_fill(self):
        self.assertTrue(BinanceAPIManager.is_filled_response({"status": "FILLED", "executedQty": "0.01"}))
        self.assertFalse(BinanceAPIManager.is_filled_response({"status": "NEW", "executedQty": "0"}))
        self.assertFalse(BinanceAPIManager.is_filled_response({"status": "PARTIALLY_FILLED", "executedQty": "0.01"}))
        for invalid in ("NaN", "Infinity", None, -1, True):
            self.assertFalse(BinanceAPIManager.is_filled_response({"status": "FILLED", "executedQty": invalid}))

    def test_conditional_types_use_algo_api_and_ack_is_not_fill(self):
        manager = self.manager()
        for order_type in ("CONDITIONAL", "STOP", "STOP_MARKET", "TAKE_PROFIT", "TAKE_PROFIT_MARKET"):
            with self.subTest(order_type=order_type):
                manager._send_request = Mock(return_value=(True, {"algoId": 10, "algoStatus": "NEW", "clientAlgoId": "protection-1"}))
                ok, response = manager.place_order_live("BTCUSDT", "SELL", order_type, 0.01, price=59_000,
                                                      stop_price=59_100, reduce_only=True, client_order_id="protection-1")
                self.assertTrue(ok)
                self.assertFalse(manager.is_filled_response(response))
                self.assertEqual(response["orderId"], "algo:10")
                args = manager._send_request.call_args.args
                self.assertEqual(args[:2], ("POST", "/fapi/v1/algoOrder"))
                params = args[2]
                self.assertEqual(params["algoType"], "CONDITIONAL")
                self.assertEqual(params["triggerPrice"], "59100")
                self.assertEqual(params["clientAlgoId"], "protection-1")
                self.assertEqual(params["reduceOnly"], "true")
                self.assertNotIn("stopPrice", params)
                self.assertNotIn("newClientOrderId", params)

    def test_trailing_activate_price_and_callback_rate(self):
        manager = self.manager()
        manager._send_request = Mock(return_value=(True, {"algoId": 12, "algoStatus": "NEW"}))
        ok, _ = manager.place_order_live("BTCUSDT", "SELL", "TRAILING_STOP", 0.01, activation_price=62_000,
                                        callback_rate=10, reduce_only=True)
        self.assertTrue(ok)
        params = manager._send_request.call_args.args[2]
        self.assertEqual(params["activatePrice"], "62000")
        self.assertNotIn("activationPrice", params)
        self.assertEqual(params["callbackRate"], "10")

    def test_query_algo_follows_actual_order_and_preserves_stable_id(self):
        manager = self.manager()
        algo = {"algoId": 12, "algoStatus": "FINISHED", "actualOrderId": "456", "actualQty": "0.01"}
        for status in ("NEW", "PARTIALLY_FILLED", "FILLED"):
            with self.subTest(status=status):
                manager._send_request = Mock(side_effect=[(True, algo), (True, {"orderId": 456, "status": status,
                    "executedQty": "0.01", "avgPrice": "61000"})])
                ok, response = manager.query_order("BTCUSDT", order_id="algo:12")
                self.assertTrue(ok)
                self.assertEqual(response["orderId"], "algo:12")
                self.assertEqual(response["actualOrderId"], "456")
                self.assertEqual(manager.is_filled_response(response), status == "FILLED")
                self.assertEqual(manager._send_request.call_args.args,
                                 ("GET", "/fapi/v1/order", {"symbol": "BTCUSDT", "orderId": "456"}))
        manager._send_request = Mock(side_effect=[(True, algo), (False, {"code": -2013})])
        ok, response = manager.query_order("BTCUSDT", order_id="algo:12")
        self.assertFalse(ok)
        self.assertFalse(manager.is_filled_response(response))
        manager._send_request = Mock(return_value=(True, {"algoId": 12, "algoStatus": "FINISHED", "actualQty": "0.01"}))
        ok, response = manager.query_order("BTCUSDT", client_order_id="algo:protection-1")
        self.assertTrue(ok)
        self.assertFalse(manager.is_filled_response(response))
        self.assertEqual(manager._send_request.call_args.args[2], {"clientAlgoId": "protection-1"})

    def test_cancel_algo_and_trigger_race(self):
        manager = self.manager()
        manager._send_request = Mock(return_value=(True, {"algoId": 12, "code": "200", "msg": "success"}))
        ok, response = manager.cancel_order("BTCUSDT", order_id="algo:12")
        self.assertTrue(ok)
        self.assertEqual(response["status"], "CANCELED")
        self.assertEqual(manager._send_request.call_args.args, ("DELETE", "/fapi/v1/algoOrder", {"algoId": "12"}))
        manager._send_request = Mock(side_effect=[(False, {"code": -2011}),
            (True, {"algoId": 12, "actualOrderId": "456"}),
            (True, {"orderId": 456, "status": "CANCELED", "executedQty": "0.001"})])
        ok, response = manager.cancel_order("BTCUSDT", order_id="algo:12")
        self.assertTrue(ok)
        self.assertEqual(response["actualOrderId"], "456")
        self.assertEqual(response["executedQty"], "0.001")
        self.assertEqual(manager._send_request.call_args.args[1], "/fapi/v1/order")

    def test_non_finite_order_parameters_never_reach_transport(self):
        manager = self.manager()
        manager._send_request = Mock()
        for invalid in (float("nan"), float("inf"), -1, 0, None, True):
            with self.subTest(invalid=invalid):
                self.assertFalse(manager.place_order_live("BTCUSDT", "BUY", "MARKET", invalid)[0])
        for name in ("price", "stop_price", "activation_price", "callback_rate"):
            self.assertFalse(manager.place_order_live("BTCUSDT", "BUY", "MARKET", .01, **{name: float("nan")})[0])
        manager._send_request.assert_not_called()

    def test_all_writes_are_testnet_only_even_direct_transport(self):
        manager = self.manager()
        with patch("urllib.request.urlopen") as transport:
            # Mainnet chua xac nhan -> chan; live tat -> chan (giong cu)
            for testnet, enabled in ((False, True), (True, False)):
                manager.is_testnet, manager.is_live_enabled = testnet, enabled
                manager.mainnet_confirmed = False
                self.assertFalse(manager.place_order_live("BTCUSDT", "BUY", "MARKET", .01)[0])
                self.assertFalse(manager.cancel_order("BTCUSDT", order_id="algo:12")[0])
                self.assertFalse(manager.prepare_testnet_trading("BTCUSDT", 2)[0])
                self.assertFalse(manager._send_request("POST", "/fapi/v1/leverage", {"leverage": 2})[0])
            transport.assert_not_called()

    def test_mainnet_writes_require_explicit_confirmation(self):
        manager = self.manager()
        manager.is_testnet, manager.is_live_enabled = False, True
        manager.mainnet_confirmed = False
        with patch("urllib.request.urlopen") as transport:
            self.assertFalse(manager.place_order_live("BTCUSDT", "BUY", "MARKET", .01)[0])
            self.assertFalse(manager.cancel_order("BTCUSDT", order_id="algo:12")[0])
            transport.assert_not_called()
        # Sau xac nhan: cho phep di qua gate (transport duoc goi, ket qua tuy
        # mock; quan trong la khong con bi chan -997).
        manager.mainnet_confirmed = True
        manager._send_request = Mock(return_value=(True, {"orderId": 1, "status": "NEW", "executedQty": "0"}))
        ok, _ = manager.cancel_order("BTCUSDT", order_id="123")
        self.assertTrue(ok)

    def test_one_way_and_leverage_are_prepared_only_on_testnet(self):
        manager = self.manager()
        manager._send_request = Mock(side_effect=[(True, {"dualSidePosition": True}), (True, {"code": 200}), (True, {"code": 200}),
                                                 (True, {"leverage": 3, "maxNotionalValue": "1000000"})])
        self.assertTrue(manager.prepare_testnet_trading("BTCUSDT", 3)[0])
        self.assertEqual(manager._send_request.call_args_list[1].args,
                         ("POST", "/fapi/v1/positionSide/dual", {"dualSidePosition": "false"}))
        self.assertEqual(manager._send_request.call_args.args,
                         ("POST", "/fapi/v1/leverage", {"symbol": "BTCUSDT", "leverage": 3}))
        self.assertEqual(manager._send_request.call_args_list[2].args,
                         ("POST", "/fapi/v1/marginType", {"symbol": "BTCUSDT", "marginType": "ISOLATED"}))
        manager._send_request.reset_mock()
        for invalid in (float("nan"), 1.5, 126, True):
            self.assertFalse(manager.prepare_testnet_trading("BTCUSDT", invalid)[0])
        manager._send_request.assert_not_called()

    def test_hedge_mode_uses_position_side_and_native_close_position(self):
        manager = self.manager()
        manager._send_request = Mock(side_effect=[(True, {"dualSidePosition": False}), (True, {"code": 200}),
                                                   (True, {"code": 200}), (True, {"leverage": 3})])
        self.assertTrue(manager.prepare_testnet_trading("BTCUSDT", 3, hedge=True)[0])
        self.assertEqual(manager._send_request.call_args_list[1].args,
                         ("POST", "/fapi/v1/positionSide/dual", {"dualSidePosition": "true"}))
        captured = {}
        manager._send_request = lambda method, endpoint, params, signed=True: (captured.update(params=params) or True,
                                                                                {"algoId": 19, "algoStatus": "NEW"})
        ok, _ = manager.place_order_live("BTCUSDT", "SELL", "STOP_MARKET", .01, stop_price=59_000,
                                         position_side="LONG", close_position=True, client_order_id="grid-stop-1")
        self.assertTrue(ok)
        self.assertEqual(captured["params"]["positionSide"], "LONG")
        self.assertEqual(captured["params"]["closePosition"], "true")
        self.assertNotIn("reduceOnly", captured["params"])
        self.assertNotIn("quantity", captured["params"])

    def test_exchange_filters_floor_risk_and_do_not_round_up_min_notional(self):
        manager = self.manager()
        manager._send_request = Mock(return_value=(True, {"symbols": [{"symbol": "BTCUSDT", "status": "TRADING", "filters": [
            {"filterType": "LOT_SIZE", "minQty": "0.001", "maxQty": "1000", "stepSize": "0.001"},
            {"filterType": "MARKET_LOT_SIZE", "minQty": "0.002", "maxQty": "100", "stepSize": "0.002"},
            {"filterType": "PRICE_FILTER", "minPrice": "0.10", "maxPrice": "1000000", "tickSize": "0.10"},
            {"filterType": "MIN_NOTIONAL", "notional": "100"}]}]}))
        ok, result = manager.normalize_order_values("BTCUSDT", .0129, 60000.19)
        self.assertTrue(ok)
        self.assertEqual((result["quantity"], result["price"]), (.012, 60000.1))
        ok, result = manager.normalize_order_values("BTCUSDT", .0129, 60000.11, price_rounding="up")
        self.assertEqual(result["price"], 60000.2)
        ok, result = manager.normalize_order_values("BTCUSDT", .0119, order_type="MARKET", reference_price=60000)
        self.assertTrue(ok)
        self.assertEqual(result["quantity"], .010)
        self.assertFalse(manager.normalize_order_values("BTCUSDT", .001, 60000)[0])
        self.assertTrue(manager.normalize_order_values("BTCUSDT", .001, 60000, reduce_only=True)[0])
        self.assertFalse(manager.normalize_order_values("BTCUSDT", .0009, 60000)[0])
        self.assertEqual(manager._send_request.call_count, 1)

    def test_signed_transport_uses_server_clock_offset_without_mutating_params(self):
        manager = self.manager()
        clock_response = Mock()
        clock_response.__enter__ = Mock(return_value=clock_response)
        clock_response.__exit__ = Mock(return_value=False)
        clock_response.read.return_value = json.dumps({"serverTime": 200000}).encode()
        account_response = Mock()
        account_response.__enter__ = Mock(return_value=account_response)
        account_response.__exit__ = Mock(return_value=False)
        account_response.read.return_value = b'{"canTrade": true}'
        params = {"symbol": "BTCUSDT"}
        with patch("urllib.request.urlopen", side_effect=[clock_response, account_response]) as transport, \
             patch("data.binance_api_manager.time.time", return_value=100):
            self.assertTrue(manager._send_request("GET", "/fapi/v2/account", params)[0])
            query = urllib.parse.parse_qs(urllib.parse.urlsplit(transport.call_args.args[0].full_url).query)
        self.assertEqual(query["timestamp"], ["200000"])
        self.assertEqual(query["recvWindow"], ["5000"])
        self.assertEqual(params, {"symbol": "BTCUSDT"})


if __name__ == "__main__":
    unittest.main()
