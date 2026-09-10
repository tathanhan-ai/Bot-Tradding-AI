"""Offline integration checks for durable execution; no network or exchange credentials."""
from copy import deepcopy
from dataclasses import dataclass
import json
import math
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from config.settings import RiskConfig
from risk.freqtrade_protections import FreqtradeProtectionEngine
from risk.order_manager import OrderQueueManager
from risk.risk_manager import FuturesRiskManager
from strategy.memory_reasoning_engine import EpisodicTradeMemoryBank
from strategy.shadow_account import ShadowAccountAnalyzer
from trading.execution import ExecutionLifecycle
from trading.pipeline import DecisionTrace


class MemoryStorage:
    def __init__(self):
        self.saved = None

    def save_execution_runtime(self, payload):
        self.saved = json.loads(json.dumps(payload, allow_nan=False))

    def load_execution_runtime(self):
        return deepcopy(self.saved)


@dataclass
class GovernorResult:
    size_multiplier: float = .5


def make_state(live=False, storage=None):
    api = SimpleNamespace(is_live_enabled=live, is_testnet=True)
    api.normalize_order_values = Mock(side_effect=lambda symbol, qty, price=None, **kwargs:
        (True, {"quantity": math.floor(qty * 1000 + 1e-9) / 1000, "price": round(price, 1) if price is not None else None}) if qty >= .001 else (False, {"code": -2}))
    api.prepare_testnet_trading = Mock(return_value=(True, {"maxNotionalValue": "1000000"}))
    api.test_connection = Mock(return_value={"success": True, "can_trade": True, "wallet_balance": 10000, "available_balance": 10000})
    counter = iter(range(100, 200))
    api.place_order_live = Mock(side_effect=lambda *args, **kwargs:
        (True, {"orderId": f"algo:{next(counter)}", "status": "NEW", "executedQty": "0", "avgPrice": "0"}))
    api.query_order = Mock(return_value=(False, {"code": -2013}))
    api.cancel_order = Mock(return_value=(False, {"code": -2011}))
    api._send_request = Mock(return_value=(True, []))
    state = SimpleNamespace(
        active_exchange="binance", binance_api=api, symbol="BTCUSDT", live_price=60000, active_timeframe="15m",
        current_balance=10000, initial_balance=10000, total_fees=0, current_position=None, trades=[], execution_blocker="",
        order_manager=OrderQueueManager(), storage=storage or MemoryStorage(), trade_memory=EpisodicTradeMemoryBank(),
        last_decision_trace=DecisionTrace("entry-1", "snapshot"),
        fee_engine=SimpleNamespace(bid_price=59999, ask_price=60000, calculate_fee=lambda value, is_maker=False: value * .0005,
                                  calculate_breakeven_price=lambda entry, direction: entry * (1 + direction * .001)),
        risk_config=SimpleNamespace(max_account_risk_pct=.05),
        risk_manager=SimpleNamespace(update_balance=Mock(), circuit_breaker_active=False,
            ai_cro=SimpleNamespace(last_verdict=None, record_trade_result=Mock()),
            calculate_liquidation_price=lambda entry, direction, leverage: entry * (1 - direction / leverage),
            evaluate_order=Mock(return_value=SimpleNamespace(approved=True, units=1, rejection_reason=""))),
        freqtrade_protections=SimpleNamespace(validate_new_trade=Mock(return_value=(True, "", 0)), on_trade_closed=Mock()),
        jesse_engine=SimpleNamespace(record_trade=Mock(), compute_metrics=lambda: SimpleNamespace(current_consecutive_losses=0)),
        shadow_account=ShadowAccountAnalyzer(), monthly_governor=SimpleNamespace(evaluate=Mock(return_value=GovernorResult())),
        build_market_snapshot=lambda: SimpleNamespace(freshness_issues=lambda: [], price=60000, environment="testnet",
            context={"hft": SimpleNamespace(is_toxic_flow=False, liquidity_drought_warning=False)}))
    return state


def entry(state, lifecycle):
    order, _ = state.order_manager.place_order("BTCUSDT", "LIMIT", "BUY", 60000, 200, 3, quantity=.01,
        stop_loss=59000, take_profit=62000, client_order_id="entry-1", candidate_payload={"metadata": {
            "entry_context": {"indicators": {"rsi": 45}, "of_data": {}, "smc_data": {}, "vwap_data": {}}}})
    lifecycle.apply_response(order, {"orderId": "1", "status": "FILLED", "executedQty": ".01", "avgPrice": "60000"})
    return order


class LiveExecutionIntegrationTest(unittest.TestCase):
    def setUp(self):
        quiet = patch("builtins.print")
        quiet.start()
        self.addCleanup(quiet.stop)

    def test_repeated_fill_restart_and_durable_realized_memory(self):
        state = make_state()
        lifecycle = ExecutionLifecycle(state)
        order = entry(state, lifecycle)
        balance = state.current_balance
        lifecycle.apply_response(order, {"orderId": "1", "status": "FILLED", "executedQty": ".01", "avgPrice": "60000"})
        self.assertEqual(state.current_balance, balance)
        self.assertEqual(len(state.current_position["orders"]), 1)
        lifecycle.realize(.004, 61000, "partial exit", "exit-receipt-1")
        after_close = state.current_balance
        self.assertEqual(state.monthly_governor.evaluate.call_count, 1)
        self.assertEqual(len(state.trade_memory.memory_records), 1)
        self.assertEqual(len(state.storage.saved["trades"]), 1)
        state2 = make_state(storage=state.storage)
        restored = ExecutionLifecycle(state2)
        restored.restore()
        restored.realize(.004, 61000, "duplicate after restart", "exit-receipt-1")
        self.assertEqual(state2.current_balance, after_close)
        self.assertAlmostEqual(state2.current_position["units"], .006)
        self.assertEqual(len(state2.trade_memory.memory_records), 1)
        self.assertIn("exit-receipt-1", restored.processed_execution_ids)
        restored.feedback()
        state2.monthly_governor.evaluate.assert_not_called()

    def test_paper_tick_receipt_restores_without_applied_quantity_mismatch(self):
        state = make_state()
        lifecycle = ExecutionLifecycle(state)
        order, _ = state.order_manager.place_order("BTCUSDT", "MARKET", "BUY", 60000, 100, 3, quantity=.005,
            stop_loss=59000, take_profit=62000, client_order_id="paper-1")
        lifecycle.tick()
        self.assertEqual(order.applied_quantity, order.exchange_executed_quantity)
        restored = ExecutionLifecycle(make_state(storage=state.storage))
        restored.restore()
        self.assertAlmostEqual(restored.state.current_position["units"], .005)

    def test_missing_protection_is_recreated_even_when_signature_is_unchanged(self):
        state = make_state(live=True)
        lifecycle = ExecutionLifecycle(state)
        entry(state, lifecycle)
        first_ids = state.current_position["protective_order_ids"][:]
        count = state.binance_api.place_order_live.call_count
        self.assertEqual(count, 2)
        lifecycle.ensure_protection()
        self.assertEqual(state.binance_api.place_order_live.call_count, count)
        lifecycle.protective[0]["status"] = "CANCELED"
        lifecycle.ensure_protection()
        self.assertEqual(state.binance_api.place_order_live.call_count, count + 2)
        self.assertNotEqual(state.current_position["protective_order_ids"], first_ids)
        self.assertTrue(lifecycle.protective[1]["cancel_requested"])

    def test_oneway_stop_uses_quantity_not_close_position(self):
        # Loi lenh 66-68: STOP dung closePosition -> san chi cho 1 lenh moi
        # huong (-4130), STOP cu chet khong dung luc la STOP moi rot theo ->
        # dap SL/TP that bai -> dong ca vi the oan. Nay STOP dung quantity +
        # reduceOnly, khong bao gio dung closePosition.
        state = make_state(live=True)
        lifecycle = ExecutionLifecycle(state)
        entry(state, lifecycle)
        sent = []

        def fake_place(symbol, side, order_type, quantity, **kw):
            sent.append({"order_type": order_type, "quantity": quantity, **kw})
            return True, {"orderId": f"p-{len(sent)}", "status": "NEW",
                          "executedQty": "0", "avgPrice": "0"}

        state.binance_api.place_order_live.side_effect = fake_place
        state.current_position["protective_order_ids"] = []
        state.current_position.pop("protected_signature", None)
        lifecycle.ensure_protection()
        stops = [c for c in sent if c["order_type"] == "STOP_MARKET"]
        self.assertTrue(stops)
        self.assertFalse(stops[0].get("close_position"))
        self.assertEqual(stops[0].get("quantity"), state.current_position["units"])
        self.assertTrue(stops[0].get("reduce_only"))

    def test_protection_revision_cancels_old_orders_on_exchange(self):
        # Dung lai bao ve phai huy lenh cu TREN SAN truoc khi dat moi,
        # neu khong STOP moi dinh -4130 chac chan.
        state = make_state(live=True)
        lifecycle = ExecutionLifecycle(state)
        entry(state, lifecycle)
        lifecycle.ensure_protection()
        old_stop = next(p for p in lifecycle.protective if p["order_type"] == "STOP_MARKET")
        old_stop["status"] = "NEW"
        old_stop["order_id"] = "algo:111"
        state.current_position.pop("protected_signature", None)
        state.binance_api.place_order_live.side_effect = lambda *a, **k: (
            True, {"orderId": "p-new", "status": "NEW", "executedQty": "0", "avgPrice": "0"})
        state.binance_api.cancel_order = __import__("unittest.mock", fromlist=["Mock"]).Mock(
            return_value=(True, {"status": "CANCELED"}))
        state.binance_api.query_order = __import__("unittest.mock", fromlist=["Mock"]).Mock(
            return_value=(False, {"code": -2013}))
        lifecycle.ensure_protection()
        self.assertTrue(old_stop.get("cancel_requested"))
        self.assertTrue(state.binance_api.cancel_order.called)

    def test_exit_receipts_reject_nonfinite_and_replayed_updates(self):
        state = make_state()
        lifecycle = ExecutionLifecycle(state)
        entry(state, lifecycle)
        intent = dict(client_order_id="exit-1", order_type="MARKET", side="SELL", quantity=.01, status="SUBMIT_UNKNOWN",
                      applied=0, quote_applied=0, reason="exit")
        baseline = state.current_balance
        for qty, avg in (("Infinity", "61000"), (".004", "Infinity"), (".004", "0"), ("0", "61000"), (".02", "61000")):
            self.assertFalse(lifecycle.apply_exit_response(intent, {"status": "FILLED", "executedQty": qty, "avgPrice": avg}))
            self.assertEqual(state.current_balance, baseline)
            self.assertEqual(intent["status"], "SUBMIT_UNKNOWN")
        response = {"orderId": "2", "status": "PARTIALLY_FILLED", "executedQty": ".004", "avgPrice": "61000"}
        self.assertTrue(lifecycle.apply_exit_response(intent, response))
        realized = state.current_balance
        self.assertTrue(lifecycle.apply_exit_response(intent, response))
        self.assertEqual(state.current_balance, realized)
        self.assertEqual(state.monthly_governor.evaluate.call_count, 1)

    def test_startup_unknown_inventory_blocks_new_orders_without_adoption(self):
        state = make_state(live=True)
        lifecycle = ExecutionLifecycle(state)
        state.binance_api._send_request.return_value = True, [{"symbol": "BTCUSDT", "positionAmt": ".01", "positionSide": "BOTH"}]
        self.assertFalse(lifecycle.verify_startup_inventory())
        self.assertIsNone(state.current_position)
        self.assertIn("differs", state.execution_blocker)
        state.binance_api._send_request.return_value = True, []
        self.assertTrue(lifecycle.verify_startup_inventory())
        self.assertEqual(state.execution_blocker, "")

    def test_conditional_trigger_is_normalized_and_definitive_rejection_is_terminal(self):
        state = make_state(live=True)
        lifecycle = ExecutionLifecycle(state)
        order, _ = state.order_manager.place_order("BTCUSDT", "CONDITIONAL", "BUY", 60000, 100, 3, quantity=.005,
            trigger_price=60010.123, stop_loss=59000, take_profit=62000, client_order_id="conditional-1")
        state.binance_api.place_order_live.return_value = False, {"code": -2010, "msg": "rejected"}
        state.binance_api.place_order_live.side_effect = None
        lifecycle.submit(order)
        self.assertEqual(order.status, "REJECTED")
        self.assertEqual(state.binance_api.place_order_live.call_args.kwargs["stop_price"], 60010.1)

    def test_partial_reservation_counts_remaining_and_scale_leverage_is_fixed(self):
        state = make_state()
        lifecycle = ExecutionLifecycle(state)
        first, _ = state.order_manager.place_order("BTCUSDT", "LIMIT", "BUY", 60000, 120, 3, quantity=.006,
            stop_loss=59000, take_profit=62000, client_order_id="partial-1")
        lifecycle.apply_response(first, {"orderId": "1", "status": "PARTIALLY_FILLED", "executedQty": ".004", "avgPrice": "60000"})
        next_order, _ = state.order_manager.place_order("BTCUSDT", "LIMIT", "BUY", 60000, 80, 3, quantity=.004,
            stop_loss=59000, take_profit=62000, client_order_id="next-1")
        state.risk_config.max_account_risk_pct = .00101
        self.assertEqual(lifecycle.entry_guard(next_order), "")
        next_order.leverage = 5
        self.assertIn("current position leverage", lifecycle.entry_guard(next_order))

    def test_oversized_exchange_receipt_is_rejected_before_position_apply(self):
        state = make_state()
        order, _ = state.order_manager.place_order(
            "BTCUSDT", "LIMIT", "BUY", 60000, 120, 3, quantity=.01,
            stop_loss=59000, take_profit=62000, client_order_id="oversized-receipt",
        )
        self.assertFalse(state.order_manager.record_exchange_update(order.order_id, {
            "orderId": "exchange-oversized", "status": "FILLED",
            "executedQty": ".02", "avgPrice": "60000",
        }))
        self.assertEqual(order.exchange_executed_quantity, 0.0)
        self.assertEqual(order.status, "PENDING")

    def test_unknown_entry_blocks_further_submission_but_reconciles(self):
        state = make_state(live=True)
        lifecycle = ExecutionLifecycle(state)
        unknown, _ = state.order_manager.place_order("BTCUSDT", "LIMIT", "BUY", 60000, 100, 3,
            stop_loss=59000, take_profit=62000, client_order_id="unknown-1")
        state.order_manager.mark_submit_pending(unknown.order_id)
        state.order_manager.mark_submit_unknown(unknown.order_id)
        state.order_manager.place_order("BTCUSDT", "LIMIT", "BUY", 60000, 100, 3,
            stop_loss=59000, take_profit=62000, client_order_id="next-1")
        lifecycle.tick()
        state.binance_api.query_order.assert_called_once()
        state.binance_api.place_order_live.assert_not_called()

    def test_notional_clamp_reserves_other_children(self):
        state = make_state(live=True)
        lifecycle = ExecutionLifecycle(state)
        first, _ = state.order_manager.place_order("BTCUSDT", "LIMIT", "BUY", 60000, 4000, 3, quantity=.2,
            stop_loss=59000, take_profit=62000, client_order_id="large-1")
        state.order_manager.place_order("BTCUSDT", "LIMIT", "BUY", 60000, 2000, 3, quantity=.1,
            stop_loss=59000, take_profit=62000, client_order_id="reserved-1")
        lifecycle.submit(first)
        self.assertAlmostEqual(state.binance_api.place_order_live.call_args.kwargs["quantity"], .075)

    def test_full_close_cancels_future_twap_children_across_restart(self):
        state = make_state()
        lifecycle = ExecutionLifecycle(state)
        first, _ = state.order_manager.place_order("BTCUSDT", "TWAP", "BUY", 60000, 180, 3, quantity=.009,
            twap_slices=3, twap_interval_seconds=60, stop_loss=59000, take_profit=62000, client_order_id="twap-close")
        lifecycle.tick()
        self.assertAlmostEqual(state.current_position["units"], .003)
        lifecycle.request_close(1.0, "manual close all")
        self.assertIsNone(state.current_position)
        self.assertTrue(all(order.status == "CANCELED" for order in state.order_manager.orders if order is not first))
        restored = ExecutionLifecycle(make_state(storage=state.storage))
        restored.restore()
        restored.tick()
        self.assertIsNone(restored.state.current_position)
        self.assertFalse(restored.state.order_manager.pending_orders)

    def test_full_protective_exit_cancels_unsent_scale_children(self):
        state = make_state()
        lifecycle = ExecutionLifecycle(state)
        entry(state, lifecycle)
        pending, _ = state.order_manager.place_order("BTCUSDT", "LIMIT", "BUY", 60000, 100, 3, quantity=.005,
            stop_loss=59000, take_profit=62000, client_order_id="pending-after-stop")
        intent = dict(client_order_id="stop-exit", order_type="STOP_MARKET", side="SELL", quantity=.01,
            status="NEW", applied=0, quote_applied=0, reason="Protective SL/TP")
        lifecycle.apply_exit_response(intent, {"orderId": "stop", "status": "FILLED", "executedQty": ".01", "avgPrice": "59000"})
        self.assertEqual(pending.status, "CANCELED")
        lifecycle.tick()
        self.assertIsNone(state.current_position)

    def test_unresolved_full_close_barrier_survives_restart_and_blocks_new_entry(self):
        state = make_state(live=True)
        lifecycle = ExecutionLifecycle(state)
        entry(state, lifecycle)
        unknown, _ = state.order_manager.place_order("BTCUSDT", "LIMIT", "BUY", 60000, 100, 3, quantity=.005,
            stop_loss=59000, take_profit=62000, client_order_id="unknown-before-close")
        state.order_manager.mark_submit_pending(unknown.order_id)
        state.order_manager.mark_submit_unknown(unknown.order_id)
        lifecycle.request_close(1.0, "manual close all")
        restored = ExecutionLifecycle(make_state(live=True, storage=state.storage))
        restored.restore()
        restored_unknown = next(item for item in restored.state.order_manager.pending_orders if item.order_id == unknown.order_id)
        self.assertEqual(restored_unknown.status, "CANCEL_REQUESTED")
        self.assertTrue(restored.entry_exit_barrier)
        self.assertIn("Full close", restored.entry_guard(restored_unknown))

    def test_late_entry_fill_during_full_close_is_reconciled_and_flattened(self):
        state = make_state(live=True)
        lifecycle = ExecutionLifecycle(state)
        entry(state, lifecycle)
        pending, _ = state.order_manager.place_order("BTCUSDT", "LIMIT", "BUY", 60000, 100, 3, quantity=.005,
            stop_loss=59000, take_profit=62000, client_order_id="cancel-race")
        state.order_manager.mark_exchange_ack(pending.order_id, "late-fill")
        lifecycle.request_close(1.0, "manual close all")
        first_exit = lifecycle.exits[-1]
        self.assertEqual(pending.status, "CANCEL_REQUESTED")
        lifecycle.apply_exit_response(first_exit, {"orderId": first_exit["order_id"], "status": "FILLED", "executedQty": ".01", "avgPrice": "60000"})
        state.binance_api.cancel_order.return_value = False, {"code": -2011}
        state.binance_api.query_order.return_value = True, {"orderId": "late-fill", "status": "FILLED", "executedQty": ".005", "avgPrice": "60000"}
        lifecycle.cancel(pending)
        self.assertAlmostEqual(state.current_position["units"], .005)
        self.assertTrue(lifecycle.entry_exit_barrier)
        lifecycle.reconcile_close_barrier()
        self.assertEqual(len(lifecycle.exits), 2)
        second_exit = lifecycle.exits[-1]
        self.assertAlmostEqual(second_exit["quantity"], .005)
        lifecycle.apply_exit_response(second_exit, {"orderId": second_exit["order_id"], "status": "FILLED", "executedQty": ".005", "avgPrice": "60000"})
        lifecycle.reconcile_close_barrier()
        self.assertIsNone(state.current_position)
        self.assertTrue(lifecycle.entry_exit_barrier)  # Old protection must not hit the next entry.
        for protective in lifecycle.protective:
            if protective["status"] == "NEW":
                lifecycle.apply_exit_response(protective, {"orderId": protective["order_id"], "status": "CANCELED", "executedQty": "0", "avgPrice": "0"})
        lifecycle.reconcile_close_barrier()
        self.assertEqual(lifecycle.entry_exit_barrier, "")
        self.assertTrue(state.binance_api.place_order_live.call_args.kwargs["reduce_only"])

    def test_tp1_fill_preserves_tp2_budget_and_restart_progress(self):
        state = make_state(live=True)
        lifecycle = ExecutionLifecycle(state)
        order, _ = state.order_manager.place_order("BTCUSDT", "LIMIT", "BUY", 60000, 200, 3, quantity=.01,
            stop_loss=59000, take_profit=63000, client_order_id="staged-entry", candidate_payload={
                "staged_take_profits": {"tp1": 61000, "tp1_ratio": .4, "tp2": 62000, "tp2_ratio": .3}})
        lifecycle.apply_response(order, {"orderId": "entry", "status": "FILLED", "executedQty": ".01", "avgPrice": "60000"})
        tp1 = next(intent for intent in lifecycle.protective if intent.get("tp_stage") == "tp1")
        lifecycle.apply_exit_response(tp1, {"orderId": tp1["order_id"], "status": "PARTIALLY_FILLED", "executedQty": ".002", "avgPrice": "61000"})
        self.assertAlmostEqual(lifecycle.staged_tp_remaining(state.current_position, "tp1"), .002)
        lifecycle.apply_exit_response(tp1, {"orderId": tp1["order_id"], "status": "FILLED", "executedQty": ".004", "avgPrice": "61000"})
        lifecycle.ensure_protection()
        self.assertAlmostEqual(state.current_position["units"], .006)
        self.assertEqual(lifecycle.staged_tp_remaining(state.current_position, "tp1"), 0)
        self.assertAlmostEqual(lifecycle.staged_tp_remaining(state.current_position, "tp2"), .003)
        active = [intent for intent in lifecycle.protective if intent["order_id"] in state.current_position["protective_order_ids"]]
        self.assertNotIn("tp1", [intent.get("tp_stage") for intent in active])
        tp2 = next(intent for intent in active if intent.get("tp_stage") == "tp2")
        self.assertAlmostEqual(tp2["quantity"], .003)
        restored = ExecutionLifecycle(make_state(live=True, storage=state.storage))
        restored.restore()
        self.assertAlmostEqual(restored.staged_tp_remaining(restored.state.current_position, "tp2"), .003)
        restored_tp2 = next(intent for intent in restored.protective if intent["order_id"] == tp2["order_id"])
        receipt = {"orderId": tp2["order_id"], "status": "FILLED", "executedQty": ".003", "avgPrice": "62000"}
        restored.apply_exit_response(restored_tp2, receipt)
        restored.apply_exit_response(restored_tp2, receipt)
        self.assertAlmostEqual(restored.state.current_position["units"], .003)
        self.assertEqual(restored.staged_tp_remaining(restored.state.current_position, "tp2"), 0)

    def test_hedge_staged_take_profits_use_quantity_not_close_position(self):
        state = make_state(live=True)
        state.hedge_positions = {
            "LONG": {
                "direction": 1, "units": .01, "stop_loss": 59000,
                "take_profit": 63000, "leverage": 3, "entry_price": 60000,
                "staged_take_profits": {
                    "tp1": 61000, "tp1_ratio": .4,
                    "tp2": 62000, "tp2_ratio": .3,
                },
                "tp_stage_initial_units": .01, "tp_stage_filled": {},
                "orders": [], "protection_revision": 0,
            }
        }
        lifecycle = ExecutionLifecycle(state)

        lifecycle._ensure_protection_for(state.hedge_positions["LONG"], "LONG")

        calls = state.binance_api.place_order_live.call_args_list
        self.assertEqual(len(calls), 4)  # stop, TP1, TP2, residual TP
        # Tat ca dung quantity (ke ca STOP): closePosition moi huong chi
        # duoc 1 lenh tren san, dung la STOP moi dinh -4130 khi lenh cu
        # chet khong dung luc (lenh live 66-68).
        for c in calls:
            self.assertFalse(c.kwargs["close_position"])
        self.assertAlmostEqual(calls[1].args[3], .004)
        self.assertAlmostEqual(calls[2].args[3], .003)

    def test_generic_partial_close_does_not_consume_staged_tp_budget_and_grid_tags_survive(self):
        state = make_state()
        lifecycle = ExecutionLifecycle(state)
        order = entry(state, lifecycle)
        pos = state.current_position
        pos["staged_take_profits"] = {"tp1": 61000, "tp1_ratio": .4, "tp2": 62000, "tp2_ratio": .3}
        pos["orders"][0].update(group_type="GRID", execution_group="grid-example")
        lifecycle.request_close(.2, "manual partial")
        self.assertAlmostEqual(lifecycle.staged_tp_remaining(state.current_position, "tp1"), .004)
        self.assertEqual(state.trades[-1]["group_type"], "GRID")
        self.assertEqual(state.trades[-1]["execution_group"], "grid-example")

    def test_restart_restores_daily_circuit_and_all_freqtrade_locks(self):
        state = make_state()
        state.risk_manager = FuturesRiskManager(RiskConfig(initial_balance=10000))
        state.freqtrade_protections = FreqtradeProtectionEngine(initial_balance=10000)
        state.current_balance = 9500
        state.risk_manager.update_balance(state.current_balance)
        state.risk_manager.ai_cro.evaluate_risk_profile(state.current_balance, "RANGING", 50, .02)
        state.freqtrade_protections.on_trade_closed({"pnl": -100, "reason": "STOP_LOSS"})
        state.freqtrade_protections.on_trade_closed({"pnl": -400, "reason": "STOP_LOSS"})
        state.freqtrade_protections.update_balance(11000, 11000)
        state.freqtrade_protections.update_balance(9500, 9500)
        risk_before = state.risk_manager.export_state()
        guards_before = state.freqtrade_protections.export_state()
        ExecutionLifecycle(state).persist()

        restored_state = make_state(storage=state.storage)
        # Constructor defaults must not overwrite the durable day's baseline.
        restored_state.risk_manager = FuturesRiskManager(RiskConfig(initial_balance=1000))
        restored_state.freqtrade_protections = FreqtradeProtectionEngine(initial_balance=1000)
        ExecutionLifecycle(restored_state).restore()
        self.assertEqual(restored_state.risk_manager.export_state(), risk_before)
        self.assertEqual(restored_state.freqtrade_protections.export_state(), guards_before)
        self.assertEqual(restored_state.risk_manager.current_balance, 9500)
        self.assertTrue(restored_state.risk_manager.ai_cro.circuit_breaker)
        self.assertFalse(restored_state.risk_manager.evaluate_order("BTCUSDT", 1, 60000, 59000, 62000, 3).approved)
        self.assertTrue(restored_state.freqtrade_protections.cooldown_guard.is_locked()[0])
        self.assertTrue(restored_state.freqtrade_protections.stoploss_guard.is_locked()[0])
        self.assertTrue(restored_state.freqtrade_protections.max_drawdown_guard.is_locked()[0])

    def test_restore_keeps_only_current_mode_trades(self):
        # Runtime cu tron paper+live: restore o live chi nhan lenh live,
        # header khong hien THANG 39.1% (64 lenh) cua paper nua.
        state = make_state(live=True)
        state.storage.mode = "live"
        lifecycle = ExecutionLifecycle(state)
        state.storage.save_execution_runtime({
            "version": 2, "balance": 99.0, "total_fees": 0.0,
            "position": None, "hedge_positions": {}, "queue": {"version": 1, "next_id": 1, "orders": []},
            "protective": [], "exits": [], "traces": {},
            "trades": [
                {"id": 1, "execution_id": "paper:1", "pnl": 5.0, "mode": "paper"},
                {"id": 65, "execution_id": "live:65", "pnl": -0.06, "mode": "live"},
            ],
            "processed_execution_ids": [], "feedback_pending": False,
            "entry_exit_barrier": "", "memory_records": [], "blocker": "",
        })
        lifecycle.restore()
        self.assertEqual([x["id"] for x in state.trades], [65])

    def test_stale_protective_blocker_clears_when_no_position(self):
        # Dot dap SL/TP that bai (vi du dot lenh 449) de lai blocker ket,
        # moi lenh moi veto vinh vien o Stage 1 du san da sach. Nay tu don
        # khi khong con vi the nao de bao ve.
        state = make_state(live=True)
        lifecycle = ExecutionLifecycle(state)
        state.execution_blocker = "Protective setup failed; reducing tracked Testnet exposure"
        state.current_position = None
        state.hedge_positions = {}
        lifecycle.ensure_protection()
        self.assertEqual(state.execution_blocker, "")

    def test_protective_blocker_kept_when_exits_pending(self):
        # Exits con do (chua FILLED/CANCELED/EXPIRED/REJECTED) thi
        # ensure_protection return som — blocker giu nguyen, khong don oan.
        state = make_state(live=True)
        lifecycle = ExecutionLifecycle(state)
        state.execution_blocker = "Protective setup failed; reducing tracked Testnet exposure"
        state.current_position = None
        state.hedge_positions = {}
        lifecycle.exits.append({"client_order_id": "exit-pending", "status": "NEW",
                                "applied": 0, "quote_applied": 0, "reason": "test"})
        lifecycle.ensure_protection()
        self.assertEqual(state.execution_blocker,
                         "Protective setup failed; reducing tracked Testnet exposure")

    def test_restart_new_utc_day_resets_daily_baseline_to_durable_balance(self):
        state = make_state()
        state.risk_manager = FuturesRiskManager(RiskConfig(initial_balance=10000))
        state.current_balance = 9500
        state.risk_manager.update_balance(state.current_balance)
        state.risk_manager.ai_cro.circuit_breaker = True
        state.risk_manager.risk_day = "2000-01-01"
        ExecutionLifecycle(state).persist()
        restored_state = make_state(storage=state.storage)
        restored_state.risk_manager = FuturesRiskManager(RiskConfig(initial_balance=1000))
        ExecutionLifecycle(restored_state).restore()
        self.assertEqual(restored_state.risk_manager.daily_start_balance, 9500)
        self.assertEqual(restored_state.risk_manager.ai_cro.daily_start_balance, 9500)
        self.assertFalse(restored_state.risk_manager.circuit_breaker_active)
        self.assertFalse(restored_state.risk_manager.ai_cro.circuit_breaker)


if __name__ == "__main__":
    unittest.main()
