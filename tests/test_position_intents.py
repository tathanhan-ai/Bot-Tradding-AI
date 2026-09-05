"""Position analysis proposes intents; only execution acknowledges progress."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from strategy.ai_position_manager import AIPositionCoordinator
from trading.replay import ReplayStorage, _state_class


def position(direction=1):
    return dict(direction=direction, entry_price=100, breakeven_price=100.1 if direction == 1 else 99.9,
                stop_loss=90 if direction == 1 else 110, take_profit=150 if direction == 1 else 50,
                units=.01, initial_risk=10, margin=1, partial_tp_done=False,
                is_risk_free=False, tp_expanded=False)


class PositionIntentTests(unittest.TestCase):
    def setUp(self):
        self.coordinator = AIPositionCoordinator()
        self.coordinator.roi_engine.evaluate_roi_exit = lambda *args: (False, "", 0, 0)

    def evaluate(self, pos, price, atr=1):
        return self.coordinator.evaluate_position(pos, price, {"atr": atr, "rsi": 60}, None, None)

    def test_breakeven_is_retryable_and_does_not_mutate_position(self):
        for direction in (1, -1):
            pos = position(direction)
            before = deepcopy(pos)
            first = self.evaluate(pos, 100 + direction * 11)
            self.assertEqual(first.action, "LOCK_BREAKEVEN")
            self.assertEqual(pos, before)
            self.assertEqual(self.evaluate(pos, 100 + direction * 11).action, first.action)

    def test_partial_close_remains_retryable_before_fill(self):
        pos = position()
        pos["is_risk_free"] = True
        before = deepcopy(pos)
        self.assertEqual(self.evaluate(pos, 112).action, "PARTIAL_TAKE_PROFIT")
        self.assertEqual(pos, before)
        self.assertEqual(self.evaluate(pos, 112).action, "PARTIAL_TAKE_PROFIT")

    def test_staged_budget_disables_independent_fifty_percent_exit(self):
        for direction in (1, -1):
            pos = position(direction)
            pos["is_risk_free"] = True
            pos["staged_take_profits"] = {"tp1": 100 + direction * 15, "tp1_ratio": .4,
                                         "tp2": 100 + direction * 30, "tp2_ratio": .35}
            before = deepcopy(pos)
            self.assertNotEqual(self.evaluate(pos, 100 + direction * 12).action, "PARTIAL_TAKE_PROFIT")
            self.assertEqual(pos, before)

    def test_losing_position_is_not_mislabeled_as_partial_profit(self):
        for direction in (1, -1):
            pos = position(direction)
            pos.update(stop_loss=100-direction*30, take_profit=100+direction*40, initial_risk=30)
            self.assertNotEqual(self.evaluate(pos, 100-direction*24).action, "PARTIAL_TAKE_PROFIT")

    def test_short_tp_expansion_uses_signed_progress(self):
        pos = position(-1)
        pos.update(stop_loss=60, is_risk_free=True, partial_tp_done=True)
        decision = self.coordinator.evaluate_position(pos, 55, {"atr":10,"rsi":35}, None, None)
        self.assertEqual(decision.action, "EXPAND_TAKE_PROFIT")
        self.assertLess(decision.new_take_profit, pos["take_profit"])
        self.assertLessEqual(decision.new_stop_loss, pos["stop_loss"])

    def test_empty_disabled_stages_do_not_disable_fallback_partial_exit(self):
        pos = position()
        pos.update(is_risk_free=True, staged_take_profits={"tp1": 115, "tp1_ratio": 0, "tp2_ratio": 0})
        self.assertEqual(self.evaluate(pos, 112).action, "PARTIAL_TAKE_PROFIT")

    def test_expansion_remains_retryable_before_protective_ack(self):
        pos = position()
        pos.update(is_risk_free=True, partial_tp_done=True, stop_loss=110)
        before = deepcopy(pos)
        self.assertEqual(self.evaluate(pos, 145, atr=100).action, "EXPAND_TAKE_PROFIT")
        self.assertEqual(pos, before)
        self.assertEqual(self.evaluate(pos, 145, atr=100).action, "EXPAND_TAKE_PROFIT")

    def state_with_entry(self):
        state = _state_class()(storage=ReplayStorage())
        order = SimpleNamespace(direction=1, order_type="MARKET", side="BUY", candidate_payload={},
                                client_order_id="intent-test", order_id="intent-test", applied_quantity=0,
                                exchange_order_id="paper-test", parent_intent_id="intent-test",
                                execution_group="intent-test", group_type="", timeframe="15m", leverage=3,
                                stop_loss=90, take_profit=150)
        state.execution.apply_entry(order, .01, 100)
        state.live_price = 112
        return state

    def test_only_confirmed_fill_marks_partial_complete(self):
        state = self.state_with_entry()
        pos = state.current_position
        pos["is_risk_free"] = True
        self.assertEqual(self.evaluate(pos, 112).action, "PARTIAL_TAKE_PROFIT")
        self.assertFalse(pos["partial_tp_done"])
        state.execution.request_close(.5, "TEST_ONLY_PARTIAL")
        self.assertTrue(pos["partial_tp_done"])
        self.assertAlmostEqual(pos["units"], .005)

    def test_paper_protection_marks_only_acknowledged_risk_free_and_expansion(self):
        state = self.state_with_entry()
        pos = state.current_position
        self.assertTrue(state.execution.replace_protection(sl=99)[0])
        self.assertFalse(pos.get("is_risk_free", False))
        self.assertTrue(state.execution.replace_protection(sl=pos["breakeven_price"], tp=160)[0])
        self.assertTrue(pos["is_risk_free"])
        self.assertTrue(pos["tp_expanded"])
        self.assertTrue(state.storage.load_execution_runtime()["position"]["tp_expanded"])

    def test_live_unconfirmed_protection_does_not_mark_completion(self):
        state = self.state_with_entry()
        pos = state.current_position
        state.binance_api.is_live_enabled = True
        before = deepcopy(pos)
        with patch.object(state.execution, "ensure_protection"):
            ok, _ = state.execution.replace_protection(sl=101, tp=160)
        self.assertFalse(ok)
        self.assertEqual(pos, before)

    def test_live_confirmed_protection_marks_completion(self):
        state = self.state_with_entry()
        pos = state.current_position
        state.binance_api.is_live_enabled = True

        def acknowledge():
            pos["protected_signature"] = [pos["units"], pos["stop_loss"], pos["take_profit"]]

        with patch.object(state.execution, "ensure_protection", side_effect=acknowledge):
            self.assertTrue(state.execution.replace_protection(sl=101, tp=160)[0])
        self.assertTrue(pos["is_risk_free"])
        self.assertTrue(pos["tp_expanded"])

    def test_staged_execution_keeps_the_original_inventory_budget(self):
        state = self.state_with_entry()
        pos = state.current_position
        pos.update(is_risk_free=True, staged_take_profits={"tp1": 115, "tp1_ratio": .4,
                                                         "tp2": 130, "tp2_ratio": .35})
        state.ai_coordinator = self.coordinator
        state.update_indicators = lambda: None
        state.indicators = {"atr": 1, "rsi": 60}
        state.process_execution_tick(112)
        self.assertEqual(pos["units"], .01)
        state.on_tick(115)
        state.process_execution_tick(115)
        self.assertAlmostEqual(pos["units"], .006)
        self.assertAlmostEqual(pos["tp_stage_filled"]["tp1"], .004)

    def test_paper_partial_close_applies_persisted_stop_followup(self):
        state = self.state_with_entry()
        self.assertTrue(state.execution.request_close(.5, "TEST_ONLY", after_stop_loss=101))
        self.assertAlmostEqual(state.current_position["units"], .005)
        self.assertEqual(state.current_position["stop_loss"], 101)
        self.assertTrue(state.current_position["is_risk_free"])
        saved = state.storage.load_execution_runtime()
        self.assertEqual(saved["exits"][-1]["after_stop_loss"], 101)
        self.assertTrue(saved["exits"][-1]["after_stop_applied"])

    def test_unknown_partial_close_restarts_then_applies_stop_only_after_fill(self):
        state = self.state_with_entry()
        state.binance_api.is_live_enabled = True
        with patch.object(state.binance_api, "normalize_order_values", return_value=(True, {"quantity": .005})), \
                patch.object(state.binance_api, "place_order_live", return_value=(False, {"code": -999})):
            self.assertFalse(state.execution.request_close(.5, "TEST_ONLY", after_stop_loss=101))
        self.assertFalse(state.current_position["partial_tp_done"])
        self.assertFalse(state.current_position.get("is_risk_free", False))
        self.assertEqual(state.current_position["stop_loss"], 90)
        restored = _state_class()(storage=state.storage)
        restored.live_price = 112
        restored.binance_api.is_live_enabled = True
        intent = restored.execution.exits[-1]
        self.assertEqual(intent["status"], "SUBMIT_UNKNOWN")
        self.assertEqual(intent["after_stop_loss"], 101)

        def acknowledge():
            pos = restored.current_position
            pos["protected_signature"] = [pos["units"], pos["stop_loss"], pos["take_profit"]]

        receipt = {"orderId": "test-only", "status": "FILLED", "executedQty": .005, "avgPrice": 112}
        with patch.object(restored.execution, "ensure_protection", side_effect=acknowledge) as protective:
            self.assertTrue(restored.execution.apply_exit_response(intent, receipt))
            self.assertTrue(restored.execution.apply_exit_response(intent, receipt))
            protective.assert_called_once()
        self.assertEqual(len(restored.trades), 1)
        self.assertAlmostEqual(restored.current_position["units"], .005)
        self.assertTrue(restored.current_position["partial_tp_done"])
        self.assertTrue(restored.current_position["is_risk_free"])
        self.assertEqual(restored.current_position["stop_loss"], 101)
        self.assertTrue(restored.storage.load_execution_runtime()["exits"][-1]["after_stop_applied"])


if __name__ == "__main__":
    unittest.main()
