"""Cum A (chot phan bo von o tang khop cuoi) + Cum B (ly do vao/huy lenh).

Cum A: entry_guard tu choi lenh vuot tran phan bo du so du da duoc
pipeline duyet (so du troi giua chung), ke ca khi risk envelope van du.
Cum B: lenh cho co entry_reason; huy luon co cancel_reason; vi the va
lat lenh giu entry_reason khi khop.
"""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from risk.order_manager import OrderQueueManager
from trading.execution import ExecutionLifecycle


def make_exec_state(balance=10000.0):
    api = SimpleNamespace(is_live_enabled=False, is_testnet=True)
    api.normalize_order_values = Mock(side_effect=lambda symbol, qty, price=None, **kw: (
        (True, {"quantity": qty, "price": price}) if qty >= 0.001 else (False, {"code": -2})))
    api.prepare_testnet_trading = Mock(return_value=(True, {"maxNotionalValue": "1000000"}))
    api.test_connection = Mock(return_value={"success": True, "can_trade": True,
                                             "wallet_balance": balance, "available_balance": balance})
    api.place_order_live = Mock(return_value=(True, {"orderId": "1", "status": "NEW"}))
    api.query_order = Mock(return_value=(False, {"code": -2013}))
    api.cancel_order = Mock(return_value=(False, {"code": -2011}))
    api._send_request = Mock(return_value=(True, []))
    from risk.fee_and_spread_engine import SpreadFeeEngine
    from risk.risk_manager import FuturesRiskManager
    from config.settings import RiskConfig
    from risk.capital_allocator import PortfolioCapitalAllocator
    from risk.monthly_target_governor import MonthlyTargetGovernor
    from strategy.memory_reasoning_engine import EpisodicTradeMemoryBank
    from strategy.shadow_account import ShadowAccountAnalyzer
    from trading.pipeline import DecisionTrace
    from risk.freqtrade_protections import FreqtradeProtectionEngine
    from risk.jesse_expectancy_engine import JesseExpectancyEngine
    state = SimpleNamespace(
        active_exchange="binance", binance_api=api, symbol="BTCUSDT",
        live_price=60000.0, active_timeframe="15m",
        current_balance=balance, initial_balance=balance, total_fees=0,
        current_position=None, hedge_positions={}, trades=[], execution_blocker="",
        order_manager=OrderQueueManager(), trade_memory=EpisodicTradeMemoryBank(),
        last_decision_trace=DecisionTrace("entry-1", "snapshot"),
        fee_engine=SpreadFeeEngine(),
        risk_config=RiskConfig(initial_balance=balance),
        risk_manager=FuturesRiskManager(RiskConfig(initial_balance=balance)),
        capital_allocator=PortfolioCapitalAllocator(),
        monthly_governor=MonthlyTargetGovernor(storage=None),
        freqtrade_protections=FreqtradeProtectionEngine(initial_balance=balance),
        jesse_engine=JesseExpectancyEngine(),
        shadow_account=ShadowAccountAnalyzer(),
        storage=SimpleNamespace(save_execution_runtime=Mock(return_value=0),
                                load_execution_runtime=Mock(return_value=None)),
        build_market_snapshot=lambda: SimpleNamespace(
            freshness_issues=lambda: [], price=60000.0, environment="testnet",
            context={"hft": SimpleNamespace(is_toxic_flow=False, depth_ready=True,
                                            vpin_ready=True, market_resilience_pct=100.0,
                                            liquidity_drought_warning=False)}),
    )
    state.fee_engine.update_book(59999.0, 60001.0)
    return state


class AllocationGateTests(unittest.TestCase):
    def test_entry_guard_rejects_when_sleeve_dry(self):
        state = make_exec_state()
        life = ExecutionLifecycle(state)
        # Chiem gan het ngan SHORT_TERM bang lenh cho gia (margin 3900/4000)
        big, _ = state.order_manager.place_order(
            "BTCUSDT", "LIMIT", "BUY", 60000.0, 3900.0, 3, quantity=0.195,
            stop_loss=59000.0, take_profit=62000.0, client_order_id="fill-sleeve",
            timeframe="15m")
        order, _ = state.order_manager.place_order(
            "BTCUSDT", "LIMIT", "BUY", 60000.0, 500.0, 3, quantity=0.025,
            stop_loss=59000.0, take_profit=62000.0, client_order_id="over-sleeve",
            timeframe="15m")
        reason = life.entry_guard(order)
        self.assertTrue(reason)
        self.assertIn("SHORT_TERM", reason)

    def test_entry_guard_passes_within_sleeve(self):
        state = make_exec_state()
        life = ExecutionLifecycle(state)
        order, _ = state.order_manager.place_order(
            "BTCUSDT", "LIMIT", "BUY", 60000.0, 300.0, 3, quantity=0.015,
            stop_loss=59000.0, take_profit=62000.0, client_order_id="fit-sleeve",
            timeframe="15m")
        self.assertEqual(life.entry_guard(order), "")


class OrderReasonTests(unittest.TestCase):
    def test_place_order_stores_entry_reason(self):
        qm = OrderQueueManager()
        order, _ = qm.place_order(
            "BTCUSDT", "LIMIT", "BUY", 60000.0, 300.0, 3, quantity=0.015,
            stop_loss=59000.0, take_profit=62000.0, client_order_id="reason-1",
            entry_reason="LONG POST_ONLY che do TRENDING_BULL tin cay 80%")
        self.assertEqual(order.entry_reason, "LONG POST_ONLY che do TRENDING_BULL tin cay 80%")
        pend = qm.get_pending_orders()
        self.assertEqual(pend[0]["entry_reason"], order.entry_reason)

    def test_cancel_always_stores_reason(self):
        qm = OrderQueueManager()
        order, _ = qm.place_order(
            "BTCUSDT", "LIMIT", "BUY", 60000.0, 300.0, 3, quantity=0.015,
            stop_loss=59000.0, take_profit=62000.0, client_order_id="reason-2")
        self.assertTrue(qm.cancel_order(order.order_id, "Nguoi dung huy tay"))
        self.assertEqual(order.cancel_reason, "Nguoi dung huy tay")
        self.assertTrue(order.cancelled_at)

    def test_cancel_default_reason_when_missing(self):
        qm = OrderQueueManager()
        order, _ = qm.place_order(
            "BTCUSDT", "LIMIT", "BUY", 60000.0, 300.0, 3, quantity=0.015,
            stop_loss=59000.0, take_profit=62000.0, client_order_id="reason-3")
        self.assertTrue(qm.cancel_order(order.order_id))
        self.assertTrue(order.cancel_reason)

    def test_queue_propagates_entry_reason(self):
        state = make_exec_state()
        life = ExecutionLifecycle(state)
        from trading.pipeline import CandidateOrder
        cand = CandidateOrder(order_id="entry-1", symbol="BTCUSDT", direction=1,
                              order_type="POST_ONLY", entry_price=60000.0,
                              stop_loss=59000.0, take_profit=62000.0,
                              leverage=3, quantity=0.015, margin=300.0,
                              regime="TRENDING_BULL", confidence=80)
        state.last_decision_trace.entries.append(
            __import__("trading.pipeline", fromlist=["StageOutcome"]).StageOutcome.pass_("Approved POST_ONLY via OctoBot WEAK_BULLISH"))
        res = life.queue(cand)
        self.assertIn(res["status"], ("pending", "active", "filled"))
        placed = next(o for o in state.order_manager.orders if o.client_order_id == "entry-1")
        self.assertTrue(placed.entry_reason)

    def test_apply_entry_keeps_position_reason(self):
        state = make_exec_state()
        life = ExecutionLifecycle(state)
        order, _ = state.order_manager.place_order(
            "BTCUSDT", "LIMIT", "BUY", 60000.0, 300.0, 3, quantity=0.015,
            stop_loss=59000.0, take_profit=62000.0, client_order_id="reason-4",
            entry_reason="Ly do vao test")
        life.apply_entry(order, 0.015, 60000.0)
        pos = state.current_position
        self.assertEqual(pos["entry_reason"], "Ly do vao test")
        self.assertEqual(pos["orders"][0]["entry_reason"], "Ly do vao test")


if __name__ == "__main__":
    unittest.main()
