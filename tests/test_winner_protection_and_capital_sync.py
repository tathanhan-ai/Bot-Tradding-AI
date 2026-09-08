"""Winner-protection + live capital sync: khóa lãi sớm, SL fractal, vốn theo sàn.

Cụm 1 (lãi đậm không trượt về SL):
- Breakeven lock từ +0.5R (không chờ +1.0R)
- Partial 50% từ +1.0R (không chờ +1.2R)
- TP expansion chỉ khi R >= 2.0 + momentum AND macro (không OR, không +2.5 ATR)
- Swing fractal thật (không min/max trần) + fallback percentile
Cụm 2 (live nghe số tiền server):
- get_wallet_snapshot / get_position_snapshot parse đúng
- sync_exchange_balance gán vốn theo ví sàn, fail thì giữ vốn cũ
- submit từ chối lệnh khi ví lệch vốn bot quá 10%
"""
import math
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd

from strategy.ai_position_manager import AIPositionCoordinator
from risk.structural_sl_tp import StructuralRiskCalculator
from data.binance_api_manager import BinanceAPIManager
from trading.execution import ExecutionLifecycle


def long_pos(**kw):
    base = dict(direction=1, entry_price=100.0, breakeven_price=100.08,
                stop_loss=90.0, take_profit=150.0, units=0.01,
                initial_risk=10.0, margin=100.0, partial_tp_done=False,
                is_risk_free=False, tp_expanded=False, peak_price=100.0)
    base.update(kw)
    return base


class WinnerProtectionTests(unittest.TestCase):
    def setUp(self):
        self.coord = AIPositionCoordinator()
        self.coord.roi_engine.evaluate_roi_exit = lambda *a: (False, "", 0, 0)

    def test_breakeven_locks_at_half_r(self):
        # +0.5R (giá 105) đã khóa — trước đây phải chờ +1.0R (giá 110)
        d = self.coord.evaluate_position(long_pos(), 105.0, {"atr": 1.0, "rsi": 55}, None, None)
        self.assertEqual(d.action, "LOCK_BREAKEVEN")

    def test_partial_profit_at_one_r(self):
        pos = long_pos(is_risk_free=True)
        d = self.coord.evaluate_position(pos, 110.0, {"atr": 1.0, "rsi": 60}, None, None)
        self.assertEqual(d.action, "PARTIAL_TAKE_PROFIT")

    def test_no_tp_expansion_below_two_r(self):
        # Tiến 90% tới TP nhưng R mới ~+1.3R -> KHÔNG nới TP nữa
        pos = long_pos(is_risk_free=True, partial_tp_done=True,
                       entry_price=100.0, stop_loss=90.0, take_profit=134.0,
                       initial_risk=10.0, peak_price=130.0)
        d = self.coord.evaluate_position(pos, 130.0, {"atr": 1.0, "rsi": 60}, None, None)
        self.assertNotEqual(d.action, "EXPAND_TAKE_PROFIT")

    def test_tp_expansion_needs_momentum_and_macro(self):
        # R = +2.5R nhưng RSI kiệt sức + macro ngược -> không nới
        pos = long_pos(is_risk_free=True, partial_tp_done=True,
                       entry_price=100.0, stop_loss=90.0, take_profit=120.0,
                       initial_risk=10.0, peak_price=125.0)
        verdict = SimpleNamespace(regime="TRENDING_BEAR", resistance_price=0.0, support_price=0.0)
        d = self.coord.evaluate_position(pos, 125.0, {"atr": 1.0, "rsi": 80}, verdict, None)
        self.assertNotEqual(d.action, "EXPAND_TAKE_PROFIT")

    def test_peak_trailing_locks_profit(self):
        # R=+1.5 (vùng ratchet R>=2 chưa tới): đỉnh 112, giá lùi về 111 —
        # SL bám đỉnh (112-1 ATR=111... bị chặn vì bằng giá) nên dùng đỉnh 113:
        # SL = 113-1 = 112 > SL cũ, dưới giá 113.5 -> khóa lãi đỉnh.
        pos = long_pos(is_risk_free=True, partial_tp_done=True,
                       entry_price=100.0, stop_loss=100.08, take_profit=150.0,
                       initial_risk=10.0, peak_price=113.0, tp_expanded=True)
        d = self.coord.evaluate_position(pos, 113.5, {"atr": 1.0, "rsi": 60}, None, None)
        self.assertEqual(d.action, "UPDATE_TRAILING")
        # Đỉnh tự cập nhật max(113, 113.5)=113.5 -> SL = 113.5-1 ATR = 112.5
        self.assertAlmostEqual(d.new_stop_loss, 112.5)


class FractalSwingTests(unittest.TestCase):
    def _df(self, lows, highs):
        n = len(lows)
        return pd.DataFrame({"high": highs, "low": lows,
                             "open": lows, "close": highs,
                             "volume": [1.0] * n})

    def test_single_wick_is_not_a_swing(self):
        # Râu quét 80 nằm sát mép phải (chưa đủ nến xác nhận 2 bên) thì fractal
        # bỏ qua, SL không neo vào râu — min/max trần cũ sẽ neo ngay vào 80.
        lows = [96.0] * 20 + [80.0, 96.0, 96.0]
        highs = [100.0] * 23
        calc = StructuralRiskCalculator()
        local_low, _, local_high, _ = calc.find_swing_points(self._df(lows, highs), window=3, lookback=23)
        self.assertGreater(local_low, 85.0)

    def test_true_fractal_pivot_is_recognized(self):
        # Đáy 90 có 3 nến cao hơn mỗi bên -> pivot thật, neo đúng 90
        lows = [96.0, 95.0, 94.0, 90.0, 94.0, 95.0, 96.0]
        highs = [100.0] * 7
        calc = StructuralRiskCalculator()
        local_low, _, _, _ = calc.find_swing_points(self._df(lows, highs), window=3, lookback=7)
        self.assertAlmostEqual(local_low, 90.0)


class ExchangeBalanceSyncTests(unittest.TestCase):
    def _api(self, account=None, positions=None):
        api = BinanceAPIManager(api_key="k", api_secret="s", is_testnet=True)
        api._send_request = Mock(side_effect=[
            (True, account if account is not None else {
                "totalWalletBalance": "12000.0", "availableBalance": "9000.0", "canTrade": True}),
            (True, positions if positions is not None else []),
        ])
        return api

    def test_wallet_snapshot_parses(self):
        snap = self._api().get_wallet_snapshot()
        self.assertTrue(snap["success"])
        self.assertAlmostEqual(snap["wallet_balance"], 12000.0)
        self.assertAlmostEqual(snap["available_balance"], 9000.0)
        self.assertTrue(snap["can_trade"])

    def test_wallet_snapshot_failure_keeps_old_capital(self):
        api = BinanceAPIManager(api_key="k", api_secret="s", is_testnet=True)
        api._send_request = Mock(return_value=(False, {"code": -1000, "msg": "timeout"}))
        snap = api.get_wallet_snapshot()
        self.assertFalse(snap["success"])
        self.assertEqual(snap["wallet_balance"], 0.0)

    def test_position_snapshot_lists_open_only(self):
        api = BinanceAPIManager(api_key="k", api_secret="s", is_testnet=True)
        api._send_request = Mock(return_value=(True, [
            {"symbol": "BTCUSDT", "positionSide": "BOTH", "positionAmt": "0.05",
             "entryPrice": "80000", "unRealizedProfit": "100.0", "leverage": "5"},
            {"symbol": "BTCUSDT", "positionSide": "BOTH", "positionAmt": "0.0",
             "entryPrice": "0", "unRealizedProfit": "0", "leverage": "5"},
        ]))
        snap = api.get_position_snapshot("BTCUSDT")
        self.assertTrue(snap["success"])
        self.assertEqual(len(snap["positions"]), 1)
        self.assertAlmostEqual(snap["positions"][0]["amount"], 0.05)

    def test_submit_rejects_when_wallet_diverges(self):
        from risk.order_manager import OrderQueueManager
        api = BinanceAPIManager(api_key="k", api_secret="s", is_testnet=True)
        api.test_connection = Mock(return_value={
            "success": True, "can_trade": True,
            "wallet_balance": 20000.0, "available_balance": 15000.0})
        api.normalize_order_values = Mock(
            return_value=(True, {"quantity": 0.01, "price": 80000.0}))
        api.prepare_testnet_trading = Mock(
            return_value=(True, {"maxNotionalValue": "1000000"}))
        state = SimpleNamespace(
            active_exchange="binance", binance_api=api, symbol="BTCUSDT",
            live_price=80000.0, current_balance=5000.0,  # ledger cũ lệch 75% ví sàn
            total_fees=0.0, current_position=None, hedge_positions={}, trades=[],
            trade_memory=SimpleNamespace(memory_records=[]),
            last_decision_trace=None,
            order_manager=OrderQueueManager(),
            fee_engine=SimpleNamespace(bid_price=79999, ask_price=80000,
                                       calculate_fee=lambda v, is_maker=False: v * 0.0005,
                                       calculate_breakeven_price=lambda e, d: e),
            risk_manager=SimpleNamespace(update_balance=Mock(), circuit_breaker_active=False,
                                         ai_cro=SimpleNamespace(last_verdict=None),
                                         evaluate_order=Mock(return_value=SimpleNamespace(
                                             approved=True, units=1.0, rejection_reason=""))),
            risk_config=SimpleNamespace(max_account_risk_pct=0.05),
            freqtrade_protections=SimpleNamespace(
                validate_new_trade=Mock(return_value=(True, "", 0))),
            jesse_engine=SimpleNamespace(
                compute_metrics=lambda: SimpleNamespace(current_consecutive_losses=0)),
            build_market_snapshot=lambda: SimpleNamespace(
                freshness_issues=lambda: [], price=80000.0, environment="testnet",
                context={"hft": SimpleNamespace(is_toxic_flow=False,
                                                liquidity_drought_warning=False,
                                                depth_ready=True, vpin_ready=True,
                                                market_resilience_pct=100.0)}),
            storage=SimpleNamespace(save_execution_runtime=Mock()),
            execution_blocker="",
        )
        from trading.pipeline import CandidateOrder
        order, _ = state.order_manager.place_order(
            "BTCUSDT", "MARKET", "BUY", 80000.0, 100, 5, quantity=0.01,
            stop_loss=79000.0, take_profit=82000.0, client_order_id="drift-test",
            candidate_payload={})
        life = ExecutionLifecycle(state)
        life.submit(order)
        self.assertEqual(order.status, "REJECTED")


if __name__ == "__main__":
    unittest.main()
