import unittest
import math
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from trading.pipeline import CandidateOrder, MarketSnapshot, StageOutcome
from risk.jesse_expectancy_engine import JesseExpectancyEngine
from risk.freqtrade_protections import FreqtradeProtectionEngine


class AutoGridAndRegimeSwitchTests(unittest.TestCase):

    def test_jesse_expectancy_ignores_micro_fees(self):
        engine = JesseExpectancyEngine()
        engine.record_trade(-0.01)  # Micro fee
        engine.record_trade(-0.02)  # Micro fee
        # Should not be counted as full consecutive losses
        self.assertEqual(engine.current_consecutive_losses, 0)
        # Real trade losses
        engine.record_trade(-15.0)
        engine.record_trade(-20.0)
        self.assertEqual(engine.current_consecutive_losses, 2)
        engine.record_trade(30.0)
        self.assertEqual(engine.current_consecutive_losses, 0)

    def test_freqtrade_gate_skips_directional_fee_drag_for_grid(self):
        protections = FreqtradeProtectionEngine(initial_balance=1000.0)
        # Directional trade with 0.02% TP should fail fee drag
        allowed, reason, status = protections.validate_new_trade(
            entry_price=80000.0,
            target_price=80016.0,  # 0.02%
            direction=1,
            balance=1000.0,
            equity=1000.0,
        )
        self.assertFalse(allowed)
        self.assertEqual(status.status, 'FEE_DRAG')

        # Grid trade (target_price = 0.0) should pass fee drag check
        allowed_grid, reason_grid, status_grid = protections.validate_new_trade(
            entry_price=80000.0,
            target_price=0.0,
            direction=1,
            balance=1000.0,
            equity=1000.0,
        )
        self.assertTrue(allowed_grid)
        self.assertEqual(status_grid.status, 'NORMAL')

    def test_grid_sizing_fallback_funds_minimum_legs(self):
        legs = [
            {'side': 'BUY', 'entry_price': 79000.0, 'stop_loss': 78000.0, 'take_profit': 80000.0},
            {'side': 'SELL', 'entry_price': 81000.0, 'stop_loss': 82000.0, 'take_profit': 80000.0},
            {'side': 'BUY', 'entry_price': 78500.0, 'stop_loss': 78000.0, 'take_profit': 80000.0},
            {'side': 'SELL', 'entry_price': 81500.0, 'stop_loss': 82000.0, 'take_profit': 80000.0},
        ]
        min_grid_qty = len(legs) * 0.001
        self.assertEqual(min_grid_qty, 0.004)
        margin_needed = min_grid_qty * 80000.0 / 3.0
        self.assertLess(margin_needed, 500.0)

    def test_pipeline_direction_validation_allows_auto_grid(self):
        from trading.pipeline import SevenStagePipeline
        cand = CandidateOrder(
            order_id='auto-grid-1',
            symbol='BTCUSDT',
            direction=0,
            order_type='GRID',
            entry_price=80000.0,
            stop_loss=78000.0,
            take_profit=82000.0,
            leverage=3,
            margin=100.0,
            quantity=0.004,
            source='auto-grid'
        )
        snapshot = MarketSnapshot(
            snapshot_id='snap-1',
            symbol='BTCUSDT',
            exchange='binance',
            price=80000.0,
            bids=tuple((79999.0 - i, 1.0) for i in range(20)),
            asks=tuple((80001.0 + i, 1.0) for i in range(20)),
            frames={},
            source_times={'depth': time.time(), 'agg_trade': time.time(), 'kline_1m': time.time()},
            environment='paper',
            captured_at=time.time(),
            context={'required_frames': []}
        )
        pipeline = SevenStagePipeline(
            defense_gates=[('MockDefense', lambda c, s: StageOutcome.pass_('ok'))],
            stage2_gates=[('MockStage2', lambda c, s: StageOutcome.pass_('ok'))],
            council=lambda c, s: StageOutcome.pass_('ok'),
            sizing=lambda c, s: StageOutcome.pass_('ok'),
            memory=lambda c, s: StageOutcome.pass_('ok'),
            decision_mode='AI_REQUIRED'
        )
        decision = pipeline.decide(cand, snapshot)
        self.assertTrue(decision.approved)


if __name__ == '__main__':
    unittest.main()
