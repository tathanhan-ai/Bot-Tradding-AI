import unittest
import pandas as pd
import numpy as np
from types import SimpleNamespace
from unittest.mock import MagicMock

from strategy.ensemble_strategy import EnsembleCoordinator
from strategy.multi_candle_patterns import MultiTimeframeCandleStrategyEngine
from strategy.grid_planner import GridPlanner
from trading.pipeline import CandidateOrder


def make_dummy_df(n=50, base_price=100.0, trend=0.0):
    prices = [base_price + i * trend + np.sin(i / 3.0) for i in range(n)]
    return pd.DataFrame({
        'open': prices,
        'high': [p + 0.5 for p in prices],
        'low': [p - 0.5 for p in prices],
        'close': prices,
        'volume': [10.0 + (i % 5) for i in range(n)]
    }, index=pd.date_range('2026-01-01', periods=n, freq='15min'))


class AdaptiveMTFAndAutoGridTests(unittest.TestCase):
    def setUp(self):
        self.data_map = {
            '1m': make_dummy_df(60, 100.0),
            '3m': make_dummy_df(60, 100.0),
            '5m': make_dummy_df(60, 100.0),
            '15m': make_dummy_df(60, 100.0),
            '30m': make_dummy_df(60, 100.0),
            '1h': make_dummy_df(60, 100.0),
            '4h': make_dummy_df(60, 100.0),
            '1d': make_dummy_df(60, 100.0),
            '1w': make_dummy_df(30, 100.0),
            '1M': make_dummy_df(20, 100.0),
        }

    def test_multi_candle_patterns_supports_all_timeframes_and_focus(self):
        engine = MultiTimeframeCandleStrategyEngine()
        res_5m = engine.evaluate(self.data_map, 100.0, active_timeframe='5m')
        self.assertIsNotNone(res_5m)
        self.assertIn('5m', res_5m.timeframe_patterns)
        self.assertIn('1h', res_5m.timeframe_patterns)
        self.assertIn('1d', res_5m.timeframe_patterns)

        res_1m = engine.evaluate(self.data_map, 100.0, active_timeframe='1m')
        self.assertIsNotNone(res_1m)
        self.assertIn('1m', res_1m.timeframe_patterns)

    def test_ensemble_coordinator_adapts_to_active_timeframe(self):
        coordinator = EnsembleCoordinator()
        res_1m = coordinator.evaluate_ensemble(self.data_map, 100.0, 'RANGING_SIDEWAY', active_timeframe='1m')
        self.assertIsNotNone(res_1m)
        vote_names = [v.name for v in res_1m.votes]
        self.assertTrue(any('1M' in name.upper() for name in vote_names))

        res_1h = coordinator.evaluate_ensemble(self.data_map, 100.0, 'TRENDING_BULL', active_timeframe='1h')
        self.assertIsNotNone(res_1h)
        vote_names_1h = [v.name for v in res_1h.votes]
        self.assertTrue(any('1H' in name.upper() for name in vote_names_1h))

    def test_auto_grid_candidate_creation_and_dispatch(self):
        candidate = CandidateOrder(
            order_id='auto-grid-12345',
            direction=0,
            order_type='GRID',
            entry_price=100.0,
            stop_loss=90.0,
            take_profit=110.0,
            leverage=3,
            margin=50.0,
            quantity=1.5,
            source='auto-grid',
            symbol='BTCUSDT'
        )
        self.assertEqual(candidate.source, 'auto-grid')
        self.assertEqual(candidate.order_type, 'GRID')

        from ui.server import LiveTradingState
        mock_state = SimpleNamespace(
            is_grid_active=False,
            grid_status='OFF',
            grid_plan={},
            execution=SimpleNamespace(
                queue_grid=MagicMock(return_value={'status': 'active', 'children': 6})
            )
        )
        plan = {'legs': [{'side': 'BUY', 'entry_price': 98.0}, {'side': 'SELL', 'entry_price': 102.0}]}
        candidate.metadata['grid_plan'] = plan

        res = LiveTradingState.queue_approved_candidate(mock_state, candidate)
        self.assertEqual(res['status'], 'active')
        self.assertTrue(mock_state.is_grid_active)
        self.assertEqual(mock_state.grid_status, 'ACTIVE')
        mock_state.execution.queue_grid.assert_called_once_with(candidate, plan['legs'])


if __name__ == '__main__':
    unittest.main()
