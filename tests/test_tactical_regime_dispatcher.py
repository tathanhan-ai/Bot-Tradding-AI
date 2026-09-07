import unittest
from types import SimpleNamespace
import pandas as pd
import numpy as np

from strategy.tactical_regime_dispatcher import TacticalRegimeDispatcher, TacticalFormation, TacticalStrategyProposal
from risk.ai_order_researcher import AIOrderResearcher, AIOrderResearchResult


def make_ohlcv_df(n=100, base_price=80000.0, trend=0.0):
    prices = [base_price + i * trend + np.sin(i / 5.0) * 150.0 for i in range(n)]
    return pd.DataFrame({
        'open': prices,
        'high': [p + 40.0 for p in prices],
        'low': [p - 40.0 for p in prices],
        'close': prices,
        'volume': [50.0 + (i % 7) for i in range(n)]
    }, index=pd.date_range('2026-01-01', periods=n, freq='15min'))


class TacticalRegimeDispatcherTests(unittest.TestCase):
    def setUp(self):
        self.dispatcher = TacticalRegimeDispatcher()
        self.df_bull = make_ohlcv_df(100, base_price=80000.0, trend=25.0)
        self.df_sideway = make_ohlcv_df(100, base_price=80000.0, trend=0.0)
        self.account_balance = 5059.69

    def test_swing_trend_formation_in_trending_market(self):
        indicators = {"atr": 250.0, "rsi": 62.0, "adx": 32.0, "chop": 38.0, "ema20": 81000.0}
        octo_mock = SimpleNamespace(recommended_direction=1, active_trading_mode="TRENDING")
        proposal = self.dispatcher.dispatch_strategy(
            current_price=81500.0,
            best_bid=81490.0,
            best_ask=81510.0,
            df_structure=self.df_bull,
            df_macro=self.df_bull,
            indicators=indicators,
            active_timeframe="1h",
            current_balance=self.account_balance,
            octobot_consensus=octo_mock
        )
        self.assertIsInstance(proposal, TacticalStrategyProposal)
        self.assertEqual(proposal.formation, TacticalFormation.SWING_TREND)
        self.assertEqual(proposal.recommended_side, "BUY")
        self.assertGreaterEqual(proposal.optimal_margin, 600.0)
        self.assertLessEqual(proposal.optimal_margin, 1600.0)
        self.assertGreaterEqual(proposal.risk_reward_ratio, 2.0)
        self.assertGreater(len(proposal.staged_exits), 0)

    def test_defensive_sniper_formation_in_sideway_market(self):
        indicators = {"atr": 180.0, "rsi": 42.0, "adx": 18.0, "chop": 52.0}
        proposal = self.dispatcher.dispatch_strategy(
            current_price=80000.0,
            best_bid=79995.0,
            best_ask=80005.0,
            df_structure=self.df_sideway,
            df_macro=self.df_sideway,
            indicators=indicators,
            active_timeframe="15m",
            current_balance=self.account_balance
        )
        self.assertIsInstance(proposal, TacticalStrategyProposal)
        self.assertIn(proposal.formation, (TacticalFormation.DEFENSIVE_SNIPER, TacticalFormation.LIQUIDITY_SWEEP))
        self.assertGreaterEqual(proposal.optimal_margin, 500.0)
        self.assertLessEqual(proposal.optimal_margin, 1600.0)
        self.assertGreaterEqual(proposal.risk_reward_ratio, 2.0)
        # Verify SL distance is reasonable and not the old artificial $720
        self.assertLess(proposal.stop_distance_usdt, 500.0)

    def test_deficit_patience_formation_when_carried_deficit_high(self):
        indicators = {"atr": 180.0, "rsi": 50.0, "adx": 19.0, "chop": 55.0}
        gov_status = SimpleNamespace(enabled=True, regime="DEFENSIVE_PATIENCE", carried_deficit_pct=18.0, min_risk_reward_ratio=2.5)
        proposal = self.dispatcher.dispatch_strategy(
            current_price=80000.0,
            best_bid=79995.0,
            best_ask=80005.0,
            df_structure=self.df_sideway,
            df_macro=self.df_sideway,
            indicators=indicators,
            active_timeframe="15m",
            current_balance=self.account_balance,
            governor_status=gov_status
        )
        self.assertEqual(proposal.formation, TacticalFormation.DEFICIT_PATIENCE)
        self.assertLessEqual(proposal.optimal_leverage, 4)
        self.assertGreaterEqual(proposal.risk_reward_ratio, 2.5)

    def test_ai_order_researcher_tactical_integration(self):
        researcher = AIOrderResearcher()
        indicators = {"atr": 180.0, "rsi": 45.0, "adx": 18.0, "chop": 52.0}
        ai_verdict = SimpleNamespace(regime="RANGING_SIDEWAY")
        ensemble = SimpleNamespace(consensus_score=0.0, confidence=75.0, consensus_verdict="NEUTRAL")
        gov_status = SimpleNamespace(enabled=True, regime="ON_TRACK", carried_deficit_pct=0.0, size_multiplier=1.0, min_risk_reward_ratio=2.0)

        res = researcher.research(
            current_price=80000.0,
            best_bid=79995.0,
            best_ask=80005.0,
            spread=10.0,
            indicators=indicators,
            ai_verdict=ai_verdict,
            ensemble_result=ensemble,
            ai_cro=None,
            current_balance=self.account_balance,
            df_structure=self.df_sideway,
            df_macro=self.df_sideway,
            active_timeframe="15m",
            monthly_governor_status=gov_status
        )
        self.assertIsInstance(res, AIOrderResearchResult)
        self.assertTrue(hasattr(res, "tactical_formation"))
        self.assertIn(res.tactical_formation, [f.value for f in TacticalFormation])
        self.assertGreaterEqual(res.optimal_margin, 400.0)
        self.assertGreaterEqual(res.rr_ratio, 2.0)
        self.assertTrue(len(res.staged_exits) > 0)


if __name__ == '__main__':
    unittest.main()
