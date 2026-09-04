# -*- coding: utf-8 -*-
"""
Unit Test Suite for HKUDS Vibe-Trading Quantitative Integration
Tests:
1. VibeAlphaZooEngine: 12 Alpha factors calculation (EMA acceleration, Kyle's Lambda, Amihud, OU, Yang-Zhang, etc.)
2. VibeSwarmCouncil: 4-Agent Debate (Macro, Quant, Risk, Execution), 9Router connection, fallback, consensus threshold.
3. ShadowAccountAnalyzer: Behavioral bias detection (Premature exit, Loss aversion, Revenge trade), Discipline Score.
4. PersistentStorage: Vibe swarm settings persistence in SQLite.
"""
import unittest
import pandas as pd
import numpy as np
import time
from dataclasses import asdict

from strategy.vibe_alpha_zoo import VibeAlphaZooEngine, VibeAlphaMetrics
from strategy.vibe_swarm_council import VibeSwarmCouncil, SwarmCouncilVerdict
from strategy.shadow_account import ShadowAccountAnalyzer, BehavioralBiasReport
from data.persistent_storage import PersistentStorageManager


class TestVibeTrading(unittest.TestCase):

    def setUp(self):
        # Generate 60 bars of synthetic OHLCV data
        np.random.seed(42)
        n = 60
        base_price = 50000.0
        returns = np.random.normal(0.0005, 0.003, n)
        prices = base_price * np.cumprod(1 + returns)
        
        df = pd.DataFrame({
            "open": prices * (1 + np.random.uniform(-0.001, 0.001, n)),
            "high": prices * (1 + np.random.uniform(0.001, 0.004, n)),
            "low": prices * (1 - np.random.uniform(0.001, 0.004, n)),
            "close": prices,
            "volume": np.random.uniform(10.0, 150.0, n)
        })
        self.df = df
        self.current_price = float(df["close"].iloc[-1])

    def test_alpha_zoo_12_factors(self):
        zoo = VibeAlphaZooEngine(ema_period=20, lookback=50)
        metrics = zoo.evaluate(
            df=self.df,
            current_price=self.current_price,
            best_bid=self.current_price - 0.5,
            best_ask=self.current_price + 0.5,
            spread=1.0
        )
        self.assertIsInstance(metrics, VibeAlphaMetrics)
        
        # Verify factors exist and are finite numbers
        self.assertTrue(np.isfinite(metrics.ema_slope))
        self.assertTrue(np.isfinite(metrics.ema_acceleration))
        self.assertTrue(np.isfinite(metrics.momentum_zscore))
        self.assertTrue(np.isfinite(metrics.kyles_lambda))
        self.assertTrue(np.isfinite(metrics.amihud_illiquidity))
        self.assertTrue(np.isfinite(metrics.volume_force_ratio))
        self.assertTrue(np.isfinite(metrics.yang_zhang_vol))
        self.assertGreater(metrics.yang_zhang_vol, 0.0)
        self.assertGreaterEqual(metrics.ou_half_life_bars, 1.0)
        self.assertGreaterEqual(metrics.squeeze_intensity, 0.0)
        self.assertLessEqual(metrics.squeeze_intensity, 1.0)
        self.assertTrue(np.isfinite(metrics.tail_risk_skew))
        self.assertGreaterEqual(metrics.absorption_ratio, 0.0)
        self.assertLessEqual(metrics.absorption_ratio, 1.0)
        
        # Composite score
        self.assertGreaterEqual(metrics.composite_alpha_score, -100.0)
        self.assertLessEqual(metrics.composite_alpha_score, 100.0)
        self.assertIn(metrics.alpha_regime, [
            "STRONG_BULLISH_MOMENTUM", "BULLISH", "NEUTRAL",
            "BEARISH", "STRONG_BEARISH_MOMENTUM"
        ])
        self.assertTrue(len(metrics.summary_thesis) > 10)

    def test_swarm_council_fallback_and_consensus(self):
        # Swarm with 3 required votes
        council = VibeSwarmCouncil(
            gateway_url="http://127.0.0.1:8039/v1",
            default_model="gemini-2.0-flash",
            min_votes_required=3,
            enabled=True
        )
        self.assertEqual(council.min_votes_required, 3)
        self.assertEqual(council.consensus_threshold, 3)
        self.assertTrue(council.enabled)

        # Evaluate council in fallback mode
        verdict = council._fallback_cognitive_swarm(
            current_price=self.current_price,
            indicators={"ema": self.current_price * 0.99, "rsi": 55.0, "atr": 200.0, "adx": 28.0},
            ai_verdict=None,
            ensemble_result=None,
            order_research=None,
            alpha_zoo_metrics=None,
            visual_hft_metrics=None,
            jesse_metrics=None,
            octobot_metrics=None,
            current_position=None,
            user_instruction="Test debate"
        )
        self.assertIsInstance(verdict, SwarmCouncilVerdict)
        self.assertEqual(len(verdict.votes), 4)
        
        # Check all 4 agents voted
        agent_ids = [v.agent_id for v in verdict.votes]
        self.assertIn("macro", agent_ids)
        self.assertIn("quant", agent_ids)
        self.assertIn("risk", agent_ids)
        self.assertIn("execution", agent_ids)

        # Check verdict logic
        self.assertGreaterEqual(verdict.approved_votes, 0)
        self.assertEqual(verdict.total_votes, 4)
        self.assertEqual(verdict.approved, verdict.approved_votes >= council.min_votes_required)

        # Test threshold change to 4
        council.update_config(min_votes=4)
        self.assertEqual(council.min_votes_required, 4)

        # Test model override
        council.update_config(macro_model="gemini-2.5-pro", risk_model="deepseek-r1")
        self.assertEqual(council.agent_models["macro"], "gemini-2.5-pro")
        self.assertEqual(council.agent_models["risk"], "deepseek-r1")

    def test_shadow_account_biases_and_discipline(self):
        analyzer = ShadowAccountAnalyzer()
        now = time.time()

        # Synthetic trades containing:
        # 1. Premature exit (manual profit cut early)
        # 2. Loss aversion (large loss > $45)
        # 3. Revenge trade (trade opened within 60s after a loss)
        sample_trades = [
            {
                "id": 1,
                "direction": "LONG",
                "entry_price": 50000.0,
                "exit_price": 50050.0,
                "pnl": 5.0,
                "pnl_pct": 0.1,
                "reason": "Đóng thủ công",
                "open_time": now - 1000,
                "closed_at_ts": now - 800
            },
            {
                "id": 2,
                "direction": "LONG",
                "entry_price": 50100.0,
                "exit_price": 49400.0,
                "pnl": -55.0,
                "pnl_pct": -1.4,
                "reason": "Cắt lỗ SL",
                "open_time": now - 700,
                "closed_at_ts": now - 500
            },
            {
                "id": 3,
                "direction": "SHORT",
                "entry_price": 49350.0,
                "exit_price": 49300.0,
                "pnl": 4.0,
                "pnl_pct": 0.1,
                "reason": "Vào trả thù",
                "open_time": now - 450,
                "closed_at_ts": now - 300
            }
        ]

        report = analyzer.analyze_trade_history(sample_trades, initial_balance=5000.0)
        self.assertIsInstance(report, BehavioralBiasReport)
        self.assertEqual(report.premature_exits_count, 1)
        self.assertEqual(report.loss_aversion_count, 1)
        self.assertEqual(report.revenge_trades_count, 1)
        
        # Penalties: 6 (premature) + 10 (loss) + 14 (revenge) = 30 -> 100 - 30 = 70
        self.assertEqual(report.discipline_score, 70)
        self.assertEqual(report.discipline_grade, "B")
        self.assertGreater(report.missed_alpha_usdt, 0.0)
        self.assertTrue(len(report.behavioral_diagnostics) > 5)

    def test_persistent_storage_vibe_config(self):
        storage = PersistentStorageManager(db_path="data/trading_platform_test.db")
        cfg = storage.save_vibe_config(
            enabled=True,
            min_votes=4,
            macro_model="ag/gemini-3.8-flash-high",
            quant_model="ag/claude-sonnet-4-6",
            risk_model="ds/deepseek-reasoner",
            exec_model="ag/gemini-3.8-flash"
        )
        self.assertTrue(cfg["enabled"])
        self.assertEqual(cfg["min_votes"], 4)
        self.assertEqual(cfg["macro_model"], "ag/gemini-3.8-flash-high")
        self.assertEqual(cfg["risk_model"], "ds/deepseek-reasoner")

        loaded = storage.get_vibe_config()
        self.assertTrue(loaded["enabled"])
        self.assertEqual(loaded["min_votes"], 4)
        self.assertEqual(loaded["macro_model"], "ag/gemini-3.8-flash-high")
        self.assertEqual(loaded["quant_model"], "ag/claude-sonnet-4-6")
        self.assertEqual(loaded["risk_model"], "ds/deepseek-reasoner")
        self.assertEqual(loaded["exec_model"], "ag/gemini-3.8-flash")


if __name__ == "__main__":
    unittest.main()
