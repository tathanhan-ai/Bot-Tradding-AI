import unittest
from unittest.mock import MagicMock
from trading.pipeline import CandidateOrder
from strategy.vibe_swarm_council import SwarmCouncilVerdict, AgentVote
from risk.ai_order_researcher import AIOrderResearcher
import pandas as pd


class TestAdaptiveCouncilRevision(unittest.TestCase):
    def setUp(self):
        self.researcher = AIOrderResearcher()

    def test_swarm_council_verdict_rejection_categorization(self):
        votes = [
            AgentVote(agent_id="risk", agent_name="Chief Risk Officer", icon="🛡️", model_used="test", vote="REJECT", confidence=85, thesis="R:R ratio 1.2 is too low, stop loss 91000 is too loose, high risk"),
            AgentVote(agent_id="macro", agent_name="Macro Economist", icon="🌐", model_used="test", vote="REJECT", confidence=80, thesis="Higher timeframe 1h trend is bearish, counter-trend conflict"),
            AgentVote(agent_id="quant", agent_name="Quant Strategist", icon="📐", model_used="test", vote="APPROVE", confidence=75, thesis="Alpha score positive"),
            AgentVote(agent_id="execution", agent_name="Execution Specialist", icon="⚡", model_used="test", vote="REJECT", confidence=70, thesis="Spread is wide, high slippage and toxic flow VPIN")
        ]
        verdict = SwarmCouncilVerdict(
            council_verdict="REJECT",
            consensus_passed=False,
            approvals_count=1,
            total_agents=4,
            min_votes_required=3,
            votes=votes,
            council_rationale="Risk and Macro objections",
            available=True,
            can_negotiate=True,
            rejection_categories={
                "risk": [votes[0].thesis],
                "macro": [votes[1].thesis],
                "execution": [votes[3].thesis],
                "quant": []
            }
        )
        self.assertTrue(verdict.can_negotiate)
        self.assertIn("risk", verdict.rejection_categories)
        self.assertIn("macro", verdict.rejection_categories)
        self.assertIn("execution", verdict.rejection_categories)
        self.assertEqual(len(verdict.rejection_categories["risk"]), 1)

    def test_build_adaptive_counter_proposal_rr_critique(self):
        candidate = CandidateOrder(
            order_id="test-order-1",
            symbol="BTCUSDT",
            direction=1,
            order_type="MARKET",
            entry_price=90000.0,
            stop_loss=89000.0,
            take_profit=91200.0,
            leverage=5,
            quantity=0.05,
            margin=900.0
        )
        indicators = {"atr": 120.0, "rsi": 52.0}
        df_structure = pd.DataFrame([
            {"open": 89500, "high": 90200, "low": 89400, "close": 90000, "volume": 10}
        ])

        alt = self.researcher.build_adaptive_counter_proposal(
            original_candidate=candidate,
            veto_stage="Stage 3 / AI Council",
            veto_reason="Risk: R:R ratio 1.2 is too low, stop loss is too wide, need better risk-reward",
            current_price=90000.0,
            best_bid=89995.0,
            best_ask=90000.0,
            indicators=indicators,
            df_structure=df_structure,
            current_balance=5000.0
        )

        self.assertIsNotNone(alt)
        self.assertTrue(alt.metadata.get("is_counter_proposal"))
        self.assertIn("negotiation_solution", alt.metadata)
        orig_rr = abs(candidate.take_profit - candidate.entry_price) / abs(candidate.entry_price - candidate.stop_loss)
        alt_rr = abs(alt.take_profit - alt.entry_price) / max(1.0, abs(alt.entry_price - alt.stop_loss))
        self.assertGreaterEqual(alt_rr, orig_rr)
        self.assertEqual(alt.order_type, "POST_ONLY")

    def test_build_adaptive_counter_proposal_macro_critique(self):
        candidate = CandidateOrder(
            order_id="test-order-2",
            symbol="BTCUSDT",
            direction=1,
            order_type="MARKET",
            entry_price=90000.0,
            stop_loss=89500.0,
            take_profit=91500.0,
            leverage=5,
            quantity=0.05,
            margin=900.0
        )
        indicators = {"atr": 120.0, "rsi": 40.0}

        alt = self.researcher.build_adaptive_counter_proposal(
            original_candidate=candidate,
            veto_stage="Stage 3 / AI Council",
            veto_reason="Macro Officer: Counter-trend 1h macro downtrend, strongly advise against long position",
            current_price=90000.0,
            best_bid=89995.0,
            best_ask=90000.0,
            indicators=indicators,
            current_balance=5000.0
        )

        self.assertIsNotNone(alt)
        self.assertTrue(alt.metadata.get("is_counter_proposal"))
        self.assertTrue(alt.order_type in ("GRID", "POST_ONLY", "LIMIT") or alt.direction == -1 or "trend" in alt.metadata.get("negotiation_solution", "").lower() or "grid" in alt.metadata.get("negotiation_solution", "").lower())

    def test_build_adaptive_counter_proposal_sizing_critique(self):
        candidate = CandidateOrder(
            order_id="test-order-3",
            symbol="BTCUSDT",
            direction=-1,
            order_type="MARKET",
            entry_price=90000.0,
            stop_loss=90500.0,
            take_profit=88500.0,
            leverage=10,
            quantity=0.1,
            margin=1500.0
        )
        indicators = {"atr": 120.0, "rsi": 58.0}

        alt = self.researcher.build_adaptive_counter_proposal(
            original_candidate=candidate,
            veto_stage="Stage 4 / Carver + Risk",
            veto_reason="Exceeds allowable margin utilization and risk per unit cap",
            current_price=90000.0,
            best_bid=90000.0,
            best_ask=90005.0,
            indicators=indicators,
            current_balance=5000.0
        )

        self.assertIsNotNone(alt)
        self.assertTrue(alt.metadata.get("is_counter_proposal"))
        self.assertLessEqual(alt.leverage, candidate.leverage)
        self.assertIn(alt.order_type, ("SCALE_RATIO", "POST_ONLY", "LIMIT"))


if __name__ == "__main__":
    unittest.main()
