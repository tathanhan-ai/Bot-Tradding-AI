import math
import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from risk.monthly_target_governor import MonthlyTargetGovernor
from trading.pipeline import (
    CandidateOrder,
    DecisionMode,
    DecisionTrace,
    MarketSnapshot,
    PipelineDecision,
    SevenStagePipeline,
    StageOutcome,
)


def make_snapshot(now: float = None, symbol: str = "BTCUSDT", age: float = 0.0) -> MarketSnapshot:
    cur_time = time.time() if now is None else now
    t = cur_time - age
    return MarketSnapshot(
        snapshot_id=f"snap-{int(t)}",
        symbol=symbol,
        exchange="binance",
        captured_at=t,
        price=60000.0,
        source_times={"depth": t, "agg_trade": t, "kline_1m": t},
        bids=[[59990.0 - i * 1.0, 1.0] for i in range(20)],
        asks=[[60010.0 + i * 1.0, 1.0] for i in range(20)],
    )


def make_candidate(direction: int = 1, source: str = "auto", **metadata) -> CandidateOrder:
    return CandidateOrder(
        order_id="cand-123",
        symbol="BTCUSDT",
        direction=direction,
        order_type="MARKET",
        entry_price=60000.0,
        stop_loss=59400.0,
        take_profit=61200.0,
        leverage=3,
        source=source,
        metadata=metadata,
    )


class FailClosedPolicyTests(unittest.TestCase):
    def test_entry_fails_closed_when_council_is_unavailable_in_ai_required_mode(self):
        now = 1000.0
        snapshot = make_snapshot(now=now)
        candidate = make_candidate()

        def mock_council_unavailable(_cand, _snap):
            return StageOutcome.unavailable("9Router gateway 504 Gateway Timeout")

        pipeline = SevenStagePipeline(
            defense_gates=[("Defense", lambda *_: StageOutcome.pass_("ok"))],
            stage2_gates=[("Stage2", lambda *_: StageOutcome.pass_("ok"))],
            council=mock_council_unavailable,
            sizing=lambda cand, *_: StageOutcome.pass_("ok", quantity=0.01),
            memory=lambda *_: StageOutcome.pass_("ok"),
            decision_mode=DecisionMode.AI_REQUIRED,
        )

        decision = pipeline.decide(candidate, snapshot, now=now)
        self.assertFalse(decision.approved)
        last_entry = decision.trace.entries[-1]
        self.assertEqual(last_entry.stage, "Stage 3 / AI Council")
        self.assertEqual(last_entry.verdict, "VETO")
        self.assertIn("504 Gateway Timeout", last_entry.reason)

    def test_entry_fails_closed_when_council_is_none_in_ai_required_mode(self):
        now = 1000.0
        snapshot = make_snapshot(now=now)
        candidate = make_candidate()

        pipeline = SevenStagePipeline(
            defense_gates=[("Defense", lambda *_: StageOutcome.pass_("ok"))],
            stage2_gates=[("Stage2", lambda *_: StageOutcome.pass_("ok"))],
            council=None,
            sizing=lambda cand, *_: StageOutcome.pass_("ok", quantity=0.01),
            memory=lambda *_: StageOutcome.pass_("ok"),
            decision_mode=DecisionMode.AI_REQUIRED,
        )

        decision = pipeline.decide(candidate, snapshot, now=now)
        self.assertFalse(decision.approved)
        last_entry = decision.trace.entries[-1]
        self.assertEqual(last_entry.verdict, "VETO")
        self.assertIn("Fail-Closed", last_entry.reason)

    def test_entry_fails_closed_on_council_agent_abstain(self):
        now = 1000.0
        snapshot = make_snapshot(now=now)
        candidate = make_candidate()

        def mock_council_abstain(_cand, _snap):
            return StageOutcome.veto("Council rejected: Agent macro abstained; consensus quorum not reached")

        pipeline = SevenStagePipeline(
            defense_gates=[("Defense", lambda *_: StageOutcome.pass_("ok"))],
            stage2_gates=[("Stage2", lambda *_: StageOutcome.pass_("ok"))],
            council=mock_council_abstain,
            sizing=lambda cand, *_: StageOutcome.pass_("ok", quantity=0.01),
            memory=lambda *_: StageOutcome.pass_("ok"),
            decision_mode=DecisionMode.AI_REQUIRED,
        )

        decision = pipeline.decide(candidate, snapshot, now=now)
        self.assertFalse(decision.approved)
        self.assertEqual(decision.trace.entries[-1].verdict, "VETO")
        self.assertIn("consensus quorum not reached", decision.trace.entries[-1].reason)

    def test_risk_reducing_exit_bypasses_council_and_executes_even_if_offline(self):
        now = 1000.0
        snapshot = make_snapshot(now=now)
        exit_candidate = make_candidate(
            direction=-1,
            source="exit",
            reduce_only=True,
            intent="EMERGENCY_CLOSE",
            is_exit=True,
        )

        def mock_council_explodes(_cand, _snap):
            raise ConnectionRefusedError("9Router connection refused")

        pipeline = SevenStagePipeline(
            defense_gates=[("Defense", lambda *_: StageOutcome.pass_("ok"))],
            stage2_gates=[("Stage2", lambda *_: StageOutcome.pass_("ok"))],
            council=mock_council_explodes,
            sizing=lambda cand, *_: StageOutcome.pass_("ok", quantity=0.05),
            memory=lambda *_: StageOutcome.pass_("ok"),
            decision_mode=DecisionMode.AI_REQUIRED,
        )

        decision = pipeline.decide(exit_candidate, snapshot, now=now)
        self.assertTrue(decision.approved)
        council_entry = next(e for e in decision.trace.entries if e.stage == "Stage 3 / AI Council")
        self.assertEqual(council_entry.verdict, "PASS")
        self.assertIn("Risk-reducing operation", council_entry.reason)

    def test_deterministic_only_mode_allows_entries_without_council(self):
        now = 1000.0
        snapshot = make_snapshot(now=now)
        candidate = make_candidate()

        pipeline = SevenStagePipeline(
            defense_gates=[("Defense", lambda *_: StageOutcome.pass_("ok"))],
            stage2_gates=[("Stage2", lambda *_: StageOutcome.pass_("ok"))],
            council=None,
            sizing=lambda cand, *_: StageOutcome.pass_("ok", quantity=0.02),
            memory=lambda *_: StageOutcome.pass_("ok"),
            decision_mode=DecisionMode.DETERMINISTIC_ONLY,
        )

        decision = pipeline.decide(candidate, snapshot, now=now)
        self.assertTrue(decision.approved)
        council_entry = next(e for e in decision.trace.entries if e.stage == "Stage 3 / AI Council")
        self.assertEqual(council_entry.verdict, "PASS")
        self.assertIn("DETERMINISTIC_ONLY", council_entry.reason)

    def test_exit_only_mode_blocks_entries_but_permits_exits(self):
        now = 1000.0
        snapshot = make_snapshot(now=now)
        entry_candidate = make_candidate(direction=1, source="auto")
        exit_candidate = make_candidate(direction=-1, source="exit", reduce_only=True)

        pipeline = SevenStagePipeline(
            defense_gates=[("Defense", lambda *_: StageOutcome.pass_("ok"))],
            stage2_gates=[("Stage2", lambda *_: StageOutcome.pass_("ok"))],
            council=lambda *_: StageOutcome.pass_("ok"),
            sizing=lambda cand, *_: StageOutcome.pass_("ok", quantity=0.01),
            memory=lambda *_: StageOutcome.pass_("ok"),
            decision_mode=DecisionMode.EXIT_ONLY,
        )

        entry_decision = pipeline.decide(entry_candidate, snapshot, now=now)
        self.assertFalse(entry_decision.approved)
        self.assertIn("EXIT_ONLY mode", entry_decision.trace.entries[0].reason)

        exit_decision = pipeline.decide(exit_candidate, snapshot, now=now)
        self.assertTrue(exit_decision.approved)

    def test_monthly_governor_anti_target_chasing(self):
        governor = MonthlyTargetGovernor(base_target_pct=10.0, enabled=True)
        governor.carried_deficit_pct = 4.0

        # In deficit ($100 profit on $5000 capital, below target with carried deficit):
        trades_deficit = [{"pnl": 100.0, "time": time.time()}]
        status_deficit = governor.evaluate(current_balance=5000.0, trades=trades_deficit)

        # Anti-target-chasing invariant: size multiplier MUST NEVER exceed 1.0!
        self.assertLessEqual(status_deficit.size_multiplier, 0.85)
        # Leverage cap must be protected (<= 4)
        self.assertLessEqual(status_deficit.max_leverage_cap, 4)
        self.assertEqual(status_deficit.regime, "DEFICIT_CATCHUP")

        # In target achieved (e.g. $800 profit on $5000 capital = 16%):
        trades_achieved = [{"pnl": 800.0, "time": time.time()}]
        status_achieved = governor.evaluate(current_balance=5800.0, trades=trades_achieved)
        self.assertLessEqual(status_achieved.size_multiplier, 0.6)
        self.assertEqual(status_achieved.regime, "TARGET_ACHIEVED")

    def test_stale_market_snapshot_fails_closed_at_stage_0(self):
        now = 1000.0
        stale_snapshot = make_snapshot(now=now, age=15.0)  # > 3s freshness threshold
        candidate = make_candidate()

        defense_called = []
        pipeline = SevenStagePipeline(
            defense_gates=[("Defense", lambda *_: defense_called.append(True) or StageOutcome.pass_("ok"))],
            stage2_gates=[],
            council=None,
            sizing=None,
            memory=None,
        )

        decision = pipeline.decide(candidate, stale_snapshot, now=now)
        self.assertFalse(decision.approved)
        self.assertEqual(len(decision.trace.entries), 1)
        self.assertEqual(decision.trace.entries[0].stage, "Stage 0 / Market Snapshot")
        self.assertEqual(decision.trace.entries[0].verdict, "VETO")
        self.assertIn("exceeds 3s", decision.trace.entries[0].reason)
        self.assertEqual(defense_called, [])  # Never invoked downstream gates


if __name__ == "__main__":
    unittest.main()
