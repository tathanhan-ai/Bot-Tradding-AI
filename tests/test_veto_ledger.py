"""P0: so theo doi veto theo tung cong (quan sat, khong doi logic).

Kiem tra _record_veto_ledger dem dung, giu ly do moi nhat, reset khi PASS.
"""
import unittest
from types import SimpleNamespace

from trading.pipeline import DecisionTrace, StageOutcome
from ui.server import LiveTradingState


def make_state():
    state = LiveTradingState.__new__(LiveTradingState)
    state.veto_ledger = {}
    return state


def make_trace(*stage_verdicts):
    trace = DecisionTrace("order-1", "snap-1")
    for stage, verdict, reason in stage_verdicts:
        if verdict == "VETO":
            trace.add(stage, StageOutcome.veto(reason))
        else:
            trace.add(stage, StageOutcome.pass_(reason))
    return trace


def make_candidate(order_type="MARKET", direction=1):
    return SimpleNamespace(order_type=order_type, direction=direction, metadata={})


class VetoLedgerTests(unittest.TestCase):
    def test_counts_veto_per_stage(self):
        state = make_state()
        trace = make_trace(("Council", "VETO", "Chi 1/4 phieu dong y"),
                           ("OctoBot", "VETO", "Matrix khong tradable"))
        state._record_veto_ledger(trace, make_candidate())
        self.assertEqual(state.veto_ledger["Council"]["count_24h"], 1)
        self.assertEqual(state.veto_ledger["OctoBot"]["count_24h"], 1)
        self.assertIn("1/4", state.veto_ledger["Council"]["last_reason_vi"])

    def test_accumulates_repeat_veto(self):
        state = make_state()
        for _ in range(3):
            trace = make_trace(("Council", "VETO", "Chi 2/4 phieu dong y"))
            state._record_veto_ledger(trace, make_candidate())
        self.assertEqual(state.veto_ledger["Council"]["count_24h"], 3)

    def test_keeps_latest_reason(self):
        state = make_state()
        state._record_veto_ledger(make_trace(("Council", "VETO", "Ly do cu")), make_candidate())
        state._record_veto_ledger(make_trace(("Council", "VETO", "Ly do moi nhat")), make_candidate())
        self.assertEqual(state.veto_ledger["Council"]["last_reason_vi"], "Ly do moi nhat")

    def test_resets_when_pass(self):
        state = make_state()
        state._record_veto_ledger(make_trace(("Council", "VETO", "Bi veto")), make_candidate())
        self.assertTrue(state.veto_ledger)
        state._record_veto_ledger(make_trace(("Council", "PASS", "Qua cong")), make_candidate())
        self.assertEqual(state.veto_ledger, {})


if __name__ == "__main__":
    unittest.main()
