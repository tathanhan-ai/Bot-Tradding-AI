import time
import unittest

import pandas as pd

from trading.pipeline import (
    CandidateOrder,
    MarketSnapshot,
    SevenStagePipeline,
    StageOutcome,
    apply_closed_kline,
)


def fresh_snapshot(now=None):
    now = time.time() if now is None else now
    return MarketSnapshot(
        snapshot_id="snap-1",
        symbol="BTCUSDT",
        exchange="binance",
        captured_at=now,
        price=60_000.0,
        source_times={"depth": now, "agg_trade": now, "kline_1m": now},
        bids=[[59_999.0 - n, 1.0] for n in range(20)],
        asks=[[60_001.0 + n, 1.0] for n in range(20)],
    )


def candidate():
    return CandidateOrder(
        order_id="candidate-1",
        symbol="BTCUSDT",
        direction=1,
        order_type="MARKET",
        entry_price=60_000.0,
        stop_loss=59_400.0,
        take_profit=61_200.0,
    )


class SevenStagePipelineTest(unittest.TestCase):
    def test_veto_stops_every_later_stage(self):
        calls = []

        def freqtrade(_candidate, _snapshot):
            calls.append("freqtrade")
            return StageOutcome.veto("max drawdown")

        def unexpected(name):
            def gate(_candidate, _snapshot):
                calls.append(name)
                return StageOutcome.pass_(name)
            return gate

        decision = SevenStagePipeline(
            defense_gates=[("Freqtrade", freqtrade), ("VisualHFT", unexpected("visual")), ("Jesse", unexpected("jesse"))],
            stage2_gates=[("DeterministicGuard", unexpected("guard")), ("AlphaRegime", unexpected("alpha"))],
            council=unexpected("council"),
            sizing=unexpected("sizing"),
            memory=unexpected("memory"),
        ).decide(candidate(), fresh_snapshot())

        self.assertFalse(decision.approved)
        self.assertEqual(calls, ["freqtrade"])
        self.assertEqual(decision.trace.entries[-1].stage, "Stage 1 / Freqtrade")
        self.assertEqual(decision.trace.entries[-1].verdict, "VETO")

    def test_llm_unavailable_is_traced_without_fake_approval(self):
        calls = []

        def passes(name):
            def gate(_candidate, _snapshot):
                calls.append(name)
                return StageOutcome.pass_(name)
            return gate

        def llm_unavailable(_candidate, _snapshot):
            calls.append("council")
            return StageOutcome.unavailable("9Router timed out")

        def size(order, _snapshot):
            calls.append("sizing")
            return StageOutcome.pass_("clamped", quantity=0.02, margin=120.0, leverage=3)

        decision = SevenStagePipeline(
            defense_gates=[("Freqtrade", passes("freqtrade")), ("VisualHFT", passes("visual")), ("Jesse", passes("jesse"))],
            stage2_gates=[("DeterministicGuard", passes("guard")), ("AlphaRegime", passes("alpha"))],
            council=llm_unavailable,
            sizing=size,
            memory=passes("memory"),
        ).decide(candidate(), fresh_snapshot())

        # Under AI_REQUIRED (default fail-closed), unavailable LLM vetoes entry candidate
        self.assertFalse(decision.approved)
        self.assertEqual(calls, ["freqtrade", "visual", "jesse", "guard", "alpha", "council"])
        council_entry = next(entry for entry in decision.trace.entries if entry.stage == "Stage 3 / AI Council")
        self.assertEqual(council_entry.verdict, "VETO")
        self.assertIn("9Router timed out", council_entry.reason)

        # Under DETERMINISTIC_ONLY mode, AI Council is bypassed and deterministic sizing proceeds
        calls_det = []
        def passes_det(name):
            def gate(_candidate, _snapshot):
                calls_det.append(name)
                return StageOutcome.pass_(name)
            return gate
        def size_det(order, _snapshot):
            calls_det.append("sizing")
            return StageOutcome.pass_("clamped", quantity=0.02, margin=120.0, leverage=3)

        decision_det = SevenStagePipeline(
            defense_gates=[("Freqtrade", passes_det("freqtrade")), ("VisualHFT", passes_det("visual")), ("Jesse", passes_det("jesse"))],
            stage2_gates=[("DeterministicGuard", passes_det("guard")), ("AlphaRegime", passes_det("alpha"))],
            council=llm_unavailable,
            sizing=size_det,
            memory=passes_det("memory"),
            decision_mode="DETERMINISTIC_ONLY",
        ).decide(candidate(), fresh_snapshot())

        self.assertTrue(decision_det.approved)
        self.assertEqual(calls_det, ["freqtrade", "visual", "jesse", "guard", "alpha", "sizing", "memory"])
        self.assertEqual(decision_det.candidate.quantity, 0.02)

    def test_stale_or_wrong_exchange_snapshot_cannot_open_order(self):
        calls = []
        snapshot = fresh_snapshot(now=100.0)
        snapshot.exchange = "mexc"
        snapshot.source_times["depth"] = 90.0

        decision = SevenStagePipeline(
            defense_gates=[("Freqtrade", lambda *_: calls.append("called") or StageOutcome.pass_("unexpected"))]
        ).decide(candidate(), snapshot, now=100.0)

        self.assertFalse(decision.approved)
        self.assertEqual(calls, [])
        self.assertEqual(decision.trace.entries[-1].stage, "Stage 0 / Market Snapshot")

    def test_only_closed_real_ohlcv_can_change_timeframe_data(self):
        frames = {"1m": pd.DataFrame(columns=["open", "high", "low", "close", "volume", "quote_volume"])}
        forming = {"time": 1_700_000_000, "open": 1, "high": 2, "low": 1, "close": 2, "volume": 99, "quote_volume": 198, "is_closed": False}
        closed = {**forming, "is_closed": True}

        self.assertFalse(apply_closed_kline(frames, "1m", forming))
        self.assertTrue(frames["1m"].empty)
        self.assertTrue(apply_closed_kline(frames, "1m", closed))
        row = frames["1m"].iloc[-1]
        self.assertEqual(row["volume"], 99)
        self.assertEqual(row["quote_volume"], 198)


if __name__ == "__main__":
    unittest.main()
