"""P1+P3+P4+P5: test cho cum chong ket lenh live (ha cap co kiem soat).

- P1 wave_alignment: parser chi thi song + luat VETO/probation/PASS.
- P4 council context: _role_contexts phai mang user_instruction + wave_alignment.
- P5 counter-proposal: ke thua song/Hurst/probation + uu tien Maker khi Hurst thap.
"""
import unittest

import pandas as pd

from strategy.wave_alignment import (
    compute_wave_alignment,
    evaluate_wave_gate,
    parse_wave_instruction,
)
from strategy.vibe_swarm_council import VibeSwarmCouncil
from risk.ai_order_researcher import AIOrderResearcher
from trading.pipeline import CandidateOrder


def make_frame(close_start, close_end, n=60):
    closes = [close_start + (close_end - close_start) * i / max(1, n - 1) for i in range(n)]
    return pd.DataFrame([
        {"open": c, "high": c * 1.001, "low": c * 0.999, "close": c, "volume": 10.0}
        for c in closes
    ])


class WaveInstructionTests(unittest.TestCase):
    def test_parse_real_user_instruction(self):
        instr = parse_wave_instruction("Uu tien song 1D, cho phep song 15m khi cung chieu 1D")
        self.assertTrue(instr["has_instruction"])
        self.assertTrue(instr["prefer_1d"])
        self.assertTrue(instr["allow_15m_aligned"])

    def test_parse_empty_has_no_instruction(self):
        instr = parse_wave_instruction("")
        self.assertFalse(instr["has_instruction"])

    def test_parse_unrelated_text_has_no_instruction(self):
        instr = parse_wave_instruction("Danh scalping nhanh, cat lo chat")
        self.assertFalse(instr["has_instruction"])

    def test_counter_1d_is_vetoed(self):
        instr = {"has_instruction": True, "prefer_1d": True,
                 "allow_15m_aligned": True, "block_counter_1d": True}
        res = evaluate_wave_gate(1, instr, {"wave_1d": -1, "wave_15m": -1},
                                 "POST_ONLY", "auto", {})
        self.assertEqual(res["verdict"], "VETO")
        self.assertIn("1D", res["reason"])

    def test_misaligned_15m_gets_probation(self):
        instr = {"has_instruction": True, "prefer_1d": True,
                 "allow_15m_aligned": True, "block_counter_1d": False}
        res = evaluate_wave_gate(1, instr, {"wave_1d": 1, "wave_15m": -1},
                                 "POST_ONLY", "auto", {})
        self.assertEqual(res["verdict"], "PASS")
        self.assertTrue(res["updates"].get("probation"))
        self.assertTrue(res["updates"].get("wave_misaligned"))

    def test_aligned_passes(self):
        instr = {"has_instruction": True, "prefer_1d": True,
                 "allow_15m_aligned": True, "block_counter_1d": False}
        res = evaluate_wave_gate(1, instr, {"wave_1d": 1, "wave_15m": 1},
                                 "POST_ONLY", "auto", {})
        self.assertEqual(res["verdict"], "PASS")
        self.assertTrue(res["updates"].get("wave_aligned"))

    def test_grid_is_exempt(self):
        instr = {"has_instruction": True, "prefer_1d": True,
                 "allow_15m_aligned": False, "block_counter_1d": True}
        res = evaluate_wave_gate(0, instr, {"wave_1d": -1, "wave_15m": -1},
                                 "GRID", "auto-grid", {})
        self.assertEqual(res["verdict"], "PASS")

    def test_compute_alignment_from_frames(self):
        frames = {"15m": make_frame(90000, 91000), "1h": make_frame(85000, 91000, n=120)}
        align = compute_wave_alignment(frames)
        self.assertEqual(align["wave_15m"], 1)
        # 1h tang dan -> resample 1d cung tang (neu du lieu).
        self.assertIn(align["wave_1d"], (-1, 0, 1))


class CouncilWaveContextTests(unittest.TestCase):
    def test_role_contexts_carry_instruction_and_wave(self):
        council = VibeSwarmCouncil(enabled=True)
        ctx = council._role_contexts(
            90000.0, {"rsi": 50.0}, None, None, None, None, None, None, None,
            user_instruction="Uu tien song 1D",
            wave_alignment={"wave_1d": 1, "wave_15m": -1},
        )
        for role in ("macro", "quant", "risk", "execution"):
            self.assertEqual(ctx[role]["user_instruction"], "Uu tien song 1D")
            self.assertEqual(ctx[role]["wave_alignment"], {"wave_1d": 1, "wave_15m": -1})
        self.assertIn("lech song", ctx["macro"]["wave_guidance"])


class CounterProposalWaveTests(unittest.TestCase):
    def setUp(self):
        self.researcher = AIOrderResearcher()

    def _candidate(self, **kw):
        base = dict(order_id="test-wave-1", symbol="BTCUSDT", direction=1,
                    order_type="MARKET", entry_price=90000.0, stop_loss=89500.0,
                    take_profit=91500.0, leverage=5, quantity=0.05, margin=900.0)
        base.update(kw)
        cand = CandidateOrder(**base)
        cand.metadata["wave_alignment"] = {"wave_1d": 1, "wave_15m": -1}
        cand.metadata["wave_misaligned"] = True
        return cand

    def test_low_hurst_macro_veto_prefers_maker_probation(self):
        cand = self._candidate()
        alt = self.researcher.build_adaptive_counter_proposal(
            original_candidate=cand,
            veto_stage="Stage 2 / OctoBot Matrix & Alpha Zoo",
            veto_reason="Macro Officer: Counter-trend macro downtrend",
            current_price=90000.0, best_bid=89995.0, best_ask=90000.0,
            indicators={"atr": 120.0, "rsi": 40.0, "hurst": 0.32},
            current_balance=5000.0)
        self.assertIsNotNone(alt)
        self.assertEqual(alt.order_type, "POST_ONLY")
        self.assertTrue(alt.metadata.get("probation"))
        self.assertEqual(alt.metadata.get("wave_alignment"), {"wave_1d": 1, "wave_15m": -1})

    def test_inherits_probation_flags(self):
        cand = self._candidate()
        cand.metadata["hurst_downgraded"] = True
        alt = self.researcher.build_adaptive_counter_proposal(
            original_candidate=cand,
            veto_stage="Stage 4 / Carver + Risk",
            veto_reason="Risk: margin too high",
            current_price=90000.0, best_bid=89995.0, best_ask=90000.0,
            indicators={"atr": 120.0, "hurst": 0.50},
            current_balance=5000.0)
        self.assertIsNotNone(alt)
        self.assertTrue(alt.metadata.get("hurst_downgraded"))


if __name__ == "__main__":
    unittest.main()
