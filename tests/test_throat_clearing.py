"""Thoat that co chai 1D-vs-15m: song 1D TANG cam SHORT, SMC 15m BEARISH
cam LONG -> khong huong nao qua duoc.

Quy tac moi (giam sat chat, chi mo cua Maker probation):
1. SMC khung nho nguoc nhung thuan song 1D + lenh Maker -> PASS probation
   (co smc_wave_override), Taker van veto cung.
2. Weak-combo 15m-LONG (PF qua khu <1) + Alpha am nhe + thuan song 1D +
   lenh Maker + OctoBot khong nguoc -> PASS probation (co
   weak_combo_probation). OctoBot nguoc huong van veto cung.
"""
import unittest


def smc_wave_override_decision(blocked, wave_1d, direction, order_type):
    """Mo phong nhanh moi trong tactical_lessons_gate (ui/server.py)."""
    wave_ok = wave_1d != 0 and direction == wave_1d
    maker = order_type in ("POST_ONLY", "LIMIT", "SCALE_RATIO")
    if blocked == ["SMC_ALIGN"] and wave_ok and maker:
        return ("PASS", True)
    return ("VETO", False)


def weak_combo_decision(octo_against, wave_1d, direction, order_kind):
    """Mo phong nhanh moi trong alpha_regime_gate (ui/server.py)."""
    maker = order_kind in ("POST_ONLY", "LIMIT", "SCALE_RATIO")
    if wave_1d != 0 and direction == wave_1d and maker and not octo_against:
        return ("PASS", True)
    return ("VETO", False)


class ThroatClearingTests(unittest.TestCase):
    def test_long_thuan_song_1d_qua_smc_probation(self):
        verdict, flag = smc_wave_override_decision(["SMC_ALIGN"], 1, 1, "POST_ONLY")
        self.assertEqual(verdict, "PASS")
        self.assertTrue(flag)

    def test_taker_khong_duoc_huong(self):
        verdict, _ = smc_wave_override_decision(["SMC_ALIGN"], 1, 1, "MARKET")
        self.assertEqual(verdict, "VETO")

    def test_nguoc_song_1d_van_veto(self):
        verdict, _ = smc_wave_override_decision(["SMC_ALIGN"], 1, -1, "POST_ONLY")
        self.assertEqual(verdict, "VETO")

    def test_smc_va_fee_cung_chan_thi_veto(self):
        verdict, _ = smc_wave_override_decision(["SMC_ALIGN", "FEE_AWARE"], 1, 1, "POST_ONLY")
        self.assertEqual(verdict, "VETO")

    def test_weak_combo_thuan_song_qua_probation(self):
        verdict, flag = weak_combo_decision(False, 1, 1, "POST_ONLY")
        self.assertEqual(verdict, "PASS")
        self.assertTrue(flag)

    def test_octo_nguoc_huong_van_veto(self):
        verdict, _ = weak_combo_decision(True, 1, 1, "POST_ONLY")
        self.assertEqual(verdict, "VETO")


if __name__ == "__main__":
    unittest.main()
