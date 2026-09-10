"""Dieu phoi don bay + chot lai theo dinh (peak-lock).

- Don bay theo ATR: thap -> cao (toi 10x), cao -> thap (3x), luon giu
  thanh ly cach SL >= 3x.
- Chot 3 tang theo % dinh (khong doi TP goc): +0.30% chot 30% + SL ve
  entry; +0.50% chot them 30% + SL +0.25%; tut 0.25% tu dinh thi chot not.
"""
import unittest

from strategy.peak_lock import (
    GIVEBACK_PCT,
    LOCK_1_PCT,
    LOCK_2_PCT,
    leverage_for_atr,
    leverage_with_liq_floor,
    liq_distance_pct,
    peak_action,
    peak_levels,
)


class LeverageCoordTests(unittest.TestCase):
    def test_low_atr_high_leverage(self):
        self.assertEqual(leverage_for_atr(0.002), 10)
        self.assertEqual(leverage_for_atr(0.005), 8)
        self.assertEqual(leverage_for_atr(0.009), 6)
        self.assertEqual(leverage_for_atr(0.015), 4)
        self.assertEqual(leverage_for_atr(0.030), 3)

    def test_liq_floor_keeps_clearance(self):
        # SL 0.5%: thanh ly phai cach >= 1.5%. 10x cho 9.1% -> giu 10x.
        self.assertEqual(leverage_with_liq_floor(0.002, 0.005), 10)
        # SL 4%: can thanh ly >= 12%. 10x chi 9.1% -> ha den khi du.
        lev = leverage_with_liq_floor(0.002, 0.04)
        self.assertLess(lev, 10)
        self.assertGreaterEqual(liq_distance_pct(lev), 0.04 * 3.0 * 100.0 - 0.5)

    def test_liq_distance_math(self):
        # 5x: (1/5 - 0.004 - 0.0005)*100 = 19.55%
        self.assertAlmostEqual(liq_distance_pct(5), 19.55, places=1)


class PeakLockTests(unittest.TestCase):
    def pos(self, **kw):
        base = {"entry_price": 78000.0, "direction": 1,
                "peak_flags": {}, "peak_lock_price": 78000.0}
        base.update(kw)
        return base

    def test_below_lock1_holds(self):
        action, _ = peak_action(self.pos(), 78000.0 * (1 + LOCK_1_PCT - 0.0005))
        self.assertEqual(action, "hold")

    def test_lock1_at_030(self):
        action, info = peak_action(self.pos(), 78000.0 * (1 + LOCK_1_PCT))
        self.assertEqual(action, "lock1")
        self.assertAlmostEqual(info["ratio"], 0.30)

    def test_lock1_fires_once(self):
        p = self.pos(peak_flags={"lock1": True}, peak_lock_price=78250.0)
        action, _ = peak_action(p, 78250.0)
        self.assertEqual(action, "hold")

    def test_lock2_at_050(self):
        p = self.pos(peak_flags={"lock1": True}, peak_lock_price=78300.0)
        action, info = peak_action(p, 78000.0 * (1 + LOCK_2_PCT))
        self.assertEqual(action, "lock2")
        self.assertAlmostEqual(info["ratio"], 0.30)

    def test_giveback_exits_rest(self):
        peak = 78000.0 * (1 + LOCK_2_PCT)
        p = self.pos(peak_flags={"lock1": True, "lock2": True}, peak_lock_price=peak)
        action, info = peak_action(p, peak * (1 - GIVEBACK_PCT - 0.0005))
        self.assertEqual(action, "giveback_exit")

    def test_small_pullback_holds(self):
        peak = 78000.0 * (1 + LOCK_2_PCT)
        p = self.pos(peak_flags={"lock1": True, "lock2": True}, peak_lock_price=peak)
        action, _ = peak_action(p, peak * (1 - GIVEBACK_PCT + 0.001))
        self.assertEqual(action, "hold")

    def test_short_mirrors_long(self):
        p = {"entry_price": 78000.0, "direction": -1, "peak_flags": {}, "peak_lock_price": 78000.0}
        action, _ = peak_action(p, 78000.0 * (1 - LOCK_1_PCT))
        self.assertEqual(action, "lock1")

    def test_levels_math(self):
        lv = peak_levels(78000.0, 1)
        self.assertAlmostEqual(lv["lock1"], 78000.0 * 1.003, places=1)
        self.assertAlmostEqual(lv["lock2_sl"], 78000.0 * 1.0025, places=1)


if __name__ == "__main__":
    unittest.main()
