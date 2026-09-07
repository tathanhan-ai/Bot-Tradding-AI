import unittest

from strategy.tactical_lessons import HurstRegimeFilter, SMCTrendAlignment, FeeAwareEntry, TacticalLessonsEngine


class TacticalLessonsTests(unittest.TestCase):
    def test_hurst_blocks_momentum_when_mean_reverting(self):
        f = HurstRegimeFilter()
        # Bai hoc that: Hurst 0.22-0.34 nhung bot van danh directional -> thua
        v = f.evaluate(0.30, "MARKET")
        self.assertFalse(v.allowed)
        v2 = f.evaluate(0.30, "POST_ONLY")
        self.assertTrue(v2.allowed)

    def test_smc_blocks_counter_trend(self):
        a = SMCTrendAlignment()
        # Martingale 14-18: LONG trong BEARISH_TREND (khong sweep moi) -> cam
        v = a.evaluate("BEARISH_TREND", 1)
        self.assertFalse(v.allowed)
        self.assertEqual(v.direction_bias, -1)
        v2 = a.evaluate("BEARISH_TREND", -1)
        self.assertTrue(v2.allowed)

    def test_smc_allows_reversal_on_fresh_sweep(self):
        a = SMCTrendAlignment()
        # Sweep dao chieu moi (AI Brain CASE 1): cho LONG tham do du structure cu van bearish
        v = a.evaluate("BEARISH_TREND", 1, smc_bias="BULLISH_REVERSAL", has_fresh_sweep=True)
        self.assertTrue(v.allowed)
        self.assertLess(v.penalty, 0.5)

    def test_fee_blocks_dust_trades(self):
        fa = FeeAwareEntry(taker_fee_rate=0.0005, max_fee_ratio=0.35)
        # Lenh 1077674: lai gop +$0.11, phi $0.11 -> chan
        v = fa.evaluate(0.11, 220.0)
        self.assertFalse(v.allowed)

    def test_engine_combines_lessons(self):
        eng = TacticalLessonsEngine()
        res = eng.evaluate(0.30, "BEARISH_TREND", 1, "MARKET", 50.0, 5000.0)
        self.assertFalse(res["allowed"])
        self.assertIn("HURST_FILTER", res["blocked_lessons"])
        self.assertIn("SMC_ALIGN", res["blocked_lessons"])


if __name__ == "__main__":
    unittest.main()
