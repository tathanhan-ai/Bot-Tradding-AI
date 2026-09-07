import unittest

from config.settings import RiskConfig
from risk.dynamic_leverage import DynamicLeverageEngine
from risk.risk_manager import FuturesRiskManager


class LeverageMathTests(unittest.TestCase):
    def test_mmr_bracket_increases_with_notional(self):
        self.assertAlmostEqual(FuturesRiskManager.mmr_for_notional(100_000), 0.004)
        self.assertAlmostEqual(FuturesRiskManager.mmr_for_notional(10_000_000), 0.005)
        self.assertAlmostEqual(FuturesRiskManager.mmr_for_notional(100_000_000), 0.025)
        self.assertGreater(FuturesRiskManager.mmr_for_notional(300_000_000),
                           FuturesRiskManager.mmr_for_notional(100_000))

    def test_liq_price_includes_fee_cushion(self):
        rm = FuturesRiskManager(RiskConfig(initial_balance=5000))
        entry, lev = 79159.5, 4
        liq_plain = entry * (1.0 - 1.0 / lev + 0.004)
        liq = rm.calculate_liquidation_price(entry, 1, lev)
        # Cao hon (gan entry hon) vi cong them phi taker -> bao thu hon
        self.assertGreater(liq, liq_plain)
        liq_short = rm.calculate_liquidation_price(entry, -1, lev)
        self.assertGreater(liq_short, entry)

    def test_dynamic_engine_respects_single_hard_cap(self):
        eng = DynamicLeverageEngine(max_leverage=5)
        adv = eng.calculate_optimal_leverage(
            current_price=80000.0, atr=160.0, regime="TRENDING_BULL",
            confidence=85, horizon="SHORT_TERM",
        )
        self.assertLessEqual(adv.leverage, 5)

    def test_clearance_uses_tiered_mmr(self):
        eng = DynamicLeverageEngine(max_leverage=5)
        adv = eng.calculate_optimal_leverage(
            current_price=80000.0, atr=400.0, sl_distance_pct=0.035,
        )
        self.assertGreaterEqual(adv.liq_clearance_ratio, 3.0)
        self.assertIn("sl_distance_pct", adv.factor_breakdown)


if __name__ == "__main__":
    unittest.main()
