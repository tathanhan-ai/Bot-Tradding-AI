import unittest

from risk.antigravity_risk_officer import AntigravityRiskOfficer


class RiskFeedbackTest(unittest.TestCase):
    def test_cro_exposes_current_drawdown_for_carver_overlay(self):
        cro = AntigravityRiskOfficer(initial_balance=1_000.0)

        cro.evaluate_risk_profile(
            current_balance=900.0,
            market_regime="RANGING_SIDEWAY",
            confidence=60,
            atr_pct=1.0,
        )

        self.assertAlmostEqual(cro.current_drawdown_pct, 0.10)


if __name__ == "__main__":
    unittest.main()
