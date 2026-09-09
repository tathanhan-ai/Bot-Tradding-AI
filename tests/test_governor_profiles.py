import unittest

from risk.monthly_target_governor import MonthlyTargetGovernor


class GovernorProfileTests(unittest.TestCase):
    def test_growth_profile_is_default(self):
        gov = MonthlyTargetGovernor(storage=None)
        self.assertEqual(gov.profile_name, "growth")
        self.assertIn("growth", MonthlyTargetGovernor.PROFILES)
        self.assertEqual(len(MonthlyTargetGovernor.PROFILES), 4)

    def test_switch_profile_updates_base_target(self):
        gov = MonthlyTargetGovernor(storage=None)
        gov.update_config(profile="sustainable")
        self.assertEqual(gov.profile_name, "sustainable")
        self.assertAlmostEqual(gov.base_target_pct, 4.0)
        gov.update_config(profile="aggressive")
        self.assertAlmostEqual(gov.base_target_pct, 25.0)

    def test_invalid_profile_keeps_current(self):
        gov = MonthlyTargetGovernor(storage=None)
        gov.update_config(profile="growth")
        gov.update_config(profile="nope")
        self.assertEqual(gov.profile_name, "growth")

    def test_evaluate_uses_profile_numbers(self):
        gov = MonthlyTargetGovernor(storage=None)
        gov.month_start_balance = 5000.0
        gov.carried_deficit_pct = 0.0
        gov.update_config(profile="sustainable")
        st = gov.evaluate(5000.0, [])
        self.assertEqual(st.regime, "ON_TRACK")
        self.assertEqual(st.max_leverage_cap, 5)
        self.assertEqual(st.min_ai_confidence, 55)
        self.assertAlmostEqual(st.min_risk_reward_ratio, 1.5)
        self.assertEqual(st.profile_name, "sustainable")

    def test_size_multiplier_never_exceeds_one(self):
        gov = MonthlyTargetGovernor(storage=None)
        gov.month_start_balance = 5000.0
        for profile in ("sustainable", "balanced", "growth", "aggressive"):
            gov.update_config(profile=profile)
            for regime_setup in (0.0, 5.0):
                gov.carried_deficit_pct = regime_setup
                st = gov.evaluate(5000.0, [])
                self.assertLessEqual(st.size_multiplier, 1.0)

    def test_custom_target_survives_profile_reload(self):
        # Loi that: user luu 20% xong mo lai nhay ve 10% (profile base ghi de).
        gov = MonthlyTargetGovernor(storage=None)
        gov.update_config(20.0, True, True, profile="growth")
        self.assertAlmostEqual(gov.base_target_pct, 20.0)
        gov2 = MonthlyTargetGovernor(storage=None)
        gov2.update_config(20.0, True, True, profile="growth")
        self.assertAlmostEqual(gov2.base_target_pct, 20.0)
        st = gov2.evaluate(5000.0, [])
        self.assertAlmostEqual(st.base_target_pct, 20.0)


if __name__ == "__main__":
    unittest.main()
