import math
from types import SimpleNamespace
import unittest

import pandas as pd

from risk.ai_order_researcher import AIOrderResearcher
from strategy.quant_skills_brain import QuantSkillsBrain


class QuantPipelineContractsTest(unittest.TestCase):
    def bars(self, close=105.0):
        return pd.DataFrame({"open": [100.0] * 30, "high": [110.0] * 30,
                             "low": [90.0] * 30, "close": [close] * 30})

    def research(self, momentum="BALANCED", score=60.0):
        return AIOrderResearcher().research(
            current_price=60_000.0, best_bid=59_999.9, best_ask=60_000.1, spread=0.2,
            indicators={"atr": 150.0, "rsi": 50.0, "adx": 30.0},
            ai_verdict=SimpleNamespace(regime="TRENDING_BULL", garman_klass_vol=20.0, mtf_radar={}),
            ensemble_result=SimpleNamespace(consensus_score=score, confidence=80),
            ai_cro=SimpleNamespace(current_drawdown_pct=0.0), current_balance=5_000.0,
            order_flow_verdict=SimpleNamespace(delta_momentum=momentum, absorption_divergence=True),
            effective_leverage=5,
        )

    def test_gk_has_equal_volatility_for_opposite_log_returns(self):
        positive = QuantSkillsBrain.calculate_garman_klass_volatility(self.bars())
        negative = QuantSkillsBrain.calculate_garman_klass_volatility(self.bars(100.0 ** 2 / 105.0))
        self.assertAlmostEqual(positive.garman_klass_bar, negative.garman_klass_bar)

    def test_gk_annualization_matches_timeframe_and_infers_datetime_index(self):
        frame = self.bars()
        one_minute = QuantSkillsBrain.calculate_garman_klass_volatility(frame, timeframe="1m")
        quarter_hour = QuantSkillsBrain.calculate_garman_klass_volatility(frame, timeframe="15m")
        self.assertAlmostEqual(one_minute.garman_klass_annualized / quarter_hour.garman_klass_annualized,
                               math.sqrt(15), places=3)
        frame.index = pd.date_range("2026-01-01", periods=len(frame), freq="h")
        inferred = QuantSkillsBrain.calculate_garman_klass_volatility(frame)
        explicit = QuantSkillsBrain.calculate_garman_klass_volatility(frame, timeframe="1h")
        self.assertEqual(inferred.garman_klass_annualized, explicit.garman_klass_annualized)

    def test_annual_percent_becomes_daily_fraction_before_carver(self):
        result = self.research()
        expected_daily_vol = 20.0 / 100.0 / math.sqrt(365)
        self.assertAlmostEqual(result.carver_output["instrument_value_vol"],
                               result.optimal_price * expected_daily_vol, places=2)
        self.assertEqual(result.carver_action, "BUY")
        self.assertGreater(result.carver_contracts, 0.01)
        self.assertLess(result.carver_contracts, 1.0)

    def test_real_cvd_enums_change_carver_forecast_and_side(self):
        for momentum, direction, side in (
            ("STRONG_BUY_PRESSURE", 1, "BUY"), ("STRONG_SELL_PRESSURE", -1, "SELL"),
            ("ABSORPTION_BUY", 1, "BUY"), ("ABSORPTION_SELL", -1, "SELL"),
        ):
            with self.subTest(momentum=momentum):
                result = self.research(momentum, score=0.0)
                self.assertEqual(result.recommended_side, side)
                self.assertGreater(result.carver_output["scaled_forecast"] * direction, 0)


if __name__ == "__main__":
    unittest.main()
