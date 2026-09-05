from types import SimpleNamespace
import unittest

from risk.ai_order_researcher import AIOrderResearcher
from strategy.hummingbot_inventory_skew import HummingbotInventorySkewEngine


class InventorySkewIntegrationTest(unittest.TestCase):
    def test_empty_indicator_value_is_safe_before_first_analysis(self):
        status = HummingbotInventorySkewEngine().calculate_reservation_price(100.0, None, None)

        self.assertEqual(status.current_position_side, "FLAT")
        self.assertEqual(status.volatility_sigma, 0.0)

    def test_skew_changes_entry_only_when_scaling_the_same_inventory_side(self):
        researcher = AIOrderResearcher()
        position = {"direction": 1, "margin": 500.0}
        skew = HummingbotInventorySkewEngine().calculate_reservation_price(100.0, position, 10.0, 1_000.0)
        result = researcher.research(
            current_price=100.0, best_bid=99.9, best_ask=100.1, spread=0.2,
            indicators={"atr": 10.0, "rsi": 55.0, "adx": 20.0},
            ai_verdict=SimpleNamespace(regime="RANGING_SIDEWAY", garman_klass_vol=0.01, mtf_radar={}),
            ensemble_result=SimpleNamespace(consensus_score=40.0, confidence=80),
            ai_cro=SimpleNamespace(current_drawdown_pct=0.0), current_balance=1_000.0,
            current_position=position, inventory_skew=skew,
        )

        self.assertEqual(result.recommended_side, "BUY")
        self.assertEqual(result.optimal_price, skew.reservation_price)


if __name__ == "__main__":
    unittest.main()
