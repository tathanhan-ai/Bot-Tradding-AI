import unittest

from strategy.grid_planner import GridPlanner


class GridPlannerTest(unittest.TestCase):
    def test_hedge_grid_is_paired_and_passive(self):
        plan = GridPlanner.plan_hedge_grid(
            price=100, best_bid=99.9, best_ask=100.1, support=94, resistance=106,
            vwap=100, atr=1.5, regime="RANGING_SIDEWAY", recommended_strategy="TWO_WAY_RANGE",
            adx=15, hurst=.4,
        )
        self.assertEqual(plan.model, "HEDGE_GRID")
        self.assertEqual(len(plan.legs) % 2, 0)
        self.assertTrue(all(leg.entry_price < 100.1 for leg in plan.legs if leg.side == "BUY"))
        self.assertTrue(all(leg.entry_price > 99.9 for leg in plan.legs if leg.side == "SELL"))

    def test_trending_or_off_center_market_is_rejected(self):
        base = dict(price=100, best_bid=99.9, best_ask=100.1, support=94, resistance=106,
                    vwap=100, atr=1.5, regime="RANGING_SIDEWAY", recommended_strategy="TWO_WAY_RANGE", adx=15)
        self.assertEqual(GridPlanner.plan_hedge_grid(**base, hurst=.7).model, "NO_TRADE")
        self.assertEqual(GridPlanner.plan_hedge_grid(**{**base, "price": 102}, hurst=.4).model, "NO_TRADE")


if __name__ == "__main__":
    unittest.main()
