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

    def test_acceptable_sideway_regimes_and_strategies(self):
        base = dict(price=100, best_bid=99.9, best_ask=100.1, support=94, resistance=106,
                    vwap=100, atr=1.5, adx=18, hurst=.4)
        for regime in ("RANGING_SIDEWAY", "SIDEWAY_GRID", "CHOPPY", "NEUTRAL", "EQUILIBRIUM_FAIR"):
            for strat in ("TWO_WAY_RANGE", "GRID_BOT", "SIDEWAY_GRID", "MEAN_REVERSION_GRID"):
                plan = GridPlanner.plan_hedge_grid(**base, regime=regime, recommended_strategy=strat)
                self.assertEqual(plan.model, "HEDGE_GRID", f"Failed for regime={regime}, strat={strat}")

    def test_vwap_tolerance_up_to_075_atr(self):
        # ATR = 1.5, 0.75 * ATR = 1.125
        base = dict(support=94, resistance=106,
                    vwap=100, atr=1.5, regime="RANGING_SIDEWAY", recommended_strategy="TWO_WAY_RANGE", adx=15, hurst=.4)
        # price = 100.8 -> abs(price - vwap) = 0.8 <= 1.125 -> Acceptable
        plan = GridPlanner.plan_hedge_grid(**base, price=100.8, best_bid=100.7, best_ask=100.9)
        self.assertEqual(plan.model, "HEDGE_GRID")
        # price = 101.5 -> abs(price - vwap) = 1.5 > 1.125 -> Rejected
        plan_far = GridPlanner.plan_hedge_grid(**base, price=101.5, best_bid=101.4, best_ask=101.6)
        self.assertEqual(plan_far.model, "NO_TRADE")


if __name__ == "__main__":
    unittest.main()
