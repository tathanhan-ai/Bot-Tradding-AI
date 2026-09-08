import unittest

from risk.order_manager import OrderQueueManager


class OrderManagerValidationTests(unittest.TestCase):
    def test_rejects_unknown_trigger_condition(self):
        manager = OrderQueueManager()
        order, reason = manager.place_order(
            "BTCUSDT", "CONDITIONAL", "BUY", 60_000, 100, 3,
            trigger_price=60_100, trigger_condition="CROSS",
        )
        self.assertIsNone(order)
        self.assertIn("trigger condition", reason.lower())

    def test_trailing_callback_rate_matches_exchange_bounds(self):
        manager = OrderQueueManager()
        for callback in (0.0, 0.09, 10.01):
            order, reason = manager.place_order(
                "BTCUSDT", "TRAILING_STOP", "BUY", 60_000, 100, 3,
                callback_pct=callback,
            )
            self.assertIsNone(order)
            self.assertIn("callback", reason.lower())

        order, _ = manager.place_order(
            "BTCUSDT", "TRAILING_STOP", "BUY", 60_000, 100, 3,
            callback_pct=0.1,
        )
        self.assertIsNotNone(order)

    def test_update_rejects_side_conflict_without_mutating_order(self):
        manager = OrderQueueManager()
        order, _ = manager.place_order(
            "BTCUSDT", "LIMIT", "BUY", 59_000, 100, 3,
            position_side="long",
        )
        self.assertIsNotNone(order)
        updated, reason = manager.update_order(order.order_id, side="SELL")
        self.assertFalse(updated)
        self.assertIn("position side", reason.lower())
        self.assertEqual(order.side, "BUY")
        self.assertEqual(order.position_side, "LONG")

    def test_scale_ratio_sell_guard_checks_nearest_ladder_leg(self):
        manager = OrderQueueManager()
        order, reason = manager.place_order(
            "BTCUSDT", "SCALE_RATIO", "SELL", 99.9, 100, 3,
            best_bid=100.0, best_ask=100.1,
        )
        self.assertIsNone(order)
        self.assertIn("SCALE_RATIO", reason)

    def test_update_trailing_callback_rate_is_bounded(self):
        manager = OrderQueueManager()
        order, _ = manager.place_order(
            "BTCUSDT", "TRAILING_STOP", "BUY", 60_000, 100, 3,
            callback_pct=1.0,
        )
        self.assertIsNotNone(order)
        updated, reason = manager.update_order(order.order_id, callback_pct=10.01)
        self.assertFalse(updated)
        self.assertIn("callback", reason.lower())
        self.assertEqual(order.callback_pct, 1.0)


if __name__ == "__main__":
    unittest.main()
