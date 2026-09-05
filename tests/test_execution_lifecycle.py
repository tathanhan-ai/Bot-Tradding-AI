import json
import math
import unittest

from risk.order_manager import OrderQueueManager


class ExecutionLifecycleTest(unittest.TestCase):
    def test_exchange_ack_never_becomes_a_local_fill(self):
        manager = OrderQueueManager()
        order, _ = manager.place_order("BTCUSDT", "LIMIT", "BUY", 60_000, 100, 3)

        self.assertTrue(manager.mark_exchange_ack(order.order_id, "12345", "pipe-1"))
        self.assertEqual(order.status, "EXCHANGE_ACK")
        self.assertEqual(manager.match_orders(59_000), [])

    def test_oco_sibling_is_cancelled_after_confirmed_fill(self):
        manager = OrderQueueManager()
        first, _ = manager.place_order("BTCUSDT", "LIMIT", "BUY", 59_000, 100, 3, execution_group="oco-1")
        second, _ = manager.place_order("BTCUSDT", "LIMIT", "SELL", 61_000, 100, 3, execution_group="oco-1")
        manager.mark_exchange_ack(first.order_id, "1", "pipe-first")
        manager.mark_exchange_ack(second.order_id, "2", "pipe-second")

        cancelled = manager.mark_exchange_fill(first.order_id)

        self.assertEqual(first.status, "FILLED")
        self.assertEqual([order.order_id for order in cancelled], [second.order_id])
        self.assertEqual(second.status, "CANCEL_REQUESTED")

    def test_exchange_ack_remains_pending_for_duplicate_entry_guard(self):
        manager = OrderQueueManager()
        order, _ = manager.place_order("BTCUSDT", "LIMIT", "BUY", 59_000, 100, 3)

        manager.mark_exchange_ack(order.order_id, "123", "pipe-ack")

        self.assertEqual(manager.pending_orders, [order])
        self.assertEqual(manager.get_pending_orders()[0]["status"], "EXCHANGE_ACK")

    def test_twap_has_unique_due_children_and_no_tick_acceleration(self):
        manager = OrderQueueManager()
        first, _ = manager.place_order("BTCUSDT", "TWAP", "BUY", 60000, 100, 3, quantity=.005,
                                       twap_slices=5, twap_interval_seconds=10, due_at=1000,
                                       client_order_id="twap-1", candidate_payload={"source": "pipeline"})
        self.assertEqual(first.order_type, "TWAP_SLICE")
        self.assertEqual(len(manager.orders), 5)
        self.assertEqual([order.due_at for order in manager.orders], [1000, 1010, 1020, 1030, 1040])
        self.assertEqual(len({order.client_order_id for order in manager.orders}), 5)
        self.assertEqual(len(manager.match_orders(60000, now=1000)), 1)
        for _ in range(1000):
            self.assertEqual(manager.match_orders(60000, now=1001), [])
        self.assertEqual(len(manager.match_orders(60000, now=1020)), 2)
        self.assertEqual(len(manager.match_orders(60000, now=1040)), 2)
        self.assertLessEqual(math.fsum(order.units for order in manager.orders), .005)
        again, _ = manager.place_order("BTCUSDT", "TWAP", "BUY", 60000, 100, 3, quantity=.005, client_order_id="twap-1")
        self.assertIs(again, first)
        self.assertEqual(len(manager.orders), 5)

    def test_scale_quantity_budget_and_non_oco_groups(self):
        for side in ("BUY", "SELL"):
            manager = OrderQueueManager()
            first, _ = manager.place_order("BTCUSDT", "SCALE_RATIO", side, 60000, 100, 3,
                                           quantity=.01234567, execution_group="scale-1", client_order_id="scale-client")
            self.assertLessEqual(math.fsum(order.units for order in manager.orders), .01234567)
            self.assertAlmostEqual(math.fsum(order.units for order in manager.orders), .01234567)
            self.assertEqual(manager.mark_exchange_fill(first.order_id), [])
            self.assertTrue(all(order.status == "PENDING" for order in manager.orders[1:]))
        manager = OrderQueueManager()
        first, _ = manager.place_order("BTCUSDT", "LIMIT", "BUY", 59000, 100, 3, execution_group="grid-1", group_type="GRID")
        other, _ = manager.place_order("BTCUSDT", "LIMIT", "SELL", 61000, 100, 3, execution_group="grid-1", group_type="GRID")
        self.assertEqual(manager.mark_exchange_fill(first.order_id), [])
        self.assertEqual(other.status, "PENDING")

    def test_partial_oco_fill_cancels_only_opposite_branch(self):
        manager = OrderQueueManager()
        orders = [manager.place_order("BTCUSDT", "LIMIT", side, 60000, 100, 3, quantity=.01,
                                      execution_group="group-1", group_type="OCO")[0] for side in ("BUY", "BUY", "SELL")]
        self.assertTrue(manager.record_exchange_update(orders[0].order_id, {"orderId": "123", "status": "PARTIALLY_FILLED",
            "executedQty": ".001", "avgPrice": "60000"}))
        self.assertEqual(orders[0].status, "EXCHANGE_ACK")
        self.assertEqual(orders[1].status, "PENDING")
        self.assertEqual(orders[2].status, "CANCEL_REQUESTED")

    def test_restart_preserves_idempotency_unapplied_fills_and_unknown_submits(self):
        manager = OrderQueueManager()
        first, _ = manager.place_order("BTCUSDT", "LIMIT", "BUY", 60000, 100, 3, quantity=.01,
                                       client_order_id="persist-1", candidate_payload={"order_id": "persist-1"}, decision_trace={"entries": ["PASS"]})
        unknown, _ = manager.place_order("BTCUSDT", "LIMIT", "BUY", 59000, 100, 3, client_order_id="persist-2")
        manager.mark_submit_pending(unknown.order_id)
        self.assertTrue(manager.record_exchange_update(first.order_id, {"orderId": "123", "status": "PARTIALLY_FILLED",
            "executedQty": ".004", "avgPrice": "60000"}))
        self.assertEqual(manager.pending_fill(first.order_id), (.004, 60000))
        manager.mark_fill_applied(first.order_id)
        self.assertTrue(manager.record_exchange_update(first.order_id, {"orderId": "123", "status": "FILLED",
            "executedQty": ".01", "avgPrice": "60600"}))
        restored = OrderQueueManager()
        restored.restore_state(json.loads(json.dumps(manager.export_state())))
        self.assertEqual(restored.orders[1].status, "SUBMIT_UNKNOWN")
        self.assertEqual(restored.orders[0].candidate_payload, {"order_id": "persist-1"})
        self.assertEqual(restored.orders[0].decision_trace, {"entries": ["PASS"]})
        qty, price = restored.pending_fill(first.order_id)
        self.assertAlmostEqual(qty, .006)
        self.assertAlmostEqual(price, 61000)
        self.assertEqual(restored.match_orders(58000), [])
        self.assertFalse(restored.force_execute_order(unknown.order_id, 58000))
        self.assertFalse(restored.cancel_order(unknown.order_id))
        self.assertFalse(restored.update_order(unknown.order_id, price=58000)[0])
        restored.mark_fill_applied(first.order_id)
        self.assertEqual(restored.pending_fill(first.order_id), (0, 0))
        restored.record_exchange_update(first.order_id, {"orderId": "123", "status": "FILLED", "executedQty": ".01", "avgPrice": "60600"})
        self.assertEqual(restored.pending_fill(first.order_id), (0, 0))
        self.assertFalse(restored.record_exchange_update(first.order_id, {"orderId": "123", "status": "NEW", "executedQty": 0}))
        new, _ = restored.place_order("BTCUSDT", "LIMIT", "BUY", 60000, 100, 3, client_order_id="persist-3")
        self.assertGreater(new.order_id, unknown.order_id)

    def test_unknown_submit_remains_pending_and_cannot_match(self):
        manager = OrderQueueManager()
        order, _ = manager.place_order("BTCUSDT", "LIMIT", "BUY", 60000, 100, 3, client_order_id="unknown-1")
        self.assertTrue(manager.mark_submit_pending(order.order_id))
        self.assertFalse(manager.mark_submit_pending(order.order_id))
        self.assertTrue(manager.mark_submit_unknown(order.order_id))
        self.assertEqual(manager.pending_orders, [order])
        self.assertEqual(manager.due_orders(), [])
        self.assertEqual(manager.match_orders(59000), [])
        self.assertTrue(manager.mark_exchange_ack(order.order_id, "123"))
        self.assertEqual(order.status, "EXCHANGE_ACK")

    def test_restore_failure_is_atomic(self):
        manager = OrderQueueManager()
        order, _ = manager.place_order("BTCUSDT", "LIMIT", "BUY", 60000, 100, 3)
        payload = manager.export_state()
        payload["orders"].append(payload["orders"][0])
        with self.assertRaises(ValueError):
            manager.restore_state(payload)
        self.assertEqual(manager.orders, [order])


if __name__ == "__main__":
    unittest.main()
