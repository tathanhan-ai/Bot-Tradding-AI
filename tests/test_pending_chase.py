"""Lenh cho bam gia khi truot: doi ve book thay vi huy cho dat lai.

Cu: lech >0.8% la huy + return som (khong dat lai) -> gia truot la
mat lenh. Nay: doi ve best bid/ask (giua Maker) nhung khong qua 0.3%
so voi gia goc (chong mua duoi), giu lenh song de gui san.
"""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock


def chase(order, current_price, best_bid, best_ask):
    """Mo phong nhanh bam gia trong evaluate_ensemble_automated_decision."""
    if order.get("group_type") == "GRID":
        return order["price"], "skip-grid"
    if order.get("order_type") not in ("POST_ONLY", "LIMIT", "SCALE_RATIO"):
        return order["price"], "skip-type"
    if order.get("status") not in ("PENDING", "ACTIVE"):
        return order["price"], "skip-status"
    if order.get("exchange_order_id") or order.get("exchange_status"):
        return order["price"], "skip-sent"
    drift = abs(current_price - order["price"]) / max(1.0, current_price)
    if drift <= 0.008:
        return order["price"], "fresh"
    anchor = best_bid if order.get("side") == "BUY" else best_ask
    if anchor <= 0:
        return order["price"], "no-book"
    cap = order["price"] * (1.003 if order.get("side") == "BUY" else 0.997)
    floor = order["price"] * (0.997 if order.get("side") == "BUY" else 1.003)
    new_px = round(min(max(anchor, min(cap, floor)), max(cap, floor)), 1)
    if new_px <= 0 or abs(new_px - order["price"]) / max(1.0, order["price"]) < 0.0005:
        return order["price"], "tiny"
    return new_px, "chased"


class ChaseTests(unittest.TestCase):
    def test_drifted_buy_chases_bid(self):
        px, why = chase({"order_type": "POST_ONLY", "side": "BUY", "status": "PENDING",
                         "price": 77000.0, "group_type": ""}, 79000.0, 78999.0, 79001.0)
        self.assertEqual(why, "chased")
        # Bien 0.3%: khong vuot 77000*1.003=77231 du book o 78999.
        self.assertAlmostEqual(px, 77231.0, places=1)

    def test_fresh_order_untouched(self):
        px, why = chase({"order_type": "POST_ONLY", "side": "BUY", "status": "PENDING",
                         "price": 78990.0, "group_type": ""}, 79000.0, 78999.0, 79001.0)
        self.assertEqual(why, "fresh")
        self.assertEqual(px, 78990.0)

    def test_grid_skipped(self):
        px, why = chase({"order_type": "GRID", "side": "BUY", "status": "PENDING",
                         "price": 70000.0, "group_type": "GRID"}, 79000.0, 78999.0, 79001.0)
        self.assertEqual(why, "skip-grid")

    def test_sent_order_skipped(self):
        px, why = chase({"order_type": "POST_ONLY", "side": "BUY", "status": "ACTIVE",
                         "price": 77000.0, "group_type": "", "exchange_order_id": "123"},
                        79000.0, 78999.0, 79001.0)
        self.assertEqual(why, "skip-sent")

    def test_sell_chases_ask_down(self):
        px, why = chase({"order_type": "LIMIT", "side": "SELL", "status": "PENDING",
                         "price": 80000.0, "group_type": ""}, 79000.0, 78999.0, 79001.0)
        self.assertEqual(why, "chased")
        # Bien 0.3%: khong duoi 80000*0.997=79760 du book o 79001.
        self.assertAlmostEqual(px, 79760.0, places=1)


if __name__ == "__main__":
    unittest.main()
