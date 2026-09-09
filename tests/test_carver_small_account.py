"""Carver mo cua von nho: von $100 phai mo duoc lenh dau tien.

Truoc sua: target Carver (~0.0003 BTC) duoi buoc toi thieu san 0.001 ->
buffer tra HOLD vinh vien khi chua co vi the, khong bao gio mo duoc lenh.

Sau sua: mo vi the MOI theo huong forecast (len 1 buoc san 0.001 BTC)
khi ky quy toi thieu vua trong 35% von. Giu HOLD cho rebalance/giam
vi the (tranh fee churn nhu thiet ke goc) va khi forecast yeu (<1.0).
"""
import unittest

from strategy.carver_systematic_engine import CarverSystematicEngine


class CarverSmallAccountTests(unittest.TestCase):
    def setUp(self):
        self.eng = CarverSystematicEngine()

    def open(self, capital=99.79, signal=2.0, held=0.0, price=78500.0, lev=5):
        return self.eng.compute_systematic_position(
            current_price=price, capital_usdt=capital, raw_signal=signal,
            daily_vol_pct=0.015, current_position_contracts=held,
            effective_leverage=lev)

    def test_fresh_position_opens_minimum_step(self):
        out = self.open()
        self.assertEqual(out.rebalance_action, "BUY")
        self.assertEqual(out.contracts_to_execute, 0.001)
        # Ky quy ~$15.7, trong han muc 35% vi $99.
        self.assertLess(out.optimal_margin_usdt, 99.79 * 0.35)

    def test_weak_forecast_stays_hold(self):
        out = self.open(signal=0.2)
        self.assertEqual(out.rebalance_action, "HOLD")
        self.assertEqual(out.contracts_to_execute, 0.0)

    def test_existing_position_keeps_buffer_discipline(self):
        # Dang giu vi the: cua mo toi thieu KHONG ap dung (chi cho mo moi
        # tu 0). Rebalance tuan theo buffer nhu cu — khong force mo them.
        out = self.open(signal=0.5, held=0.01)
        # Khong duoc mo cua thai qua: exec phai nam trong logic buffer
        # (ve 0 hoac giam ve gan target), khong phai +0.001 moi.
        self.assertLessEqual(out.contracts_to_execute, 0.0)

    def test_tiny_balance_stays_hold(self):
        # Von $5: ca lenh toi thieu cung vuot 35% von -> HOLD nhu cu.
        out = self.open(capital=5.0, signal=5.0)
        self.assertEqual(out.rebalance_action, "HOLD")

    def test_large_account_unchanged(self):
        # Von $5000: target Carver vuot buoc san -> di theo duong cu.
        out = self.open(capital=5000.0, signal=8.0)
        self.assertNotEqual(out.rebalance_action, "HOLD")
        self.assertGreaterEqual(abs(out.contracts_to_execute), 0.001)


if __name__ == "__main__":
    unittest.main()
