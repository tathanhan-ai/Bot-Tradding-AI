"""UAT tri von nho: von $100 phai co cua vao lenh toi thieu san.

- GRID doi von >= $300 (khong nuot von nho).
- Probation bi tran ve 0 duoc nang len buoc toi thieu san khi du han muc.
- Von cuc nho (vi du $5) van veto cung voi ly do ro rang.
- Margin de xuat Phuong an 2 co gian theo von, khong thap hon ky quy
  toi thieu san.
"""
import unittest

from risk.small_account import (
    counter_margin,
    grid_affordable,
    min_viable_margin,
    probation_floor_quantity,
)


PRICE = 79000.0
LEV = 5


class SmallAccountTests(unittest.TestCase):
    def test_grid_needs_300(self):
        # Vi $99 khong du suc nuoi luoi 4 chan -> bo qua GRID.
        self.assertFalse(grid_affordable(99.79, PRICE, LEV))
        # Vi $5000 du suc -> choi GRID binh thuong.
        self.assertTrue(grid_affordable(5000.0, PRICE, LEV))

    def test_min_viable_margin_math(self):
        # 0.001 BTC x $79k / 5x = $15.8 ky quy toi thieu.
        self.assertAlmostEqual(min_viable_margin(PRICE, LEV), 15.8, places=1)

    def test_probation_floor_lifts_to_exchange_minimum(self):
        # Probation 0.10% x $99 = $0.10 risk -> quantity ~0.0003 -> lam tron 0.
        # Sau sua: nang len 0.001 BTC, gan co min_size_floor, khong veto.
        qty, is_floor, veto = probation_floor_quantity(
            0.0003, PRICE, LEV, remaining_margin=30.0, balance=99.79)
        self.assertEqual(qty, 0.001)
        self.assertTrue(is_floor)
        self.assertEqual(veto, "")

    def test_tiny_balance_still_vetoes_with_clear_reason(self):
        # Von $5: ca lenh toi thieu ($15.8 ky quy) cung vuot han muc 35%.
        qty, _, veto = probation_floor_quantity(
            0.0001, PRICE, LEV, remaining_margin=1.0, balance=5.0)
        self.assertEqual(qty, 0.0)
        self.assertIn("qua nho", veto)

    def test_normal_probation_quantity_untouched(self):
        # Quantity probation binh thuong (>= buoc san) giu nguyen.
        qty, is_floor, veto = probation_floor_quantity(
            0.005, PRICE, LEV, remaining_margin=30.0, balance=99.79)
        self.assertEqual(qty, 0.005)
        self.assertFalse(is_floor)
        self.assertEqual(veto, "")

    def test_counter_margin_scales_with_balance(self):
        # Von lon: min(300 dinh muc, 10% von) -> giu dinh muc cu.
        self.assertEqual(counter_margin(5000.0, PRICE, 3, 300.0, 0.10, 600.0), 300.0)
        # Von nho ha ve phan tram nhung khong duoi ky quy toi thieu san.
        small = counter_margin(99.79, PRICE, 5, 300.0, 0.10, 600.0)
        self.assertGreaterEqual(small, min_viable_margin(PRICE, 5))
        self.assertLess(small, 300.0)


if __name__ == "__main__":
    unittest.main()
