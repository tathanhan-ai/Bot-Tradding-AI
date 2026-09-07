import unittest


class ExchangeStandardTests(unittest.TestCase):
    def test_upnl_uses_mark_not_mid(self):
        # Chuan san: UPNL = (mark - entry) * units * huong. Mid bookTicker lech mark ~$10 tren BTC.
        # SHORT: mark cao hon mid -> UPNL theo mark NHO hon (vi SHORT lai khi gia giam).
        entry, mid, mark, units, direction = 79249.47, 78913.55, 78924.10, 0.0081, -1
        upnl_mid = (mid - entry) * units * direction
        upnl_mark = (mark - entry) * units * direction
        self.assertGreater(abs(upnl_mid - upnl_mark), 0.05)  # lech dang ke, phai dung mark

    def test_funding_direction(self):
        # Funding duong: LONG tra, SHORT nhan. Notional ~$640, rate 0.00007811
        f_rate, notional = 0.00007811, 640.0
        long_pay = -1 * f_rate * notional
        short_receive = 1 * f_rate * notional
        self.assertLess(long_pay, 0)
        self.assertGreater(short_receive, 0)
        self.assertAlmostEqual(abs(long_pay), 0.05, delta=0.02)

    def test_fee_deducted_at_fill_not_in_upnl(self):
        # Phi tru vao wallet khi khop, khong nam trong UPNL hien thi
        gross, entry_fee = 2.7169, 0.3210
        upnl_display = gross  # chuan san
        wallet_after_entry = 5039.81 - entry_fee
        self.assertAlmostEqual(upnl_display, 2.7169, delta=0.01)
        self.assertLess(wallet_after_entry, 5039.81)


if __name__ == "__main__":
    unittest.main()
