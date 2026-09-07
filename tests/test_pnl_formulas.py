import unittest

from risk.fee_and_spread_engine import SpreadFeeEngine


class PnlFormulaTests(unittest.TestCase):
    """Chuan Binance: UPNL = gross theo mark price (KHONG tru phi, phi tru vao wallet khi khop).
    ROE = UPNL / initial margin. net_upnl = gross - phi thoat uoc tinh (taker)."""

    def test_upnl_is_gross_not_net(self):
        fee = SpreadFeeEngine()
        entry, price, units, direction = 79204.4, 79185.05, 0.002, -1
        gross = (price - entry) * units * direction
        est_exit = fee.calculate_fee(price * units, is_maker=False)
        net_float = gross - est_exit
        # Len nhay cam: gross duong nho nhung net that am vi phi
        self.assertGreater(gross, 0)
        self.assertLess(net_float, 0)
        # Bot hien thi UPNL=gross, net_upnl=net
        self.assertAlmostEqual(gross, 0.0387, delta=0.01)

    def test_realized_net_counts_both_fees_once(self):
        fee = SpreadFeeEngine()
        entry, exit_px, units = 79204.4, 79185.05, 0.002
        entry_fee = fee.calculate_fee(entry * units, is_maker=False)
        exit_fee = fee.calculate_fee(exit_px * units, is_maker=False)
        gross = (exit_px - entry) * units * -1
        net = gross - entry_fee - exit_fee
        self.assertAlmostEqual(entry_fee + exit_fee, 0.1584, delta=0.01)
        self.assertLess(net, gross)

    def test_roe_uses_initial_margin(self):
        # ROE chuan = UPNL(gross) / initial margin, khong phai net/margin hien tai
        upnl, initial_margin = 2.7169, 160.48
        roe = upnl / initial_margin * 100
        self.assertAlmostEqual(roe, 1.69, delta=0.05)


if __name__ == "__main__":
    unittest.main()
