"""Paper/live ledger separation: doi che do la chuyen han bo so sach.

Bao gom: so du (account_state id 1/2), lenh (trades.mode), bai hoc AI
(trade_memory.mode), muc tieu thang (monthly_* key scope), runtime
(execution_runtime key scope). Reset/xuat-nhap chi cham che do hien tai.
"""
import tempfile
import unittest
from pathlib import Path

from data.persistent_storage import PersistentStorageManager


def make_storage():
    tmp = Path(tempfile.mkdtemp())
    return PersistentStorageManager(tmp / "t.db", tmp / "t.json")


def trade(tid, pnl, exit_time="2026-09-08T12:00:00"):
    return {"id": tid, "execution_id": f"ex-{tid}", "pnl": pnl,
            "exit_time": exit_time, "created_at": "2026-09-08T10:00:00",
            "entry_time": "2026-09-08T09:00:00", "entry_price": 60000.0,
            "exit_price": 60100.0, "fee": 0.5, "reason": "TP",
            "direction": "LONG", "symbol": "BTCUSDT"}


class ModeLedgerSeparationTests(unittest.TestCase):
    def test_trades_and_memory_are_isolated(self):
        s = make_storage()
        s.save_trade(trade(1, 10.0))
        s.save_trade_memory_record(1, 1, 60000.0, 10.0, "TP", 55.0,
                                   "FAIR", "NONE", "BUY", "BULL", "lesson-paper")
        s.set_mode("live")
        self.assertEqual(s.load_trades(), [])
        self.assertEqual(s.load_trade_memory_records(), [])
        s.save_trade(trade(2, -5.0))
        s.save_trade_memory_record(2, -1, 61000.0, -5.0, "SL", 45.0,
                                   "FAIR", "NONE", "SELL", "BEAR", "lesson-live")
        self.assertEqual([t["id"] for t in s.load_trades()], [2])
        self.assertEqual(s.get_monthly_realized_pnl("2026-09"), -5.0)
        s.set_mode("paper")
        self.assertEqual([t["id"] for t in s.load_trades()], [1])
        self.assertEqual([m["trade_id"] for m in s.load_trade_memory_records()], [1])
        self.assertEqual(s.get_monthly_realized_pnl("2026-09"), 10.0)

    def test_account_state_is_isolated(self):
        s = make_storage()
        s.save_account_state("BTCUSDT", 5000.0, 5037.0, 5100.0, 10.0,
                             True, "15m", "AI_AUTO", 3, None)
        s.set_mode("live")
        self.assertIsNone(s.load_account_state())
        s.save_account_state("BTCUSDT", 1000.0, 990.0, 1000.0, 1.0,
                             True, "15m", "AI_AUTO", 3, None)
        self.assertAlmostEqual(s.load_account_state()["current_balance"], 990.0)
        s.set_mode("paper")
        self.assertAlmostEqual(s.load_account_state()["current_balance"], 5037.0)

    def test_monthly_target_keys_are_scoped(self):
        s = make_storage()
        s.save_setting("monthly_target_pct", 10.0)
        s.save_setting("monthly_target_profile", "growth")
        s.set_mode("live")
        # Live chua co gi: ve default, khong doc lan paper
        self.assertEqual(s.get_setting("monthly_target_pct", 10.0), 10.0)
        s.save_setting("monthly_target_pct", 25.0)
        s.save_setting("monthly_target_profile", "aggressive")
        s.set_mode("paper")
        self.assertEqual(s.get_setting("monthly_target_profile"), "growth")
        s.set_mode("live")
        self.assertEqual(s.get_setting("monthly_target_pct"), 25.0)
        self.assertEqual(s.get_setting("monthly_target_profile"), "aggressive")

    def test_runtime_key_is_scoped(self):
        s = make_storage()
        s.save_execution_runtime({"balance": 5037.0, "trades": [], "version": 2})
        s.set_mode("live")
        self.assertIsNone(s.load_execution_runtime())
        s.save_execution_runtime({"balance": 990.0, "trades": [], "version": 2})
        self.assertAlmostEqual(s.load_execution_runtime()["balance"], 990.0)
        s.set_mode("paper")
        self.assertAlmostEqual(s.load_execution_runtime()["balance"], 5037.0)

    def test_reset_only_touches_current_mode(self):
        s = make_storage()
        s.save_trade(trade(1, 10.0))
        s.set_mode("live")
        s.save_trade(trade(2, -5.0))
        s.reset_database(initial_balance=1000.0)
        self.assertEqual(s.load_trades(), [])
        s.set_mode("paper")
        self.assertEqual([t["id"] for t in s.load_trades()], [1])

    def test_export_import_only_current_mode(self):
        s = make_storage()
        s.save_trade(trade(1, 10.0))
        pkg = s.export_full_package()
        self.assertEqual(pkg["mode"], "paper")
        self.assertEqual([t["id"] for t in pkg["trades"]], [1])
        s.set_mode("live")
        # Bo execution_id khi nhap sang che do khac de tranh dedup theo id cu;
        # ban ghi duoc gan mode live.
        for t in pkg["trades"]:
            t.pop("execution_id", None)
        res = s.import_full_package({"trades": pkg["trades"]}, mode="merge")
        self.assertEqual(res["imported_trades"], 1)
        # Nhap vao live: paper van 1 lenh cu, live co 1 lenh nhap
        s.set_mode("paper")
        self.assertEqual([t["id"] for t in s.load_trades()], [1])
        s.set_mode("live")
        self.assertEqual(len(s.load_trades()), 1)

    def test_legacy_data_migrates_to_paper(self):
        import sqlite3
        s = make_storage()
        with s._get_connection() as conn:
            conn.execute("INSERT INTO trades (trade_id, symbol, pnl, exit_time, created_at, "
                         "entry_time, entry_price, exit_price, fee, reason, direction, mode) "
                         "VALUES (99, 'BTCUSDT', 7.0, '2026-09-01T10:00:00', '2026-09-01T09:00:00', "
                         "'2026-09-01T08:00:00', 60000.0, 60100.0, 0.5, 'TP', 'LONG', NULL)")
            conn.execute("INSERT INTO trade_memory_records (trade_id, direction, entry_price, net_pnl, "
                         "exit_reason, rsi, vwap_status, lesson_learned, recorded_at, mode) "
                         "VALUES (99, 1, 60000.0, 7.0, 'TP', 55.0, 'FAIR', 'legacy', "
                         "'2026-09-01T10:00:00', NULL)")
            conn.commit()
        # Re-init de chay migration UPDATE ... WHERE mode IS NULL
        s._init_db()
        self.assertEqual(s.get_monthly_realized_pnl("2026-09"), 7.0)
        self.assertEqual(len(s.load_trade_memory_records()), 1)
        s.set_mode("live")
        self.assertEqual(s.load_trades(), [])


if __name__ == "__main__":
    unittest.main()
