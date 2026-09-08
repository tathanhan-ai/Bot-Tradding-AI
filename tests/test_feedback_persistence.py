from datetime import datetime
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from data.persistent_storage import PersistentStorageManager
from risk.monthly_target_governor import MonthlyTargetGovernor
from strategy.shadow_account import ShadowAccountAnalyzer


class SeptemberClock(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 9, 5, 12, 0, 0, tzinfo=tz)


class GovernorStorage:
    def __init__(self, settings=None, balance=1_000.0, pnls=None):
        self.settings = dict(settings or {})
        self.balance = balance
        self.pnls = pnls or {}

    def get_setting(self, key):
        return self.settings.get(key)

    def save_setting(self, key, value):
        self.settings[key] = value

    def load_account_state(self):
        return {"current_balance": self.balance} if self.balance is not None else None

    def get_monthly_realized_pnl(self, month):
        return self.pnls.get(month, 0.0)


class FeedbackPersistenceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="feedback-test-")
        self.addCleanup(self.temp.cleanup)
        self.db_path = Path(self.temp.name) / "feedback.db"
        self.backup_path = Path(self.temp.name) / "backup.json"
        self.storage = PersistentStorageManager(self.db_path, self.backup_path)

    def test_latest_history_is_returned_in_chronological_order(self):
        for tid in range(1, 5):
            self.storage.save_trade({"id": tid, "pnl": -tid})
            self.storage.save_trade_memory_record(tid, 1, 60_000.0, -tid, "SL", 50.0,
                                                 "FAIR", "NONE", "BALANCED", "RANGING", f"loss {tid}")
        self.assertEqual([t["id"] for t in self.storage.load_trades(limit=2)], [3, 4])
        self.assertEqual([t["trade_id"] for t in self.storage.load_trade_memory_records(limit=2)], [3, 4])

    def test_monthly_pnl_uses_close_year_and_preserves_import_dates(self):
        self.storage.save_trade({"id": 1, "pnl": 100, "exit_time": "09-04 12:00:00", "created_at": "2025-09-04T12:00:00"})
        self.storage.save_trade({"id": 2, "pnl": 20, "exit_time": "09-04 12:00:00", "created_at": "2026-09-04T12:00:00"})
        # An old close imported now belongs to the close month, not insertion month.
        self.storage.save_trade({"id": 3, "pnl": 30, "exit_time": "2025-09-03T12:00:00", "created_at": "2026-09-04T12:00:00"})
        self.storage.import_full_package({"trades": [
            {"id": 4, "pnl": 40, "exit_time": "09-02 12:00:00", "created_at": "2025-09-02T12:00:00"}
        ]})
        self.assertEqual(self.storage.get_monthly_realized_pnl("2025-09"), 170)
        self.assertEqual(self.storage.get_monthly_realized_pnl("2026-09"), 20)
        self.assertEqual(self.storage.load_trades()[-1]["created_at"], "2025-09-02T12:00:00")

    def test_runtime_and_full_trade_payload_are_atomic_and_idempotent(self):
        trade = {"id": 1, "execution_id": "binance-101-close", "pnl": 12.0,
                 "closed_at_ts": datetime(2025, 9, 4, 12, 0).timestamp(), "entry_context": {"rsi": 42},
                 "decision_trace": {"order_id": "candidate-1", "entries": ["Stage 1"]}}
        runtime = {"balance": 1_012.0, "position": None, "queue": [], "trades": [trade]}
        self.assertEqual(self.storage.save_execution_runtime(runtime), 1)
        self.assertEqual(self.storage.save_execution_runtime(runtime), 0)
        reopened = PersistentStorageManager(self.db_path, self.backup_path)
        self.assertEqual(reopened.load_execution_runtime(), runtime)
        self.assertEqual(len(reopened.load_trades()), 1)
        self.assertEqual(reopened.get_monthly_realized_pnl("2025-09"), 12)
        for key in ("execution_id", "closed_at_ts", "entry_context", "decision_trace"):
            self.assertEqual(reopened.load_trades()[0][key], trade[key])

        # The first INSERT must roll back when the second trade cannot be encoded into numeric columns.
        broken = {"balance": 2_000.0, "trades": [
            {"execution_id": "binance-102-close", "pnl": 5.0},
            {"execution_id": "binance-103-close", "pnl": "invalid"},
        ]}
        with self.assertRaises(ValueError):
            reopened.save_execution_runtime(broken)
        self.assertEqual(reopened.load_execution_runtime(), runtime)
        self.assertEqual(len(reopened.load_trades()), 1)

    def test_reset_does_not_resurrect_previous_runtime(self):
        self.storage.save_execution_runtime({"balance": 900, "trades": []})
        self.storage.reset_database(initial_balance=1_000)
        self.assertIsNone(self.storage.load_execution_runtime())


class GovernorFeedbackTest(unittest.TestCase):
    def test_first_start_has_no_invented_deficit_and_recovers_month_baseline(self):
        storage = GovernorStorage(balance=1_100, pnls={"2026-09": 100})
        with patch("risk.monthly_target_governor.datetime", SeptemberClock):
            governor = MonthlyTargetGovernor(storage=storage)
            status = governor.evaluate(1_100)
        self.assertEqual(status.carried_deficit_pct, 0.0)
        self.assertEqual(status.month_start_balance, 1_000)
        self.assertEqual(status.regime, "TARGET_ACHIEVED")
        self.assertEqual(status.size_multiplier, 0.5)

    def test_restart_rollover_uses_saved_prior_month_once(self):
        storage = GovernorStorage(settings={"monthly_governor_month": "2026-08", "monthly_start_balance": 1_000,
                                            "monthly_carried_deficit": 2.0},
                                  balance=1_105, pnls={"2026-08": 80, "2026-09": 25})
        with patch("risk.monthly_target_governor.datetime", SeptemberClock):
            governor = MonthlyTargetGovernor(storage=storage)
            self.assertEqual(governor.carried_deficit_pct, 4.0)  # (10 + 2) - 8, not September's 2.5%.
            self.assertEqual(governor.month_start_balance, 1_080)
            restored = MonthlyTargetGovernor(storage=storage)
            self.assertEqual(restored.carried_deficit_pct, 4.0)
            self.assertEqual(restored.month_start_balance, 1_080)

    def test_empty_account_initialization_and_no_storage_rollover(self):
        with patch("risk.monthly_target_governor.datetime", SeptemberClock):
            governor = MonthlyTargetGovernor(storage=GovernorStorage(balance=None))
            self.assertEqual(governor.evaluate(1_000, []).month_start_balance, 1_000)
            self.assertEqual(governor.carried_deficit_pct, 0)
            governor = MonthlyTargetGovernor()
            governor.current_month_str = "2026-08"
            governor.month_start_balance = 1_000
            status = governor.evaluate(920, [
                {"pnl": -100, "exit_time": "2026-08-31T23:00:00"},
                {"pnl": 20, "exit_time": "2026-09-01T01:00:00"},
            ])
        self.assertEqual(status.month_start_balance, 900)
        self.assertEqual(status.month_realized_pnl, 20)
        self.assertEqual(status.carried_deficit_pct, 15)  # capped deficit

    def test_rollover_prefers_atomic_runtime_balance_over_stale_account_table(self):
        storage = GovernorStorage(settings={"monthly_governor_month": "2026-08", "monthly_start_balance": 1_000},
                                  balance=800, pnls={"2026-08": 100, "2026-09": 20})
        storage.load_execution_runtime = lambda: {"balance": 1_120}
        with patch("risk.monthly_target_governor.datetime", SeptemberClock):
            governor = MonthlyTargetGovernor(storage=storage)
        self.assertEqual(governor.month_start_balance, 1_100)
        self.assertEqual(governor.carried_deficit_pct, 0)

    def test_month_matching_never_uses_entry_or_another_year(self):
        self.assertFalse(MonthlyTargetGovernor._is_trade_in_month(
            {"exit_time": "09-03 12:00:00", "created_at": "2025-09-03T12:00:00"}, "2026-09"))
        self.assertFalse(MonthlyTargetGovernor._is_trade_in_month(
            {"entry_time": "2026-09-30T12:00:00", "exit_time": "2026-10-01T12:00:00"}, "2026-09"))


class ShadowFeedbackTest(unittest.TestCase):
    def test_no_price_path_means_no_invented_counterfactual_profit(self):
        analyzer = ShadowAccountAnalyzer()
        report = analyzer.analyze_trade_history([{"pnl": 5, "reason": "Đóng thủ công"}])
        self.assertTrue(report.analysis_available)
        self.assertFalse(report.counterfactual_available)
        self.assertEqual(report.actual_pnl_usdt, 5)
        self.assertEqual(report.missed_alpha_usdt, 0)
        self.assertEqual(report.shadow_systematic_pnl_usdt, 0)
        self.assertTrue(report.counterfactual_reason)
        self.assertFalse(analyzer.analyze_trade_history([]).analysis_available)

    def test_revenge_uses_real_timestamps_and_excludes_overlapping_positions(self):
        analyzer = ShadowAccountAnalyzer()
        history = [
            {"pnl": -10, "exit_time": "2026-09-04T12:00:00"},
            {"pnl": 2, "entry_time": "2026-09-04T11:59:00", "exit_time": "2026-09-04T12:00:20"},
            {"pnl": 1, "entry_time": "2026-09-04T12:00:30", "exit_time": "2026-09-04T12:02:00"},
        ]
        self.assertEqual(analyzer.analyze_trade_history(history).revenge_trades_count, 1)
        self.assertEqual(analyzer.analyze_trade_history([{"pnl": -1}, {"pnl": 1}]).revenge_trades_count, 0)


if __name__ == "__main__":
    unittest.main()
