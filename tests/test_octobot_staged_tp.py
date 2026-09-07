import unittest
from strategy.octobot_matrix import MatrixConsensus
from strategy.octobot_trading_modes import (
    OctoBotTradingCoordinator,
    RangeTradingMode,
    DipAnalyserMode,
    DailyTradingMode,
    OctoBotTradeSetup
)
from ui.server import LiveTradingState

class TestOctoBotStagedTP(unittest.TestCase):
    def test_range_trading_mode_generates_staged_tp(self):
        coord = OctoBotTradingCoordinator()
        consensus = MatrixConsensus(
            matrix_score=0.1,
            confidence_pct=60.0,
            consensus_state="NEUTRAL",
            recommended_direction=0,
            is_tradable=False,
            threshold=0.60,
            tentacles={"TA": {}, "OrderFlow": {}, "SMC": {}, "VWAP": {}, "MTF": {}},
            summary_reason="Sideways market"
        )
        indicators = {"atr": 250.0}
        setup = coord.range_trading.evaluate(
            current_price=90000.0,
            matrix=consensus,
            indicators=indicators,
            smc_data={"demand_zone": (89500.0, 89700.0), "supply_zone": (90500.0, 90800.0)},
            vwap_data={"vwap": 90100.0}
        )
        self.assertIsNotNone(setup)
        self.assertEqual(setup.mode_name, "RANGE_TRADING")
        self.assertIsNotNone(setup.staged_tp)
        self.assertGreater(setup.staged_tp.tp1_price, 0)
        self.assertGreater(setup.staged_tp.tp2_price, setup.staged_tp.tp1_price)
        self.assertGreater(setup.staged_tp.tp3_price, setup.staged_tp.tp2_price)

    def test_coordinator_selects_range_mode_when_flat(self):
        coord = OctoBotTradingCoordinator()
        consensus = MatrixConsensus(
            matrix_score=0.05,
            confidence_pct=50.0,
            consensus_state="NEUTRAL",
            recommended_direction=0,
            is_tradable=False,
            threshold=0.60,
            tentacles={"TA": {}, "OrderFlow": {}, "SMC": {}, "VWAP": {}, "MTF": {}},
            summary_reason="Range market"
        )
        indicators = {"atr": 200.0}
        mode_name, setup = coord.select_best_setup(
            current_price=90000.0,
            matrix=consensus,
            indicators=indicators,
            smc_data={"demand_zone": (89000.0, 89200.0), "supply_zone": (90800.0, 91000.0)},
            of_data=None,
            vwap_data={"vwap": 90000.0}
        )
        self.assertEqual(mode_name, "RANGE_TRADING")
        self.assertIsNotNone(setup)
        self.assertIsNotNone(setup.staged_tp)

    def test_server_state_contains_valid_staged_tp(self):
        # Dung state sach (khong vi the live) de tranh nhiem state live ban (VD dang co SHORT that
        # entry ~79k trong khi test dat live_price gia 85k -> TP tinh tu entry that, so sanh sai).
        import tempfile
        from pathlib import Path
        from data.persistent_storage import PersistentStorageManager
        tmp = tempfile.TemporaryDirectory(prefix="octo-staged-")
        try:
            state = LiveTradingState(storage=PersistentStorageManager(Path(tmp.name) / "db.sqlite", Path(tmp.name) / "backup.json"))
            state.live_price = 85000.0
            state.current_position = None
            data = state.get_state_dict()
        finally:
            tmp.cleanup()
        self.assertIn("octobot", data)
        self.assertIn("staged_tp", data["octobot"])
        tp = data["octobot"]["staged_tp"]
        self.assertIsNotNone(tp)
        self.assertGreater(tp["tp1"], 0)
        if state.current_position and state.current_position.get("direction", 1) == -1:
            self.assertGreater(tp["tp1"], tp["tp2"])
            self.assertGreater(tp["tp2"], tp["tp3"])
        else:
            self.assertGreater(tp["tp2"], tp["tp1"])
            self.assertGreater(tp["tp3"], tp["tp2"])

if __name__ == "__main__":
    unittest.main()
