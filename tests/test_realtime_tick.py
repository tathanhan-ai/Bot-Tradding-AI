import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from data.persistent_storage import PersistentStorageManager
from ui.server import LiveTradingState


def make_state():
    tmp = tempfile.TemporaryDirectory(prefix="rt-tick-")
    root = Path(tmp.name)
    s = LiveTradingState(storage=PersistentStorageManager(root / "db.sqlite", root / "backup.json"))
    s.vibe_swarm.enabled = False
    s.monthly_governor.enabled = False
    s.decision_mode = "DETERMINISTIC_ONLY"
    now = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("min")
    for tf, rule in (("1m", "min"), ("15m", "15min")):
        end = now.floor(rule) - pd.Timedelta(rule if rule[0].isdigit() else "1" + rule)
        idx = pd.date_range(end=end, periods=60, freq=rule)
        n = np.arange(60)
        price = 80000 + n * 1.0
        s.data_map[tf] = pd.DataFrame(dict(open=price - 5, high=price + 30, low=price - 30,
                                            close=price, volume=np.full(60, 10.0), quote_volume=price * 10), index=idx)
    s.live_price = 80000.0
    s.mark_price = 80010.0
    s.fee_engine.update_book(79999.9, 80000.1)
    s._tmp = tmp
    return s


class RealtimeTickTests(unittest.TestCase):
    def test_fast_pnl_tick_uses_mark(self):
        s = make_state()
        try:
            s.current_position = {"direction": 1, "entry_price": 79900.0, "units": 0.01,
                                  "margin": 199.75, "initial_margin": 199.75}
            s.fast_pnl_tick()
            # (80010-79900)*0.01 = 1.10 gross theo mark
            self.assertAlmostEqual(s.current_position["unrealized_pnl"], 1.10, delta=0.01)
        finally:
            s._tmp.cleanup()

    def test_fast_pnl_tick_falls_back_to_live(self):
        s = make_state()
        try:
            s.mark_price = 0.0
            s.live_price = 80020.0
            s.current_position = {"direction": 1, "entry_price": 79900.0, "units": 0.01,
                                  "margin": 200.0, "initial_margin": 200.0}
            s.fast_pnl_tick()
            self.assertAlmostEqual(s.current_position["unrealized_pnl"], 1.20, delta=0.01)
        finally:
            s._tmp.cleanup()

    def test_mark_or_live_prefers_mark(self):
        s = make_state()
        try:
            s.mark_price = 80010.0
            s.live_price = 80000.0
            self.assertEqual(s.mark_or_live(), 80010.0)
            s.mark_price = 0.0
            self.assertEqual(s.mark_or_live(), 80000.0)
        finally:
            s._tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
