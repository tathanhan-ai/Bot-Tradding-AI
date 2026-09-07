import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from risk.ai_order_researcher import AIOrderResearcher


def make_df(n=100, base=80000.0):
    prices = [base + np.sin(i / 5.0) * 150.0 for i in range(n)]
    return pd.DataFrame({
        "open": prices, "high": [p + 40.0 for p in prices], "low": [p - 40.0 for p in prices],
        "close": prices, "volume": [50.0 + (i % 7) for i in range(n)],
    }, index=pd.date_range("2026-01-01", periods=n, freq="15min"))


class TacticalAnchorTests(unittest.TestCase):
    def setUp(self):
        self.r = AIOrderResearcher()
        self.df = make_df()

    def _base_kwargs(self, **over):
        kw = dict(
            current_price=80000.0, best_bid=79995.0, best_ask=80005.0, spread=10.0,
            indicators={"atr": 180.0, "rsi": 45.0, "adx": 18.0, "chop": 52.0},
            ai_verdict=SimpleNamespace(regime="RANGING_SIDEWAY"),
            ensemble_result=SimpleNamespace(consensus_score=0.0, confidence=75.0, consensus_verdict="NEUTRAL"),
            ai_cro=None, current_balance=5059.69,
            df_structure=self.df, df_macro=self.df, active_timeframe="15m",
        )
        kw.update(over)
        return kw

    def test_conditional_requires_cvd_confirmation(self):
        # Nen bien nhung CVD BALANCED (khong xac nhan) -> POST_ONLY, khong phuc kich oan
        res = self.r.research(**self._base_kwargs(
            order_flow_verdict=SimpleNamespace(delta_momentum="BALANCED"),
        ))
        self.assertNotEqual(res.recommended_type, "CONDITIONAL")

    def test_conditional_with_cvd_uses_wide_trigger(self):
        res = self.r.research(**self._base_kwargs(
            indicators={"atr": 180.0, "rsi": 45.0, "adx": 15.0, "chop": 60.0, "bb_width": 0.010},
            order_flow_verdict=SimpleNamespace(delta_momentum="STRONG_BUY_PRESSURE"),
        ))
        if res.recommended_type == "CONDITIONAL":
            # Trigger phai cach >= 0.5 ATR (thay vi 0.25 ATR cu)
            self.assertGreaterEqual(abs(res.optimal_trigger_price - 80000.0), 0.5 * 180.0)


if __name__ == "__main__":
    unittest.main()
