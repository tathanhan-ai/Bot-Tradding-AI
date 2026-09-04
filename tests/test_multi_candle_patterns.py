"""
Unit tests for Multi-Candle Patterns & Multi-Timeframe Strategy Engine
"""
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from strategy.multi_candle_patterns import (
    CandlePatternDetector,
    MultiTimeframeCandleStrategyEngine,
    CandlePatternResult,
    MTFCandleConfluenceResult
)
from strategy.ensemble_strategy import EnsembleCoordinator, MultiCandlePatternSubEngine


def create_base_candles(n=20, base_price=50000.0, step=10.0) -> pd.DataFrame:
    """Helper to create a standard background dataframe"""
    now_epoch = int(datetime.now(timezone.utc).timestamp())
    dates = pd.to_datetime([now_epoch - (n - i) * 900 for i in range(n)], unit='s')
    data = []
    p = base_price
    for i in range(n):
        o = p
        c = p + step
        h = max(o, c) + 15.0
        l = min(o, c) - 15.0
        v = 10.0
        data.append({"open": o, "high": h, "low": l, "close": c, "volume": v})
        p = c
    return pd.DataFrame(data, index=dates)


class TestCandlePatternDetector(unittest.TestCase):

    def test_bullish_engulfing(self):
        df = create_base_candles(15)
        # Previous candle: bearish
        df.iloc[-2] = {"open": 50200.0, "high": 50210.0, "low": 50050.0, "close": 50060.0, "volume": 10.0}
        # Current candle: opens below prev close, closes above prev open
        df.iloc[-1] = {"open": 50040.0, "high": 50250.0, "low": 50030.0, "close": 50230.0, "volume": 25.0}
        res = CandlePatternDetector.detect_patterns(df, "15m")
        self.assertIn(res.pattern_name, ["ENGULFING_BULLISH", "BULLISH_ENGULFING"])
        self.assertEqual(res.bias, "BULLISH")
        self.assertGreaterEqual(res.strength, 70.0)

    def test_bearish_engulfing(self):
        df = create_base_candles(15)
        # Previous candle: bullish
        df.iloc[-2] = {"open": 50000.0, "high": 50150.0, "low": 49990.0, "close": 50140.0, "volume": 10.0}
        # Current candle: opens above prev close, closes below prev open
        df.iloc[-1] = {"open": 50160.0, "high": 50180.0, "low": 49950.0, "close": 49960.0, "volume": 25.0}
        res = CandlePatternDetector.detect_patterns(df, "15m")
        self.assertIn(res.pattern_name, ["ENGULFING_BEARISH", "BEARISH_ENGULFING"])
        self.assertEqual(res.bias, "BEARISH")
        self.assertGreaterEqual(res.strength, 70.0)

    def test_hammer_pinbar(self):
        df = create_base_candles(15)
        # Hammer: long lower shadow >= 60% of range, small body at top
        c_open = 50200.0
        c_close = 50230.0
        c_high = 50240.0
        c_low = 50000.0  # Range = 240, lower shadow = 200 (83% of range)
        df.iloc[-1] = {"open": c_open, "high": c_high, "low": c_low, "close": c_close, "volume": 20.0}
        res = CandlePatternDetector.detect_patterns(df, "5m")
        self.assertEqual(res.pattern_name, "HAMMER_PINBAR")
        self.assertEqual(res.bias, "BULLISH")

    def test_shooting_star(self):
        df = create_base_candles(15)
        # Shooting star: long upper shadow >= 60% of range, small body at bottom
        c_open = 50050.0
        c_close = 50020.0
        c_high = 50300.0  # Range = 290, upper shadow = 250 (86% of range)
        c_low = 50010.0
        df.iloc[-1] = {"open": c_open, "high": c_high, "low": c_low, "close": c_close, "volume": 20.0}
        res = CandlePatternDetector.detect_patterns(df, "5m")
        self.assertEqual(res.pattern_name, "SHOOTING_STAR")
        self.assertEqual(res.bias, "BEARISH")

    def test_morning_star(self):
        df = create_base_candles(15)
        # c2: long red candle
        df.iloc[-3] = {"open": 51000.0, "high": 51050.0, "low": 50480.0, "close": 50500.0, "volume": 15.0}
        # c1: small body gap down (star)
        df.iloc[-2] = {"open": 50400.0, "high": 50450.0, "low": 50350.0, "close": 50390.0, "volume": 8.0}
        # c0: strong green candle closing above midpoint of c2 (50750)
        df.iloc[-1] = {"open": 50420.0, "high": 50850.0, "low": 50410.0, "close": 50820.0, "volume": 25.0}
        res = CandlePatternDetector.detect_patterns(df, "1h")
        self.assertEqual(res.pattern_name, "MORNING_STAR")
        self.assertEqual(res.bias, "BULLISH")

    def test_evening_star(self):
        df = create_base_candles(15)
        # c2: long green candle
        df.iloc[-3] = {"open": 50000.0, "high": 50550.0, "low": 49980.0, "close": 50500.0, "volume": 15.0}
        # c1: small body gap up
        df.iloc[-2] = {"open": 50600.0, "high": 50650.0, "low": 50580.0, "close": 50610.0, "volume": 8.0}
        # c0: strong red candle closing below midpoint of c2 (50250)
        df.iloc[-1] = {"open": 50580.0, "high": 50590.0, "low": 50150.0, "close": 50180.0, "volume": 25.0}
        res = CandlePatternDetector.detect_patterns(df, "1h")
        self.assertEqual(res.pattern_name, "EVENING_STAR")
        self.assertEqual(res.bias, "BEARISH")

    def test_three_white_soldiers(self):
        df = create_base_candles(15)
        df.iloc[-3] = {"open": 50000.0, "high": 50200.0, "low": 49980.0, "close": 50180.0, "volume": 15.0}
        df.iloc[-2] = {"open": 50150.0, "high": 50400.0, "low": 50120.0, "close": 50380.0, "volume": 18.0}
        df.iloc[-1] = {"open": 50350.0, "high": 50620.0, "low": 50330.0, "close": 50600.0, "volume": 22.0}
        res = CandlePatternDetector.detect_patterns(df, "15m")
        self.assertEqual(res.pattern_name, "THREE_WHITE_SOLDIERS")
        self.assertEqual(res.bias, "BULLISH")


class TestMultiTimeframeCandleStrategyEngine(unittest.TestCase):

    def setUp(self):
        self.engine = MultiTimeframeCandleStrategyEngine()

    def test_confluence_evaluation(self):
        data_map = {
            "1m": create_base_candles(25, 50000.0, 5.0),
            "5m": create_base_candles(25, 50000.0, 10.0),
            "15m": create_base_candles(25, 50000.0, 20.0),
            "1h": create_base_candles(25, 50000.0, 50.0),
        }
        res = self.engine.evaluate(data_map, 51000.0)
        self.assertIsInstance(res, MTFCandleConfluenceResult)
        self.assertIn("1m", res.timeframe_patterns)
        self.assertIn("5m", res.timeframe_patterns)
        self.assertIn("15m", res.timeframe_patterns)
        self.assertIn("1h", res.timeframe_patterns)
        self.assertIn(res.confluence_verdict, [
            "STRONG_BULLISH_CASCADE", "BULLISH_BIAS", "NEUTRAL_MIXED", "BEARISH_BIAS", "STRONG_BEARISH_CASCADE"
        ])

    def test_macro_filter_blocks_countertrend(self):
        # Create 1h macro strongly bearish (Three Black Crows)
        df_1h = create_base_candles(25, 50000.0, -50.0)
        df_1h.iloc[-3] = {"open": 51000.0, "high": 51020.0, "low": 50780.0, "close": 50800.0, "volume": 20.0}
        df_1h.iloc[-2] = {"open": 50810.0, "high": 50820.0, "low": 50580.0, "close": 50600.0, "volume": 25.0}
        df_1h.iloc[-1] = {"open": 50610.0, "high": 50620.0, "low": 50380.0, "close": 50400.0, "volume": 30.0}

        # Lower timeframes bullish
        df_1m = create_base_candles(25, 50400.0, 5.0)
        df_5m = create_base_candles(25, 50400.0, 10.0)
        df_15m = create_base_candles(25, 50400.0, 15.0)

        data_map = {"1m": df_1m, "5m": df_5m, "15m": df_15m, "1h": df_1h}
        res = self.engine.evaluate(data_map, 50450.0)
        
        # When 1h is strongly bearish, macro_alignment must be False
        self.assertFalse(res.macro_alignment)


class TestEnsembleIntegration(unittest.TestCase):

    def test_ensemble_includes_candle_engine(self):
        coord = EnsembleCoordinator()
        data_map = {
            "1m": create_base_candles(25, 50000.0, 5.0),
            "5m": create_base_candles(25, 50000.0, 10.0),
            "15m": create_base_candles(50, 50000.0, 20.0),
            "1h": create_base_candles(100, 50000.0, 50.0),
        }
        res = coord.evaluate_ensemble(data_map, 50500.0, "TRENDING_BULL")
        strat_names = [v.name for v in res.votes]
        self.assertTrue(any("Mẫu Hình Nến Đa Khung" in name for name in strat_names))
        self.assertIsNotNone(coord.last_candle_confluence)


if __name__ == "__main__":
    unittest.main()
