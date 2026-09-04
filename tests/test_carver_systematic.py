"""
Unit tests for Robert Carver Systematic Quantitative Framework (pysystemtrade port)
Verifies all 7 institutional pillars:
1. Forecast Scaling
2. Forecast Capping
3. Volatility Targeting
4. Position Sizing
5. Diversification Multiplier (RDM)
6. Risk Overlay (Drawdown & Volatility Shock)
7. Turnover & Cost Buffer Bands
"""

import unittest
import math
import numpy as np

from strategy.carver_systematic_engine import (
    ForecastScaler,
    ForecastCapper,
    VolatilityTargeter,
    CarverPositionSizer,
    DiversificationMultiplier,
    RiskOverlayEngine,
    PositionBufferManager,
    CarverSystematicEngine,
    CarverSystematicOutput
)


class TestCarverSystematicFramework(unittest.TestCase):

    def test_forecast_scaler_and_capper(self):
        scaler = ForecastScaler(target_abs_forecast=10.0)
        # Synthetic historical signals with mean absolute value = 2.0
        hist = [2.0, -2.0, 1.5, -2.5, 2.0, -1.8, 2.2, -2.0, 1.9, -2.1]
        calc_scalar = scaler.calculate_scalar(hist)
        self.assertAlmostEqual(calc_scalar, 5.0, places=1)

        # Scale a raw signal of 1.5 -> 7.5
        scaled = scaler.scale(1.5, calc_scalar)
        self.assertAlmostEqual(scaled, 7.5, places=1)

        # Test capper
        capper = ForecastCapper(max_forecast=20.0)
        self.assertEqual(capper.cap(7.5), 7.5)
        # Extreme fat-tail signal: 55.0 -> capped to 20.0
        self.assertEqual(capper.cap(55.0), 20.0)
        # Negative extreme: -45.0 -> capped to -20.0
        self.assertEqual(capper.cap(-45.0), -20.0)

    def test_volatility_targeter(self):
        # $5000 capital, 25% annual vol target -> Daily cash vol target ~ $65.43
        targeter = VolatilityTargeter(annual_vol_target_pct=0.25)
        daily_target = targeter.get_daily_cash_vol_target(5000.0)
        expected = (5000.0 * 0.25) / math.sqrt(365.0)
        self.assertAlmostEqual(daily_target, expected, places=2)
        self.assertGreater(daily_target, 60.0)
        self.assertLess(daily_target, 70.0)

    def test_carver_position_sizer(self):
        sizer = CarverPositionSizer()
        price = 65000.0
        daily_vol = 0.02  # 2% daily volatility
        inst_vol = sizer.compute_instrument_value_vol(price, daily_vol)
        self.assertAlmostEqual(inst_vol, 1300.0, places=1)

        # Normal forecast of +10.0 with cash target $65.43
        contracts_normal = sizer.calculate_contracts(
            daily_cash_vol_target=65.43,
            capped_forecast=10.0,
            instrument_value_vol=inst_vol,
            weight=1.0,
            diversification_mult=1.0
        )
        # 65.43 * (10/10) / 1300 ~ 0.05033 BTC
        self.assertAlmostEqual(contracts_normal, 0.05033, places=4)

        # Inverse volatility test: When market volatility doubles to 4%, position halves!
        inst_vol_double = sizer.compute_instrument_value_vol(price, 0.04)
        contracts_high_vol = sizer.calculate_contracts(
            daily_cash_vol_target=65.43,
            capped_forecast=10.0,
            instrument_value_vol=inst_vol_double,
            weight=1.0,
            diversification_mult=1.0
        )
        self.assertAlmostEqual(contracts_high_vol, contracts_normal / 2.0, places=4)

        # Max conviction (+20.0 forecast) doubles normal contracts
        contracts_max_bull = sizer.calculate_contracts(
            daily_cash_vol_target=65.43,
            capped_forecast=20.0,
            instrument_value_vol=inst_vol,
            weight=1.0,
            diversification_mult=1.0
        )
        self.assertAlmostEqual(contracts_max_bull, contracts_normal * 2.0, places=4)

    def test_diversification_multiplier(self):
        div_engine = DiversificationMultiplier()
        # 4 equal-weighted strategies with avg correlation 0.35
        weights = [0.25, 0.25, 0.25, 0.25]
        rdm = div_engine.calculate_rdm(weights, avg_correlation=0.35)
        # RDM should be greater than 1.0 (capital efficiency benefit) and reasonable (< 1.8)
        self.assertGreater(rdm, 1.25)
        self.assertLess(rdm, 1.65)

    def test_risk_overlay(self):
        overlay = RiskOverlayEngine(drawdown_threshold=0.05, max_drawdown_limit=0.20, min_degear_mult=0.15)
        
        # 0% DD -> full multiplier
        self.assertEqual(overlay.calculate_drawdown_multiplier(0.0), 1.0)
        # 5% DD -> full multiplier
        self.assertEqual(overlay.calculate_drawdown_multiplier(0.05), 1.0)
        # 12.5% DD -> halfway de-geared (~0.575)
        self.assertAlmostEqual(overlay.calculate_drawdown_multiplier(0.125), 0.575, places=2)
        # 25% DD (above max 20%) -> min multiplier 0.15
        self.assertEqual(overlay.calculate_drawdown_multiplier(0.25), 0.15)

        # Volatility shock: current vol 4% vs baseline 2% (ratio 2.0 > 1.5)
        vol_shock = overlay.calculate_vol_shock_multiplier(current_vol=0.04, baseline_vol=0.02)
        self.assertLess(vol_shock, 1.0)
        self.assertGreater(vol_shock, 0.7)

    def test_turnover_buffer_bands(self):
        buffer_mgr = PositionBufferManager(buffer_pct=0.10, min_contract_step=0.001)
        target = 0.100  # Target is 0.100 BTC
        # Buffer is 10% -> width = 0.010 BTC. Range: [0.090, 0.110]
        
        # Case 1: Current is 0.095 (inside buffer) -> HOLD, 0 execution
        action, exec_qty, b_low, b_high, b_w = buffer_mgr.evaluate_buffer(
            target_contracts=target,
            current_contracts=0.095,
            contract_step=0.001
        )
        self.assertEqual(action, 'HOLD')
        self.assertEqual(exec_qty, 0.0)
        self.assertAlmostEqual(b_low, 0.090, places=3)
        self.assertAlmostEqual(b_high, 0.110, places=3)

        # Case 2: Current is 0.080 (below buffer 0.090) -> BUY +0.020
        action, exec_qty, _, _, _ = buffer_mgr.evaluate_buffer(
            target_contracts=target,
            current_contracts=0.080,
            contract_step=0.001
        )
        self.assertEqual(action, 'BUY')
        self.assertAlmostEqual(exec_qty, 0.020, places=3)

        # Case 3: Current is 0.125 (above buffer 0.110) -> SELL -0.025
        action, exec_qty, _, _, _ = buffer_mgr.evaluate_buffer(
            target_contracts=target,
            current_contracts=0.125,
            contract_step=0.001
        )
        self.assertEqual(action, 'SELL')
        self.assertAlmostEqual(exec_qty, -0.025, places=3)

    def test_carver_systematic_engine_master(self):
        engine = CarverSystematicEngine(
            annual_vol_target_pct=0.25,
            max_forecast=20.0,
            default_rdm=1.35,
            buffer_pct=0.12
        )

        res = engine.compute_systematic_position(
            current_price=64250.0,
            capital_usdt=5000.0,
            raw_signal=1.8,
            daily_vol_pct=0.022,
            current_position_contracts=0.040,
            baseline_vol_pct=0.020,
            current_drawdown_pct=0.02,
            strategy_weight=1.0,
            ensemble_correlations=0.35,
            ensemble_num_rules=4,
            historical_signals=[1.5, -1.8, 2.0, -1.2, 1.9, -2.1, 1.6, -1.7, 2.2, -1.9],
            effective_leverage=5,
            contract_step=0.001
        )

        self.assertIsInstance(res, CarverSystematicOutput)
        self.assertGreater(res.forecast_scalar, 0.0)
        self.assertGreater(res.daily_cash_vol_target, 50.0)
        self.assertGreater(res.optimal_contracts, 0.0)
        self.assertGreater(res.optimal_margin_usdt, 0.0)
        self.assertIn(res.rebalance_action, ['HOLD', 'BUY', 'SELL'])
        self.assertIn("Carver Forecast", res.summary_rationale)
        print(f"\nCarver Master Test Output:\n{res.summary_rationale}")


if __name__ == '__main__':
    unittest.main()
