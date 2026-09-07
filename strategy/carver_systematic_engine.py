"""
Robert Carver Systematic Quantitative Trading Framework (pysystemtrade core port)
100% Self-Contained, Zero External Dependencies (Pure Python + NumPy + Pandas).

Implements the 7 Institutional Systematic Pillars:
1. Forecast Scaling: Rescales raw strategy indicators so expected absolute forecast is 10.0.
2. Forecast Capping: Hard-caps forecast at [-20.0, +20.0] to prevent fat-tail outlier sizing.
3. Volatility Targeting: Converts annualized target volatility (e.g. 25%) into daily cash volatility target.
4. Position Sizing: Sizes positions in contracts based on Instrument Value Volatility.
5. Portfolio / Rule Diversification Multiplier (RDM / IDM): Adjusts for uncorrelated multi-rule ensemble.
6. Risk Overlay: Drawdown de-gearing de-multiplier, volatility spike throttling, and margin headroom guard.
7. Turnover & Trading-Cost Control: Position buffer bands (inertia) to avoid fee churn on micro-deviations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import math
import numpy as np
import pandas as pd


@dataclass
class CarverSystematicOutput:
    # 1. Forecast Scaling & Capping
    raw_forecast: float
    forecast_scalar: float
    scaled_forecast: float
    capped_forecast: float                  # Clamped to [-20.0, +20.0]

    # 2. Volatility Targeting
    annual_vol_target_pct: float            # e.g. 0.25 (25% annual vol)
    daily_cash_vol_target: float            # In USDT (e.g. Capital * vol% / sqrt(365))
    daily_price_vol_pct: float              # Daily volatility of the instrument
    instrument_value_vol: float             # Daily dollar risk per 1 full contract (Price * daily_vol)

    # 3. Diversification & Weights
    rule_diversification_mult: float        # RDM (typically 1.2 - 1.5)
    effective_strategy_weight: float        # Strategy allocation weight

    # 4. Unconstrained Position
    raw_optimal_contracts: float            # Position size before risk overlay (can be positive or negative)

    # 5. Risk Overlay
    risk_overlay_multiplier: float          # Combined risk dampener (0.15 - 1.0)
    drawdown_multiplier: float              # Drawdown de-gearing (0.15 - 1.0)
    vol_shock_multiplier: float             # Volatility shock dampener (0.3 - 1.0)
    margin_headroom_buffer: float           # Available free margin fraction

    # 6. Final Recommended Position
    optimal_contracts: float                # Final contracts to hold (+ Long, - Short)
    optimal_margin_usdt: float              # Required margin in USDT at effective leverage
    effective_leverage: int

    # 7. Turnover & Cost Control (Buffer Bands)
    current_position_contracts: float
    buffer_width_contracts: float           # Buffer band half-width
    buffer_lower_bound: float               # Below this -> Buy to rebalance
    buffer_upper_bound: float               # Above this -> Sell to rebalance
    rebalance_action: str                   # 'HOLD', 'BUY', 'SELL'
    contracts_to_execute: float             # Net contracts to execute (0 if inside buffer)
    fee_saved_usdt: float                   # Estimated fee saved by avoiding churn
    summary_rationale: str                  # Human-readable quantitative summary


class ForecastScaler:
    """
    Pillar 1: Rescales raw rule signals so that the long-run absolute average value is 10.0.
    In Carver's framework:
        +10.0 = Standard bullish forecast
        -10.0 = Standard bearish forecast
          0.0 = Neutral
    """
    def __init__(self, target_abs_forecast: float = 10.0, default_scalar: float = 1.0):
        self.target_abs_forecast = target_abs_forecast
        self.default_scalar = default_scalar

    def calculate_scalar(self, historical_signals: Optional[List[float]]) -> float:
        if not historical_signals or len(historical_signals) < 10:
            return self.default_scalar
        
        abs_vals = [abs(x) for x in historical_signals if abs(x) > 1e-4]
        if not abs_vals:
            return self.default_scalar
        
        mean_abs = float(np.mean(abs_vals))
        if mean_abs < 1e-6:
            return self.default_scalar
        
        scalar = self.target_abs_forecast / mean_abs
        return float(np.clip(scalar, 0.05, 50.0))

    def scale(self, raw_signal: float, scalar: float) -> float:
        return float(raw_signal * scalar)


class ForecastCapper:
    """
    Pillar 2: Clamps scaled forecast to [-max_forecast, +max_forecast] (default [-20.0, +20.0]).
    Eliminates fat-tail black swan sizing errors.
    """
    def __init__(self, max_forecast: float = 20.0):
        self.max_forecast = max_forecast

    def cap(self, scaled_forecast: float) -> float:
        return float(np.clip(scaled_forecast, -self.max_forecast, self.max_forecast))


class VolatilityTargeter:
    """
    Pillar 3: Converts annualized target volatility to daily cash volatility target.
    Formula:
        Daily Cash Vol Target = (Capital * Annual Vol Target) / sqrt(365)
    """
    def __init__(self, annual_vol_target_pct: float = 0.25):
        self.annual_vol_target_pct = annual_vol_target_pct

    def get_daily_cash_vol_target(
        self,
        capital: float,
        monthly_target_pct: Optional[float] = None,
        governor_multiplier: float = 1.0
    ) -> float:
        if capital <= 0:
            return 0.0
        
        eff_annual_vol = self.annual_vol_target_pct
        if monthly_target_pct is not None and monthly_target_pct > 0:
            implied_annual_return = (monthly_target_pct / 100.0) * 12.0
            implied_vol = implied_annual_return / 1.50
            eff_annual_vol = float(np.clip(implied_vol, 0.15, 0.40))

        daily_target = (capital * eff_annual_vol) / math.sqrt(365.0)
        gov_mult = max(0.20, min(1.0, float(governor_multiplier)))
        return float(daily_target * gov_mult)


class CarverPositionSizer:
    """
    Pillar 4: Calculates optimal contracts based on instrument value volatility.
    Formula:
        Instrument Value Vol = Current Price * Daily Volatility%
        Optimal Position = (Cash Vol Target * (Forecast / 10) * Weight * Diversification_Mult) / Instrument Value Vol
    """
    def __init__(self):
        pass

    def compute_instrument_value_vol(self, current_price: float, daily_vol_pct: float) -> float:
        safe_daily_vol = max(daily_vol_pct, 0.005)
        return float(current_price * safe_daily_vol)

    def calculate_contracts(
        self,
        daily_cash_vol_target: float,
        capped_forecast: float,
        instrument_value_vol: float,
        weight: float = 1.0,
        diversification_mult: float = 1.0
    ) -> float:
        if instrument_value_vol <= 1e-4:
            return 0.0
        forecast_multiplier = capped_forecast / 10.0
        contracts = (daily_cash_vol_target * forecast_multiplier * weight * diversification_mult) / instrument_value_vol
        return float(contracts)


class DiversificationMultiplier:
    """
    Pillar 5: Portfolio & Rule Diversification Multiplier (IDM / RDM).
    Calculates RDM = 1 / sqrt(w^T * Sigma_corr * w)
    """
    def __init__(self, default_rdm: float = 1.35):
        self.default_rdm = default_rdm

    def calculate_rdm(self, weights: List[float], avg_correlation: float = 0.35) -> float:
        n = len(weights)
        if n <= 1:
            return 1.0
        
        w = np.array(weights, dtype=float)
        sum_w = np.sum(w)
        if sum_w > 0:
            w = w / sum_w
        
        var_portfolio = (1.0 - avg_correlation) * np.sum(w ** 2) + avg_correlation
        if var_portfolio <= 1e-4:
            return 1.0
        
        rdm = 1.0 / math.sqrt(var_portfolio)
        return float(np.clip(rdm, 1.0, 2.5))


class RiskOverlayEngine:
    """
    Pillar 6: Institutional Multi-Layer Risk Overlay.
    1. Drawdown De-Gearing: Linearly scales down exposure if drawdown exceeds threshold.
    2. Volatility Shock Throttling: Reduces size if current vol >> baseline vol.
    3. Margin Headroom Guard: Protects minimum free margin buffer.
    """
    def __init__(
        self,
        drawdown_threshold: float = 0.05,
        max_drawdown_limit: float = 0.20,
        min_degear_mult: float = 0.15
    ):
        self.dd_threshold = drawdown_threshold
        self.max_dd_limit = max_drawdown_limit
        self.min_degear = min_degear_mult

    def calculate_drawdown_multiplier(self, current_drawdown_pct: float) -> float:
        dd = max(0.0, current_drawdown_pct)
        if dd <= self.dd_threshold:
            return 1.0
        if dd >= self.max_dd_limit:
            return self.min_degear
        
        slope = (1.0 - self.min_degear) / (self.max_dd_limit - self.dd_threshold)
        multiplier = 1.0 - slope * (dd - self.dd_threshold)
        return float(np.clip(multiplier, self.min_degear, 1.0))

    def calculate_vol_shock_multiplier(
        self,
        current_vol: float,
        baseline_vol: float
    ) -> float:
        if baseline_vol <= 1e-6 or current_vol <= 1e-6:
            return 1.0
        
        ratio = current_vol / baseline_vol
        if ratio <= 1.5:
            return 1.0
        
        dampener = math.sqrt(1.5 / ratio)
        return float(np.clip(dampener, 0.30, 1.0))

    def combined_multiplier(
        self,
        current_drawdown_pct: float,
        current_vol: float,
        baseline_vol: float
    ) -> Tuple[float, float, float]:
        dd_mult = self.calculate_drawdown_multiplier(current_drawdown_pct)
        vol_mult = self.calculate_vol_shock_multiplier(current_vol, baseline_vol)
        combined = float(dd_mult * vol_mult)
        return combined, dd_mult, vol_mult


class PositionBufferManager:
    """
    Pillar 7: Turnover & Trading-Cost Control (Carver Buffer Bands).
    Buffer Width = buffer_pct * |Target Position|
    """
    def __init__(self, buffer_pct: float = 0.25, min_contract_step: float = 0.001, min_rebalance_notional: float = 350.0):
        self.buffer_pct = buffer_pct
        self.min_contract_step = min_contract_step
        self.min_rebalance_notional = min_rebalance_notional

    def evaluate_buffer(
        self,
        target_contracts: float,
        current_contracts: float,
        contract_step: float = 0.001,
        current_price: float = 0.0,
        min_notional: float = 350.0
    ) -> Tuple[str, float, float, float, float]:
        step = max(contract_step, self.min_contract_step)
        abs_target = abs(target_contracts)
        buffer_width = max(abs_target * self.buffer_pct, step * 2.0)
        eff_min_notional = max(min_notional, self.min_rebalance_notional)
        
        if abs_target < step:
            buffer_lower = -step
            buffer_upper = step
            if abs(current_contracts) >= step:
                action = 'SELL' if current_contracts > 0 else 'BUY'
                contracts_to_exec = -current_contracts
                if current_price > 0.0 and abs(contracts_to_exec) * current_price < eff_min_notional:
                    return 'HOLD', 0.0, buffer_lower, buffer_upper, buffer_width
                return action, contracts_to_exec, buffer_lower, buffer_upper, buffer_width
            else:
                return 'HOLD', 0.0, buffer_lower, buffer_upper, buffer_width

        buffer_lower = target_contracts - buffer_width
        buffer_upper = target_contracts + buffer_width

        if current_contracts < buffer_lower:
            action = 'BUY'
            contracts_to_exec = target_contracts - current_contracts
        elif current_contracts > buffer_upper:
            action = 'SELL'
            contracts_to_exec = target_contracts - current_contracts
        else:
            action = 'HOLD'
            contracts_to_exec = 0.0

        # Enforce minimum rebalance notional floor ($350 USDT) to prevent fee churn
        if action != 'HOLD' and current_price > 0.0 and (abs(contracts_to_exec) * current_price) < eff_min_notional:
            action = 'HOLD'
            contracts_to_exec = 0.0

        return action, contracts_to_exec, buffer_lower, buffer_upper, buffer_width


class CarverSystematicEngine:
    """
    Master Integration Engine for Robert Carver's Systematic Futures Trading Architecture.
    """
    def __init__(
        self,
        annual_vol_target_pct: float = 0.25,
        max_forecast: float = 20.0,
        default_rdm: float = 1.35,
        buffer_pct: float = 0.25,
        drawdown_threshold: float = 0.05,
        max_drawdown_limit: float = 0.20
    ):
        self.scaler = ForecastScaler(target_abs_forecast=10.0)
        self.capper = ForecastCapper(max_forecast=max_forecast)
        self.vol_targeter = VolatilityTargeter(annual_vol_target_pct=annual_vol_target_pct)
        self.sizer = CarverPositionSizer()
        self.div_mult = DiversificationMultiplier(default_rdm=default_rdm)
        self.risk_overlay = RiskOverlayEngine(
            drawdown_threshold=drawdown_threshold,
            max_drawdown_limit=max_drawdown_limit
        )
        self.buffer_manager = PositionBufferManager(buffer_pct=buffer_pct)

    def compute_systematic_position(
        self,
        current_price: float,
        capital_usdt: float,
        raw_signal: float,
        daily_vol_pct: float,
        current_position_contracts: float = 0.0,
        baseline_vol_pct: Optional[float] = None,
        current_drawdown_pct: float = 0.0,
        strategy_weight: float = 1.0,
        ensemble_correlations: float = 0.35,
        ensemble_num_rules: int = 4,
        historical_signals: Optional[List[float]] = None,
        effective_leverage: int = 5,
        contract_step: float = 0.001,
        monthly_target_pct: Optional[float] = None,
        governor_multiplier: float = 1.0
    ) -> CarverSystematicOutput:
        # 1. Forecast Scaling
        scalar = self.scaler.calculate_scalar(historical_signals)
        scaled_fc = self.scaler.scale(raw_signal, scalar)

        # 2. Forecast Capping [-20, +20]
        capped_fc = self.capper.cap(scaled_fc)

        # 3. Volatility Targeting
        daily_cash_vol = self.vol_targeter.get_daily_cash_vol_target(
            capital_usdt,
            monthly_target_pct=monthly_target_pct,
            governor_multiplier=governor_multiplier
        )
        base_vol = baseline_vol_pct if (baseline_vol_pct and baseline_vol_pct > 0) else daily_vol_pct
        inst_val_vol = self.sizer.compute_instrument_value_vol(current_price, daily_vol_pct)

        # 4. Diversification Multiplier (RDM)
        weights = [1.0 / max(1, ensemble_num_rules)] * ensemble_num_rules
        rdm = self.div_mult.calculate_rdm(weights, avg_correlation=ensemble_correlations)

        # 5. Position Sizing (Unconstrained)
        raw_contracts = self.sizer.calculate_contracts(
            daily_cash_vol_target=daily_cash_vol,
            capped_forecast=capped_fc,
            instrument_value_vol=inst_val_vol,
            weight=strategy_weight,
            diversification_mult=rdm
        )

        # 6. Risk Overlay
        combined_mult, dd_mult, vol_mult = self.risk_overlay.combined_multiplier(
            current_drawdown_pct=current_drawdown_pct,
            current_vol=daily_vol_pct,
            baseline_vol=base_vol
        )

        final_contracts = raw_contracts * combined_mult

        # Margin Headroom Guard (Cap total margin at 35% of capital)
        max_notional = capital_usdt * effective_leverage * 0.35
        max_allowed_contracts = max_notional / max(current_price, 1.0)
        final_contracts = float(np.clip(final_contracts, -max_allowed_contracts, max_allowed_contracts))

        if contract_step > 0:
            final_contracts = round(final_contracts / contract_step) * contract_step

        notional_value = abs(final_contracts) * current_price
        lev = max(1, effective_leverage)
        optimal_margin = round(notional_value / lev, 2)

        # 7. Turnover & Cost Control (Buffer Bands)
        action, exec_contracts, b_lower, b_upper, b_width = self.buffer_manager.evaluate_buffer(
            target_contracts=final_contracts,
            current_contracts=current_position_contracts,
            contract_step=contract_step,
            current_price=current_price,
            min_notional=350.0
        )

        fee_saved = 0.0
        drift = abs(final_contracts - current_position_contracts)
        if action == 'HOLD' and drift > contract_step:
            fee_saved = round(drift * current_price * 0.0005, 3)

        fc_direction = 'BULLISH' if capped_fc > 1.0 else ('BEARISH' if capped_fc < -1.0 else 'NEUTRAL')
        summary = (
            f'Carver Forecast: {capped_fc:+.1f}/20.0 ({fc_direction}) | '
            f'Daily Vol Target: ${daily_cash_vol:,.1f} USDT | '
            f'Target Position: {final_contracts:+.3f} BTC (${optimal_margin:,.1f} Margin @ {lev}x) | '
            f'Risk Overlay: {combined_mult*100:.0f}% (DD: {dd_mult*100:.0f}%, VolShock: {vol_mult*100:.0f}%) | '
            f'Buffer Action: {action} (Inertia Band: [{b_lower:+.3f}, {b_upper:+.3f}])'
        )

        return CarverSystematicOutput(
            raw_forecast=raw_signal,
            forecast_scalar=scalar,
            scaled_forecast=scaled_fc,
            capped_forecast=capped_fc,
            annual_vol_target_pct=self.vol_targeter.annual_vol_target_pct,
            daily_cash_vol_target=daily_cash_vol,
            daily_price_vol_pct=daily_vol_pct,
            instrument_value_vol=inst_val_vol,
            rule_diversification_mult=rdm,
            effective_strategy_weight=strategy_weight,
            raw_optimal_contracts=raw_contracts,
            risk_overlay_multiplier=combined_mult,
            drawdown_multiplier=dd_mult,
            vol_shock_multiplier=vol_mult,
            margin_headroom_buffer=0.35,
            optimal_contracts=final_contracts,
            optimal_margin_usdt=optimal_margin,
            effective_leverage=lev,
            current_position_contracts=current_position_contracts,
            buffer_width_contracts=b_width,
            buffer_lower_bound=b_lower,
            buffer_upper_bound=b_upper,
            rebalance_action=action,
            contracts_to_execute=exec_contracts,
            fee_saved_usdt=fee_saved,
            summary_rationale=summary
        )
