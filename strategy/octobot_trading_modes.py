"""
OctoBot Trading Modes & Staged Take Profit Framework
Modeled after https://github.com/Drakkar-Software/OctoBot Trading Modes

Key Trading Modes:
1. Dip Analyser Mode:
   - Identifies local bottoms (panic dumps into institutional Demand Zones with Whale Absorption).
   - Precision limit entry with multi-stage staged take-profits (TP1, TP2, TP3).
2. Daily Trading Mode:
   - High-conviction trend following based on OctoBot Evaluator Matrix consensus.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
from strategy.octobot_matrix import MatrixConsensus


@dataclass
class StagedTakeProfit:
    tp1_price: float
    tp1_ratio: float = 0.40  # 40% position closed at TP1
    tp2_price: float = 0.0
    tp2_ratio: float = 0.35  # 35% position closed at TP2
    tp3_price: float = 0.0
    tp3_ratio: float = 0.25  # 25% position closed at TP3 (runner)


@dataclass
class OctoBotTradeSetup:
    mode_name: str              # "DIP_ANALYSER" or "DAILY_TRADING"
    direction: int              # 1 (Long), -1 (Short), 0 (None)
    entry_price: float
    stop_loss: float
    take_profit: float          # Primary / weighted TP
    staged_tp: StagedTakeProfit
    risk_reward_ratio: float
    confidence_score: float     # OctoBot Matrix score [-1.0, 1.0]
    rationale: str
    is_valid: bool = True


class DipAnalyserMode:
    """
    OctoBot Dip Analyser Trading Mode:
    Detects extreme capitulation into institutional Demand Zones + Absorption divergence.
    """
    def evaluate(
        self,
        current_price: float,
        matrix: MatrixConsensus,
        smc_data: Optional[Dict[str, Any]],
        indicators: Dict[str, Any],
        of_data: Optional[Dict[str, Any]],
        vwap_data: Optional[Dict[str, Any]]
    ) -> Optional[OctoBotTradeSetup]:
        if not smc_data or current_price <= 0:
            return None

        demand_zone = smc_data.get("demand_zone", [0.0, 0.0])
        supply_zone = smc_data.get("supply_zone", [0.0, 0.0])
        rsi = indicators.get("rsi", 50.0)
        atr = indicators.get("atr", current_price * 0.005)
        vwap_price = vwap_data.get("vwap", current_price) if vwap_data else current_price
        absorption = of_data.get("absorption_signal", "NONE") if of_data else "NONE"

        # -------------------------------------------------------------
        # 1. LONG DIP SETUP: Price in/near Demand Zone + Oversold / Absorption
        # -------------------------------------------------------------
        if demand_zone and demand_zone[0] > 0 and demand_zone[1] > 0:
            d_low, d_high = demand_zone
            # Within 0.3% of demand zone or inside
            in_demand = (d_low * 0.997 <= current_price <= d_high * 1.003)
            oversold_or_absorbing = (rsi <= 38.0 or absorption == "BULL_ABSORPTION" or matrix.matrix_score >= 0.40)

            if in_demand and oversold_or_absorbing:
                entry = round(current_price, 2)
                sl = round(min(d_low - (0.8 * atr), entry - (1.2 * atr)), 2)
                risk_dist = abs(entry - sl)
                if risk_dist <= 0:
                    risk_dist = entry * 0.005

                # Staged TPs:
                # TP1: Scalp first target at 1.2R
                tp1 = round(entry + (1.2 * risk_dist), 2)
                # TP2: Mean-reversion to VWAP equilibrium
                tp2 = round(max(tp1 + (0.5 * atr), vwap_price), 2) if vwap_price > entry else round(entry + (2.2 * risk_dist), 2)
                # TP3: Macro extension to Supply Zone or 3.5R
                tp3_target = supply_zone[0] if (supply_zone and supply_zone[0] > tp2) else (entry + (3.5 * risk_dist))
                tp3 = round(tp3_target, 2)

                weighted_tp = round(tp1 * 0.40 + tp2 * 0.35 + tp3 * 0.25, 2)
                rr = round((weighted_tp - entry) / risk_dist, 2)

                staged = StagedTakeProfit(tp1_price=tp1, tp2_price=tp2, tp3_price=tp3)
                rationale = (
                    f"🎯 [OCTOBOT DIP ANALYSER] Bắt đáy Demand Zone [${d_low:,.1f}-${d_high:,.1f}]. "
                    f"RSI={rsi:.1f} | Hấp thụ={absorption} | Matrix={matrix.matrix_score:+.2f}. "
                    f"TP Đa Tầng: TP1=${tp1:,.1f} (40%), TP2=${tp2:,.1f} (35%), TP3=${tp3:,.1f} (25%)."
                )

                return OctoBotTradeSetup(
                    mode_name="DIP_ANALYSER",
                    direction=1,
                    entry_price=entry,
                    stop_loss=sl,
                    take_profit=weighted_tp,
                    staged_tp=staged,
                    risk_reward_ratio=rr,
                    confidence_score=matrix.matrix_score,
                    rationale=rationale
                )

        # -------------------------------------------------------------
        # 2. SHORT TOP-BLOWOFF SETUP: Price in/near Supply Zone + Overbought / Absorption
        # -------------------------------------------------------------
        if supply_zone and supply_zone[0] > 0 and supply_zone[1] > 0:
            s_low, s_high = supply_zone
            in_supply = (s_low * 0.997 <= current_price <= s_high * 1.003)
            overbought_or_absorbing = (rsi >= 62.0 or absorption == "BEAR_ABSORPTION" or matrix.matrix_score <= -0.40)

            if in_supply and overbought_or_absorbing:
                entry = round(current_price, 2)
                sl = round(max(s_high + (0.8 * atr), entry + (1.2 * atr)), 2)
                risk_dist = abs(sl - entry)
                if risk_dist <= 0:
                    risk_dist = entry * 0.005

                tp1 = round(entry - (1.2 * risk_dist), 2)
                tp2 = round(min(tp1 - (0.5 * atr), vwap_price), 2) if vwap_price < entry else round(entry - (2.2 * risk_dist), 2)
                tp3_target = demand_zone[1] if (demand_zone and demand_zone[1] < tp2) else (entry - (3.5 * risk_dist))
                tp3 = round(tp3_target, 2)

                weighted_tp = round(tp1 * 0.40 + tp2 * 0.35 + tp3 * 0.25, 2)
                rr = round((entry - weighted_tp) / risk_dist, 2)

                staged = StagedTakeProfit(tp1_price=tp1, tp2_price=tp2, tp3_price=tp3)
                rationale = (
                    f"🎯 [OCTOBOT PEAK ANALYSER] Bán chặn Supply Zone [${s_low:,.1f}-${s_high:,.1f}]. "
                    f"RSI={rsi:.1f} | Hấp thụ={absorption} | Matrix={matrix.matrix_score:+.2f}. "
                    f"TP Đa Tầng: TP1=${tp1:,.1f} (40%), TP2=${tp2:,.1f} (35%), TP3=${tp3:,.1f} (25%)."
                )

                return OctoBotTradeSetup(
                    mode_name="DIP_ANALYSER",
                    direction=-1,
                    entry_price=entry,
                    stop_loss=sl,
                    take_profit=weighted_tp,
                    staged_tp=staged,
                    risk_reward_ratio=rr,
                    confidence_score=matrix.matrix_score,
                    rationale=rationale
                )

        return None


class DailyTradingMode:
    """
    OctoBot Daily Trading Mode:
    Executes trend-momentum trades with high Matrix Consensus agreement.
    """
    def evaluate(
        self,
        current_price: float,
        matrix: MatrixConsensus,
        indicators: Dict[str, Any],
        smc_data: Optional[Dict[str, Any]],
        vwap_data: Optional[Dict[str, Any]]
    ) -> Optional[OctoBotTradeSetup]:
        if not matrix.is_tradable or current_price <= 0:
            return None

        atr = indicators.get("atr", current_price * 0.005)
        entry = round(current_price, 2)
        direction = matrix.recommended_direction

        if direction == 1:  # LONG TREND
            sl = round(entry - (1.5 * atr), 2)
            risk_dist = entry - sl
            tp1 = round(entry + (1.5 * risk_dist), 2)
            tp2 = round(entry + (2.5 * risk_dist), 2)
            tp3 = round(entry + (4.0 * risk_dist), 2)
            weighted_tp = round(tp1 * 0.40 + tp2 * 0.35 + tp3 * 0.25, 2)
            rr = round((weighted_tp - entry) / risk_dist, 2)

            staged = StagedTakeProfit(tp1_price=tp1, tp2_price=tp2, tp3_price=tp3)
            rationale = (
                f"🚀 [OCTOBOT DAILY TREND] Long theo Ma Trận Đồng Thuận (Điểm={matrix.matrix_score:+.2f}). "
                f"Mục tiêu TP1: ${tp1:,.1f}, TP2: ${tp2:,.1f}, TP3: ${tp3:,.1f}."
            )

            return OctoBotTradeSetup(
                mode_name="DAILY_TRADING",
                direction=1,
                entry_price=entry,
                stop_loss=sl,
                take_profit=weighted_tp,
                staged_tp=staged,
                risk_reward_ratio=rr,
                confidence_score=matrix.matrix_score,
                rationale=rationale
            )

        elif direction == -1:  # SHORT TREND
            sl = round(entry + (1.5 * atr), 2)
            risk_dist = sl - entry
            tp1 = round(entry - (1.5 * risk_dist), 2)
            tp2 = round(entry - (2.5 * risk_dist), 2)
            tp3 = round(entry - (4.0 * risk_dist), 2)
            weighted_tp = round(tp1 * 0.40 + tp2 * 0.35 + tp3 * 0.25, 2)
            rr = round((entry - weighted_tp) / risk_dist, 2)

            staged = StagedTakeProfit(tp1_price=tp1, tp2_price=tp2, tp3_price=tp3)
            rationale = (
                f"🚀 [OCTOBOT DAILY TREND] Short theo Ma Trận Đồng Thuận (Điểm={matrix.matrix_score:+.2f}). "
                f"Mục tiêu TP1: ${tp1:,.1f}, TP2: ${tp2:,.1f}, TP3: ${tp3:,.1f}."
            )

            return OctoBotTradeSetup(
                mode_name="DAILY_TRADING",
                direction=-1,
                entry_price=entry,
                stop_loss=sl,
                take_profit=weighted_tp,
                staged_tp=staged,
                risk_reward_ratio=rr,
                confidence_score=matrix.matrix_score,
                rationale=rationale
            )

        return None


class OctoBotTradingCoordinator:
    """
    Unified OctoBot Mode Coordinator:
    Priority 1: Dip Analyser (Snipers local extremes at key levels)
    Priority 2: Daily Trading (Trend riders on strong consensus)
    """
    def __init__(self):
        self.dip_analyser = DipAnalyserMode()
        self.daily_trading = DailyTradingMode()

    def select_best_setup(
        self,
        current_price: float,
        matrix: MatrixConsensus,
        indicators: Dict[str, Any],
        smc_data: Optional[Dict[str, Any]],
        of_data: Optional[Dict[str, Any]],
        vwap_data: Optional[Dict[str, Any]]
    ) -> Tuple[str, Optional[OctoBotTradeSetup]]:
        # 1. Check Dip Analyser first (High probability mean reversion)
        dip_setup = self.dip_analyser.evaluate(current_price, matrix, smc_data, indicators, of_data, vwap_data)
        if dip_setup:
            return "DIP_ANALYSER", dip_setup

        # 2. Check Daily Trading (Trend following)
        daily_setup = self.daily_trading.evaluate(current_price, matrix, indicators, smc_data, vwap_data)
        if daily_setup:
            return "DAILY_TRADING", daily_setup

        return "STAND_ASIDE", None
