"""
OctoBot-Grade Tentacle Evaluator Matrix Architecture
Modeled after https://github.com/Drakkar-Software/OctoBot and OctoBot-Tentacles

Key Concepts:
1. Tentacle Evaluator Tree: Independent evaluators analyzing distinct market dimensions.
   - Technical (TA): RSI, MACD, EMA momentum.
   - Real-Time Order Flow (RT): CVD delta, Taker Buy/Sell ratio, Absorption divergence.
   - Smart Money Concepts (SMC): Institutional Order Blocks, FVG, Liquidity Sweeps.
   - Anchored VWAP Valuation: Mean-reversion distance to VWAP equilibrium.
   - Multi-Timeframe (MTF): Trend alignment across 1m, 5m, 15m, 1h.
2. Evaluator Matrix: Computes a confidence-weighted consensus score in [-1.0, +1.0].
   - Score > +0.65: Strong Bullish -> Trigger Long
   - Score < -0.65: Strong Bearish -> Trigger Short
   - -0.30 <= Score <= +0.30: Neutral / Chop -> Filter out trades
"""
import math
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple


@dataclass
class TentacleEvaluation:
    tentacle_name: str          # e.g., "TA", "OrderFlow", "SMC", "VWAP", "MTF"
    score: float                # Normalized score from -1.0 (Strong Bearish) to +1.0 (Strong Bullish)
    weight: float               # Confidence weight assigned to this tentacle [0.5, 2.0]
    detail: str                 # Human-readable rationale
    status_label: str           # "BULLISH 🟢", "BEARISH 🔴", "NEUTRAL ⚪"


@dataclass
class MatrixConsensus:
    matrix_score: float         # Consolidated weighted score in [-1.0, +1.0]
    confidence_pct: float       # Percentage confidence [0% - 100%]
    consensus_state: str        # 'STRONG_BULLISH', 'WEAK_BULLISH', 'NEUTRAL', 'WEAK_BEARISH', 'STRONG_BEARISH'
    recommended_direction: int  # 1 for Long, -1 for Short, 0 for Stand Aside
    is_tradable: bool           # True if |matrix_score| >= confidence_threshold
    threshold: float            # Required threshold (e.g. 0.60)
    tentacles: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    summary_reason: str = ""


class TATentacleEvaluator:
    """
    Technical Analysis Tentacle Evaluator:
    Combines RSI oversold/overbought conditions, EMA moving average alignment, and MACD momentum.
    """
    def __init__(self, weight: float = 1.0):
        self.weight = weight

    def evaluate(self, indicators: Dict[str, Any]) -> TentacleEvaluation:
        rsi = indicators.get("rsi", 50.0)
        ema_fast = indicators.get("ema_fast", 0.0)
        ema_slow = indicators.get("ema_slow", 0.0)
        macd = indicators.get("macd", 0.0)
        macd_signal = indicators.get("macd_signal", 0.0)

        sub_scores: List[float] = []

        # 1. RSI Score
        if rsi <= 25.0:
            rsi_score = 1.0  # Extremely oversold -> Bullish bounce
        elif rsi <= 35.0:
            rsi_score = 0.65
        elif rsi >= 75.0:
            rsi_score = -1.0  # Extremely overbought -> Bearish reversal
        elif rsi >= 65.0:
            rsi_score = -0.65
        else:
            # Linear map around 50
            rsi_score = (rsi - 50.0) / 25.0  # [-0.6, +0.6]
        sub_scores.append(rsi_score)

        # 2. EMA Trend
        if ema_fast > 0 and ema_slow > 0:
            diff_pct = (ema_fast - ema_slow) / ema_slow * 100.0
            ema_score = max(-1.0, min(1.0, diff_pct / 0.3))  # 0.3% gap gives +/- 1.0
            sub_scores.append(ema_score)

        # 3. MACD Momentum
        if macd != 0.0 or macd_signal != 0.0:
            hist = macd - macd_signal
            hist_score = 0.5 if hist > 0 else -0.5
            sub_scores.append(hist_score)

        avg_score = sum(sub_scores) / len(sub_scores) if sub_scores else 0.0
        avg_score = max(-1.0, min(1.0, round(avg_score, 2)))

        if avg_score >= 0.4:
            status = "BULLISH 🟢"
        elif avg_score <= -0.4:
            status = "BEARISH 🔴"
        else:
            status = "NEUTRAL ⚪"

        detail = f"RSI: {rsi:.1f} | EMA fast/slow: {ema_fast:.1f}/{ema_slow:.1f}"
        return TentacleEvaluation("TA", avg_score, self.weight, detail, status)


class OrderFlowTentacleEvaluator:
    """
    Real-Time Order Flow Tentacle Evaluator:
    Analyzes Cumulative Volume Delta (CVD), Taker Buy/Sell volume ratio, and Absorption Divergences.
    """
    def __init__(self, weight: float = 1.3):
        self.weight = weight

    def evaluate(self, of_telemetry: Optional[Dict[str, Any]]) -> TentacleEvaluation:
        if not of_telemetry:
            return TentacleEvaluation("OrderFlow", 0.0, self.weight, "Chưa có dữ liệu Order Flow", "NEUTRAL ⚪")

        buy_ratio = of_telemetry.get("buy_ratio", 50.0)
        cvd_delta_60s = of_telemetry.get("cvd_delta_60s", 0.0)
        absorption = of_telemetry.get("absorption_signal", "NONE")

        # Base score from buy ratio (50% = 0.0, 70% = +1.0, 30% = -1.0)
        ratio_score = (buy_ratio - 50.0) / 20.0
        ratio_score = max(-1.0, min(1.0, ratio_score))

        # Institutional absorption bonus (very high predictive value)
        absorption_bonus = 0.0
        if absorption == "BULL_ABSORPTION":
            absorption_bonus = 0.45  # Whales absorbing retail market dump
        elif absorption == "BEAR_ABSORPTION":
            absorption_bonus = -0.45  # Whales distributing into retail FOMO

        final_score = max(-1.0, min(1.0, round(ratio_score * 0.6 + absorption_bonus, 2)))

        if final_score >= 0.35:
            status = "BULLISH 🟢"
        elif final_score <= -0.35:
            status = "BEARISH 🔴"
        else:
            status = "NEUTRAL ⚪"

        detail = f"Taker Mua: {buy_ratio:.1f}% | CVD 60s: {cvd_delta_60s:+.3f} BTC | Hấp thụ: {absorption}"
        return TentacleEvaluation("OrderFlow", final_score, self.weight, detail, status)


class SMCTentacleEvaluator:
    """
    Smart Money Concepts Tentacle Evaluator:
    Evaluates current price proximity to institutional Order Blocks (Demand/Supply Zones),
    Fair Value Gaps (FVG), and recent Liquidity Sweeps.
    """
    def __init__(self, weight: float = 1.2):
        self.weight = weight

    def evaluate(self, current_price: float, smc_data: Optional[Dict[str, Any]]) -> TentacleEvaluation:
        if not smc_data or current_price <= 0:
            return TentacleEvaluation("SMC", 0.0, self.weight, "Chưa xác định cấu trúc SMC", "NEUTRAL ⚪")

        structure = smc_data.get("structure", "RANGING")
        demand_zone = smc_data.get("demand_zone", [0.0, 0.0])
        supply_zone = smc_data.get("supply_zone", [0.0, 0.0])
        sweep = smc_data.get("liquidity_sweep", "NONE")

        score = 0.0

        # Proximity to Demand Zone (Buy Zone)
        if demand_zone and demand_zone[0] > 0 and demand_zone[1] > 0:
            d_low, d_high = demand_zone
            if d_low <= current_price <= d_high * 1.0025:
                score += 0.65  # Price inside or touching Demand Zone

        # Proximity to Supply Zone (Sell Zone)
        if supply_zone and supply_zone[0] > 0 and supply_zone[1] > 0:
            s_low, s_high = supply_zone
            if s_low * 0.9975 <= current_price <= s_high:
                score -= 0.65  # Price inside or touching Supply Zone

        # Liquidity Sweep confirmation
        if sweep == "SWEEP_LOW":
            score += 0.40  # Bear trap / stop hunt low -> strongly bullish
        elif sweep == "SWEEP_HIGH":
            score -= 0.40  # Bull trap / stop hunt high -> strongly bearish

        # Trend structure bias
        if structure == "BULLISH_TREND":
            score += 0.20
        elif structure == "BEARISH_TREND":
            score -= 0.20

        final_score = max(-1.0, min(1.0, round(score, 2)))

        if final_score >= 0.35:
            status = "BULLISH 🟢"
        elif final_score <= -0.35:
            status = "BEARISH 🔴"
        else:
            status = "NEUTRAL ⚪"

        detail = f"Cấu trúc: {structure} | Demand: {demand_zone} | Sweep: {sweep}"
        return TentacleEvaluation("SMC", final_score, self.weight, detail, status)


class VWAPValuationTentacleEvaluator:
    """
    Institutional Anchored VWAP Valuation Tentacle Evaluator:
    Determines whether the market is at a deep discount (cheap to buy) or premium (expensive to sell).
    """
    def __init__(self, weight: float = 1.0):
        self.weight = weight

    def evaluate(self, vwap_data: Optional[Dict[str, Any]]) -> TentacleEvaluation:
        if not vwap_data:
            return TentacleEvaluation("VWAP", 0.0, self.weight, "Chưa có dữ liệu VWAP", "NEUTRAL ⚪")

        status_str = vwap_data.get("vwap_status", "EQUILIBRIUM_FAIR")
        dist_sigma = vwap_data.get("dist_sigma", 0.0)

        # Negative sigma = Price below VWAP (Discount zone) -> Bullish mean reversion
        # Positive sigma = Price above VWAP (Premium zone) -> Bearish mean reversion
        if "DISCOUNT" in status_str or dist_sigma < -1.5:
            score = 0.70
            status = "BULLISH 🟢"
            detail = f"Vùng Chiết Khấu Rẻ (Discount {dist_sigma:+.2f}σ)"
        elif "PREMIUM" in status_str or dist_sigma > 1.5:
            score = -0.70
            status = "BEARISH 🔴"
            detail = f"Vùng Định Giá Cao (Premium {dist_sigma:+.2f}σ)"
        else:
            score = 0.0
            status = "NEUTRAL ⚪"
            detail = f"Vùng Giá Trị Cân Bằng (Equilibrium {dist_sigma:+.2f}σ)"

        return TentacleEvaluation("VWAP", score, self.weight, detail, status)


class MultiTimeframeTentacleEvaluator:
    """
    Multi-Timeframe Alignment Tentacle Evaluator:
    Checks macro alignment across 1m, 5m, 15m, 1h timeframes.
    """
    def __init__(self, weight: float = 1.1):
        self.weight = weight

    def evaluate(self, mtf_consensus: Optional[Dict[str, Any]]) -> TentacleEvaluation:
        if not mtf_consensus:
            return TentacleEvaluation("MTF", 0.0, self.weight, "Đa khung đang đồng bộ...", "NEUTRAL ⚪")

        score = mtf_consensus.get("score", 0.0)  # already normalized [-1.0, 1.0]
        consensus = mtf_consensus.get("consensus", "NEUTRAL")
        tfs = mtf_consensus.get("timeframes", {})

        score = max(-1.0, min(1.0, round(score, 2)))

        if score >= 0.4:
            status = "BULLISH 🟢"
        elif score <= -0.4:
            status = "BEARISH 🔴"
        else:
            status = "NEUTRAL ⚪"

        detail = f"Đồng thuận {consensus} across {len(tfs)} khung thời gian"
        return TentacleEvaluation("MTF", score, self.weight, detail, status)


class OctoBotMatrixEngine:
    """
    OctoBot Core Evaluator Matrix Engine:
    Gathers all 5 Tentacles, computes weighted consensus, applies confidence threshold,
    and returns a structured decision matrix for algorithmic execution.
    """
    def __init__(self, confidence_threshold: float = 0.55):
        self.confidence_threshold = confidence_threshold
        self.tentacle_ta = TATentacleEvaluator(weight=1.0)
        self.tentacle_of = OrderFlowTentacleEvaluator(weight=1.4)
        self.tentacle_smc = SMCTentacleEvaluator(weight=1.3)
        self.tentacle_vwap = VWAPValuationTentacleEvaluator(weight=1.0)
        self.tentacle_mtf = MultiTimeframeTentacleEvaluator(weight=1.1)

    def evaluate_matrix(
        self,
        current_price: float,
        indicators: Dict[str, Any],
        order_flow_telemetry: Optional[Dict[str, Any]],
        smc_data: Optional[Dict[str, Any]],
        vwap_data: Optional[Dict[str, Any]],
        mtf_consensus: Optional[Dict[str, Any]]
    ) -> MatrixConsensus:
        evals: List[TentacleEvaluation] = [
            self.tentacle_ta.evaluate(indicators),
            self.tentacle_of.evaluate(order_flow_telemetry),
            self.tentacle_smc.evaluate(current_price, smc_data),
            self.tentacle_vwap.evaluate(vwap_data),
            self.tentacle_mtf.evaluate(mtf_consensus)
        ]

        total_weighted_score = sum(e.score * e.weight for e in evals)
        total_weight = sum(e.weight for e in evals)

        matrix_score = total_weighted_score / total_weight if total_weight > 0 else 0.0
        matrix_score = max(-1.0, min(1.0, round(matrix_score, 3)))
        confidence_pct = round(abs(matrix_score) * 100.0, 1)

        # State classification
        if matrix_score >= self.confidence_threshold:
            state = "STRONG_BULLISH"
            direction = 1
            is_tradable = True
        elif matrix_score >= 0.25:
            state = "WEAK_BULLISH"
            direction = 1
            is_tradable = False
        elif matrix_score <= -self.confidence_threshold:
            state = "STRONG_BEARISH"
            direction = -1
            is_tradable = True
        elif matrix_score <= -0.25:
            state = "WEAK_BEARISH"
            direction = -1
            is_tradable = False
        else:
            state = "NEUTRAL"
            direction = 0
            is_tradable = False

        tentacles_dict = {}
        for e in evals:
            tentacles_dict[e.tentacle_name] = {
                "score": e.score,
                "weight": e.weight,
                "status_label": e.status_label,
                "detail": e.detail
            }

        summary = (
            f"🐙 [OCTOBOT MATRIX] Điểm đồng thuận: {matrix_score:+.2f} ({confidence_pct:.0f}% tin cậy). "
            f"Trạng thái: {state}."
        )

        return MatrixConsensus(
            matrix_score=matrix_score,
            confidence_pct=confidence_pct,
            consensus_state=state,
            recommended_direction=direction,
            is_tradable=is_tradable,
            threshold=self.confidence_threshold,
            tentacles=tentacles_dict,
            summary_reason=summary
        )
