"""
Dynamic Leverage Strategy Engine - Multi-Factor Quantitative Model
Calculates optimal leverage dynamically based on:
1. Continuous Market Volatility (ATR % & Garman-Klass)
2. Strategy Horizon (Short-Term Scalp 4x-8x vs Long-Term Swing 2x-3x)
3. Order Flow Toxicity (VPIN & LOB Resilience)
4. Capital Drawdown & Jesse Win Expectancy
5. Liquidation Clearance Guarantee (Est. Liquidation Distance >= 3.0x Stop-Loss)
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class LeverageAdvice:
    leverage: int
    rationale: str
    est_liq_distance_pct: float
    safety_rating: str                  # 'CỰC KỲ AN TOÀN', 'AN TOÀN', 'TRUNG BÌNH', 'RỦI RO CAO'
    base_leverage: int = 5
    horizon: str = "SHORT_TERM"         # 'SHORT_TERM' (Scalp/Intraday) vs 'LONG_TERM' (Swing/Macro)
    vpin_impact: str = "CLEAN"          # 'CLEAN', 'ELEVATED_VPIN_CAP', 'CRITICAL_TOXIC_MIN'
    drawdown_impact: str = "NONE"       # 'NONE', 'MODERATE_CUT', 'DEFENSIVE_LOCK'
    liq_clearance_ratio: float = 3.0    # Ratio of Liq Distance to SL Distance (must be >= 3.0)
    factor_breakdown: Dict[str, Any] = field(default_factory=dict)


class DynamicLeverageEngine:
    def __init__(
        self,
        min_leverage: int = 1,
        max_leverage: int = 10,
        target_stop_pct: float = 0.015,
        min_clearance_ratio: float = 3.0
    ):
        self.min_leverage = min_leverage
        self.max_leverage = max_leverage
        self.target_stop_pct = target_stop_pct          # Default 1.5% stop distance
        self.min_clearance_ratio = min_clearance_ratio  # Liquidation >= 3.0x SL distance

    def calculate_optimal_leverage(
        self,
        current_price: float,
        atr: float,
        regime: str = "RANGING_SIDEWAY",
        confidence: int = 75,
        horizon: str = "SHORT_TERM",
        vpin: float = 0.20,
        market_resilience_pct: float = 85.0,
        current_drawdown_pct: float = 0.0,
        sl_distance_pct: float = 0.0,
        win_probability: int = 70,
        timeframe: str = "15m"
    ) -> LeverageAdvice:
        """
        Dynamically selects safe leverage via multi-factor quantitative weighting:
        - Continuous ATR volatility calibration
        - Horizon partitioning (Scalp up to 8x vs Swing capped at 3x-4x)
        - VPIN adverse selection protection
        - Drawdown discipline penalty
        - Mathematical liquidation clearance proof
        """
        if current_price <= 0:
            return LeverageAdvice(
                leverage=self.min_leverage,
                rationale="Giá hiện tại không hợp lệ; áp dụng đòn bẩy tối thiểu 1x phòng thủ.",
                est_liq_distance_pct=98.0,
                safety_rating="CỰC KỲ AN TOÀN",
                base_leverage=1
            )

        # 1. Deduce Strategy Horizon from timeframe if not explicitly supplied
        norm_tf = str(timeframe or "15m").strip().lower()
        if norm_tf in ("1m", "3m", "5m", "15m"):
            eff_horizon = "SHORT_TERM"
        else:
            eff_horizon = "LONG_TERM"
        if horizon in ("SHORT_TERM", "LONG_TERM"):
            eff_horizon = horizon

        # 2. Continuous Volatility Factor (ATR / Price) — dieu phoi theo
        # strategy.peak_lock (ATR thap -> don bay cao toi 10x; ATR cao -> 3x).
        # Lenh nho von $100 can don bay tot de phi khoi bao mon lai.
        atr_pct = (atr / current_price) if current_price > 0 else 0.01
        try:
            from strategy.peak_lock import leverage_for_atr as _lev_for_atr
            base_lev = int(_lev_for_atr(atr_pct, self.max_leverage))
        except Exception:
            base_lev = 8
        if atr_pct <= 0.0035:
            vol_note = f"Biến động siêu thấp (ATR={atr_pct*100:.2f}%)"
        elif atr_pct <= 0.0065:
            vol_note = f"Biến động thấp ổn định (ATR={atr_pct*100:.2f}%)"
        elif atr_pct <= 0.0110:
            vol_note = f"Biến động trung bình (ATR={atr_pct*100:.2f}%)"
        elif atr_pct <= 0.0180:
            vol_note = f"Biến động cao (ATR={atr_pct*100:.2f}%)"
        else:
            vol_note = f"Biến động mạnh/cực đại (ATR={atr_pct*100:.2f}%)"

        lev = base_lev
        notes = [vol_note]

        # 3. Horizon Partitioning Cap
        # Swing / Macro positions hold through larger price swings, so leverage must stay low (2x-3x)
        # to ensure liquidation price is 25%-45% away from entry.
        if eff_horizon == "LONG_TERM":
            lev = min(lev, 4)
            if lev > 3 and atr_pct > 0.006:
                lev = 3
            notes.append(f"Kỳ hạn Dài Hạn (Swing/Macro {norm_tf}) khống chế đòn bẩy an toàn {lev}x")
        else:
            # Scalp / Intraday has tight SL (< 0.8%), allowing higher capital velocity
            if confidence >= 80 and win_probability >= 75 and atr_pct <= 0.006:
                lev = min(self.max_leverage, lev + 1)
            notes.append(f"Kỳ hạn Ngắn Hạn (Scalp/Intraday {norm_tf}) linh hoạt {lev}x")

        # 4. Market Regime & AI Confidence Alignment
        reg = str(regime or "RANGING_SIDEWAY").upper()
        if reg in ("VOLATILE_PANIC", "EXTREME_CHOP"):
            lev = min(lev, 2)
            notes.append(f"Chế độ thị trường hoảng loạn/nhiễu sóng ({reg}) ép về {lev}x")
        elif reg in ("TRENDING_BULL", "TRENDING_BEAR") and confidence >= 80 and eff_horizon == "SHORT_TERM":
            lev = min(self.max_leverage, max(lev, 5))
            notes.append(f"Xu hướng rõ rệt ({confidence}% tin cậy)")
        elif reg == "RANGING_SIDEWAY":
            lev = min(lev, 5 if eff_horizon == "SHORT_TERM" else 3)

        # 5. Order Flow Toxicity (VPIN & Market Resilience)
        vpin_impact = "CLEAN"
        if vpin >= 0.85:
            vpin_impact = "CRITICAL_TOXIC_MIN"
            lev = min(lev, 2)
            notes.append(f"🚨 Dòng tiền độc hại cực lớn VPIN={vpin:.2f} -> Khóa đòn bẩy tối thiểu {lev}x")
        elif vpin >= 0.70:
            vpin_impact = "ELEVATED_VPIN_CAP"
            lev = min(lev, 3)
            notes.append(f"⚠️ VPIN cảnh báo={vpin:.2f} -> Giới hạn trần {lev}x")
        elif market_resilience_pct < 35.0:
            lev = min(lev, 3)
            notes.append(f"💧 Thanh khoản mỏng (Resilience={market_resilience_pct:.0f}%) -> Hạ còn {lev}x")

        # 6. Portfolio Drawdown & Consecutive Losses Discipline
        drawdown_impact = "NONE"
        if current_drawdown_pct >= 5.0:
            drawdown_impact = "DEFENSIVE_LOCK"
            lev = self.min_leverage
            notes.append(f"🛡️ Sụt giảm vốn chạm ngưỡng 5% (DD={current_drawdown_pct:.1f}%) -> Khóa đòn bẩy {lev}x bảo toàn vốn")
        elif current_drawdown_pct >= 2.5:
            drawdown_impact = "MODERATE_CUT"
            lev = max(self.min_leverage, min(lev - 1, 3))
            notes.append(f"Phòng thủ rủi ro (DD={current_drawdown_pct:.1f}%) -> Giảm đòn bẩy còn {lev}x")

        # 7. Liquidation Clearance Guarantee (Liquidation Buffer >= 3.0x SL Distance)
        # Dung cong thuc Isolated that: liq_dist = (1/lev - MMR - phi) * 100%,
        # voi MMR tra theo bac notional uoc tinh (entry * lev) thay vi he so 0.98 cung.
        from risk.risk_manager import FuturesRiskManager
        effective_sl_pct = max(sl_distance_pct, self.target_stop_pct)
        target_clearance_pct = effective_sl_pct * self.min_clearance_ratio * 100.0

        # Iteratively reduce leverage if liquidation distance is too close to Stop Loss
        while lev > self.min_leverage:
            est_notional = current_price * lev
            est_mmr = FuturesRiskManager.mmr_for_notional(est_notional)
            est_liq_dist = max(0.0, (1.0 / lev - est_mmr - 0.0005)) * 100.0
            if est_liq_dist >= target_clearance_pct:
                break
            lev -= 1

        final_lev = max(self.min_leverage, min(self.max_leverage, lev))
        final_mmr = FuturesRiskManager.mmr_for_notional(current_price * final_lev)
        final_liq_distance = round(max(0.0, (1.0 / final_lev - final_mmr - 0.0005)) * 100.0, 1)
        clearance_ratio = round(final_liq_distance / max(0.1, effective_sl_pct * 100.0), 2)

        if final_liq_distance >= 28.0:
            safety_rating = "CỰC KỲ AN TOÀN"
        elif final_liq_distance >= 18.0:
            safety_rating = "AN TOÀN"
        elif final_liq_distance >= 10.0:
            safety_rating = "TRUNG BÌNH"
        else:
            safety_rating = "RỦI RO CAO"

        rationale = f"Đòn bẩy {final_lev}x ({eff_horizon}): " + " | ".join(notes) + f" | Biên thanh lý an toàn: {final_liq_distance}% (gấp {clearance_ratio}x khoảng dừng lỗ)."

        return LeverageAdvice(
            leverage=final_lev,
            rationale=rationale,
            est_liq_distance_pct=final_liq_distance,
            safety_rating=safety_rating,
            base_leverage=base_lev,
            horizon=eff_horizon,
            vpin_impact=vpin_impact,
            drawdown_impact=drawdown_impact,
            liq_clearance_ratio=clearance_ratio,
            factor_breakdown={
                "atr_pct": round(atr_pct * 100, 3),
                "base_leverage": base_lev,
                "horizon": eff_horizon,
                "timeframe": norm_tf,
                "regime": reg,
                "vpin": round(vpin, 3),
                "market_resilience_pct": round(market_resilience_pct, 1),
                "current_drawdown_pct": round(current_drawdown_pct, 2),
                "sl_distance_pct": round(effective_sl_pct * 100, 2),
                "clearance_ratio": clearance_ratio,
            }
        )
