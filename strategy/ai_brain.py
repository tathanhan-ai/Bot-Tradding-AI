"""
AI Quant Brain - Institutional Smart Money Concepts (SMC) & Multi-Timeframe Engine
Integrates:
1. Smart Money Concepts: Order Blocks (OB), Fair Value Gaps (FVG), Liquidity Sweeps, CHoCH/BOS
2. Institutional Anchored VWAP + Standard Deviation Bands (±1σ, ±2σ)
3. Multi-Timeframe Confluence Matrix (1m, 5m, 15m, 1h)
4. Eliminates naive single-direction retail bias; executes institutional liquidity traps and absorption
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

from strategy.smart_money_concepts import SmartMoneyEngine, SMCAnalysisResult, OrderBlock, FairValueGap, LiquiditySweep
from strategy.institutional_vwap import InstitutionalVWAPEngine, VWAPBandResult
from strategy.quant_skills_brain import QuantSkillsBrain


@dataclass
class AIRegimeVerdict:
    regime: str                 # 'TRENDING_BULL', 'TRENDING_BEAR', 'RANGING_SIDEWAY', 'VOLATILE_PANIC'
    regime_display: str         # Human readable in Vietnamese
    recommended_strategy: str   # 'GRID_BOT', 'TREND_BREAKOUT', 'TWO_WAY_RANGE', 'STAND_ASIDE', 'SMC_ORDER_BLOCK'
    strategy_display: str
    bull_score: int             # 0 - 100
    bear_score: int             # 0 - 100
    confidence: int             # 0 - 100%
    rationale: str              # Detailed reasoning narrative
    risk_level: str             # 'THẤP', 'TRUNG BÌNH', 'CAO', 'RẤT CAO'
    support_price: float = 0.0
    resistance_price: float = 0.0
    suggested_limit_buy: float = 0.0
    suggested_limit_sell: float = 0.0
    updated_time: str = ""
    mtf_radar: Dict[str, Any] = field(default_factory=dict)
    active_timeframe: str = "15m"

    # Quant Skills Intelligence Core Fields (from claude-trading-skills)
    hurst_exponent: float = 0.50
    hurst_regime: str = "RANDOM_WALK"
    garman_klass_vol: float = 0.0
    volatility_regime: str = "NORMAL"

    # Institutional SMC & Order Flow Confluence Fields
    smc_structure: str = "RANGING"
    smc_bias: str = "NEUTRAL"
    smc_rationale: str = ""
    vwap_fair_price: float = 0.0
    vwap_status: str = "EQUILIBRIUM_FAIR"
    demand_zone: Optional[Tuple[float, float]] = None
    supply_zone: Optional[Tuple[float, float]] = None
    active_fvgs: List[Dict[str, Any]] = field(default_factory=list)
    last_sweep_info: str = ""
    dist_demand_pct: float = 0.0
    dist_supply_pct: float = 0.0
    is_testing_ob: bool = False
    tested_ob_type: str = "NONE"
    dist_vwap_pct: float = 0.0
    dist_vwap_usd: float = 0.0
    vwap_sigma_dev: float = 0.0


class AIQuantBrain:
    def __init__(self):
        self.last_verdict: Optional[AIRegimeVerdict] = None
        self.smc_engine = SmartMoneyEngine(atr_mult=1.2)
        self.vwap_engine = InstitutionalVWAPEngine()

    def calculate_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculates standard technical indicators for any timeframe"""
        if df is None or len(df) < 25:
            return {"rsi": 50.0, "atr": 0.0, "adx": 0.0, "ema20": 0.0, "ema50": 0.0, "chop": 50.0}

        close = df["close"]
        high = df["high"]
        low = df["low"]

        # EMAs
        ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
        ema50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])

        # RSI
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean().iloc[-1]
        loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean().iloc[-1]
        rsi = float(100 - (100 / (1 + (gain / (loss if loss > 0 else 1e-6)))))

        # ATR
        tr = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
        atr = float(tr.ewm(alpha=1/14, adjust=False).mean().iloc[-1])

        # ADX Wilder (Directional Movement lam muot dung: +DI/-DI smoothed 14 ky roi moi ra DX/ADX)
        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
        minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)
        atr_w = tr.ewm(alpha=1 / 14, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr_w.replace(0, np.nan))
        minus_di = 100 * (minus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr_w.replace(0, np.nan))
        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan))
        adx_raw = float(dx.ewm(alpha=1 / 14, adjust=False).mean().iloc[-1])
        adx = float(min(100.0, max(0.0, adx_raw))) if np.isfinite(adx_raw) else 0.0

        # Choppiness Index (CHOP)
        tr_14 = tr.rolling(14).sum().iloc[-1]
        hh_14 = high.rolling(14).max().iloc[-1]
        ll_14 = low.rolling(14).min().iloc[-1]
        range_14 = hh_14 - ll_14
        chop = 50.0
        if range_14 > 0 and tr_14 > 0:
            chop = float(100 * (np.log10(tr_14 / range_14) / np.log10(14)))

        # Quant Skills Integration (Hurst Exponent & Garman-Klass Volatility)
        hurst_res = QuantSkillsBrain.calculate_hurst_exponent(close)
        gk_res = QuantSkillsBrain.calculate_garman_klass_volatility(df)

        return {
            "rsi": round(rsi, 1),
            "atr": round(atr, 2),
            "adx": round(adx, 1),
            "ema20": round(ema20, 2),
            "ema50": round(ema50, 2),
            "chop": round(chop, 1),
            "hurst": hurst_res.hurst_exponent,
            "hurst_regime": hurst_res.regime_type,
            "gk_vol": gk_res.garman_klass_annualized,
            "vol_regime": gk_res.volatility_regime,
            "support": float(low.iloc[-24:].min()),
            "resistance": float(high.iloc[-24:].max())
        }

    def analyze(
        self,
        data_map: Dict[str, pd.DataFrame],
        current_price: float,
        spread: float = 0.1,
        active_timeframe: str = "15m"
    ) -> AIRegimeVerdict:
        now_str = datetime.now().strftime("%H:%M:%S")

        # 1. Multi-Timeframe Indicator Extraction (1m, 5m, 15m, 1h, 1w, 1M)
        tf_data = {}
        for tf in ["1m", "5m", "15m", "1h", "1w", "1M"]:
            df_tf = data_map.get(tf)
            if df_tf is not None and not df_tf.empty:
                tf_data[tf] = self.calculate_indicators(df_tf)
            else:
                tf_data[tf] = {
                    "rsi": 50.0, "atr": current_price * 0.005, "adx": 15.0, "chop": 50.0,
                    "hurst": 0.50, "hurst_regime": "RANDOM_WALK", "gk_vol": 0.0, "vol_regime": "NORMAL",
                    "support": current_price * 0.99, "resistance": current_price * 1.01
                }

        # 2. Institutional Smart Money Concepts (SMC) Analysis on Active Timeframe
        df_active = data_map.get(active_timeframe)
        if df_active is None or df_active.empty:
            df_active = data_map.get("15m")

        smc_res: SMCAnalysisResult = self.smc_engine.analyze(df_active, current_price)
        vwap_res: Optional[VWAPBandResult] = self.vwap_engine.calculate(df_active, current_price)

        # 3. Build Multi-Timeframe Radar
        radar = {}
        bull_counts = 0
        bear_counts = 0
        chop_counts = 0

        for tf in ["1m", "5m", "15m", "1h", "1w", "1M"]:
            ind = tf_data[tf]
            rsi = ind["rsi"]
            adx = ind["adx"]
            chop = ind.get("chop", 50.0)

            if chop >= 52.0 or adx < 22.0 or (43 <= rsi <= 57):
                trend_str = "SIDEWAY / CHOP ⚪"
                bias = "NEUTRAL"
                chop_counts += 1
            elif rsi > 57 and ind.get("ema20", 0) >= ind.get("ema50", 0):
                trend_str = "TĂNG (BULL) 🟢"
                bias = "BULL"
                bull_counts += 1
            elif rsi < 43 and ind.get("ema20", 0) <= ind.get("ema50", 0):
                trend_str = "GIẢM (BEAR) 🔴"
                bias = "BEAR"
                bear_counts += 1
            else:
                trend_str = "TÍCH LŨY ⚪"
                bias = "NEUTRAL"
                chop_counts += 1

            radar[tf] = {
                "trend": trend_str,
                "bias": bias,
                "rsi": rsi,
                "adx": adx,
                "chop": chop,
                "hurst": ind.get("hurst", 0.50),
                "hurst_regime": ind.get("hurst_regime", "RANDOM_WALK"),
                "gk_vol": ind.get("gk_vol", 0.0),
                "vol_regime": ind.get("vol_regime", "NORMAL"),
                "support": round(ind.get("support", current_price * 0.995), 1),
                "resistance": round(ind.get("resistance", current_price * 1.005), 1)
            }

        # 4. Institutional Support & Resistance from SMC Order Blocks & VWAP
        primary_ind = tf_data.get(active_timeframe) or tf_data.get("15m")
        atr = primary_ind.get("atr", current_price * 0.006)
        adx = primary_ind.get("adx", 18.0)
        chop = primary_ind.get("chop", 55.0)

        # Institutional Support = Top of Bullish Order Block / Demand Zone, fallback to VWAP Lower Band, fallback to rolling min
        if smc_res.nearest_demand_zone:
            support_price = smc_res.nearest_demand_zone[1]
        elif vwap_res and vwap_res.lower_band_1 < current_price:
            support_price = vwap_res.lower_band_1
        else:
            support_price = primary_ind.get("support", current_price * 0.995)

        # Institutional Resistance = Bottom of Bearish Order Block / Supply Zone, fallback to VWAP Upper Band, fallback to rolling max
        if smc_res.nearest_supply_zone:
            resistance_price = smc_res.nearest_supply_zone[0]
        elif vwap_res and vwap_res.upper_band_1 > current_price:
            resistance_price = vwap_res.upper_band_1
        else:
            resistance_price = primary_ind.get("resistance", current_price * 1.005)

        suggested_limit_buy = round(support_price + (0.1 * atr), 1)
        suggested_limit_sell = round(resistance_price - (0.1 * atr), 1)

        # Format FVGs for UI
        fvg_list = [
            {"type": f.type, "top": f.top_price, "bottom": f.bottom_price, "mid": f.mid_price}
            for f in smc_res.active_fvgs
        ]
        sweep_text = (
            f"{'🔴 Quét Đỉnh' if smc_res.last_sweep.type == 'BEARISH_SWEEP' else '🟢 Quét Đáy'} ${smc_res.last_sweep.swept_price:,.1f}"
            if smc_res.last_sweep else "Không có bẫy gần đây"
        )

        # 5. Hierarchical Multi-Timeframe Synthesis & Macro Anchor Check
        macro_w_bias = radar.get("1w", {}).get("bias", "NEUTRAL")
        macro_m_bias = radar.get("1M", {}).get("bias", "NEUTRAL")
        
        if macro_w_bias == "BULL" and macro_m_bias == "BULL":
            macro_cycle = "SECULAR_BULL"
            macro_desc = "Đại Chu Kỳ Tăng (1W & 1M Xanh 🟢)"
        elif macro_w_bias == "BEAR" and macro_m_bias == "BEAR":
            macro_cycle = "SECULAR_BEAR"
            macro_desc = "Đại Chu Kỳ Giảm (1W & 1M Đỏ 🔴)"
        elif macro_w_bias == "BULL":
            macro_cycle = "WEEKLY_BULL"
            macro_desc = "Xu Hướng Tuần Tăng (1W Xanh 🟢)"
        elif macro_w_bias == "BEAR":
            macro_cycle = "WEEKLY_BEAR"
            macro_desc = "Xu Hướng Tuần Giảm (1W Đỏ 🔴)"
        else:
            macro_cycle = "SECULAR_CHOP"
            macro_desc = "Chu Kỳ Vĩ Mô Tích Lũy ⚪"

        # 6. Institutional Confluence Decision Matrix
        # CASE 1: Liquidity Sweep Detected (Institutional Trapping & CHoCH)
        if smc_res.last_sweep and smc_res.last_sweep.type == "BEARISH_SWEEP":
            is_macro_aligned = macro_cycle in ("SECULAR_BEAR", "WEEKLY_BEAR")
            bear_sc = 85 if is_macro_aligned else 70
            bull_sc = 100 - bear_sc
            conf = 95 if is_macro_aligned else 84
            macro_note = " 🩸 Thuận đại chu kỳ vĩ mô xả hàng." if is_macro_aligned else " ⚠️ Sóng chỉnh ngắn hạn trong chu kỳ tăng tuần."

            verdict = AIRegimeVerdict(
                regime="TRENDING_BEAR",
                regime_display=f"BẪY THANH KHOẢN ĐỈNH (SMC {active_timeframe})",
                recommended_strategy="SMC_ORDER_BLOCK",
                strategy_display="Săn Bẫy Cá Mập (CHoCH Short)",
                bull_score=bull_sc, bear_score=bear_sc, confidence=conf,
                rationale=(
                    f"[{now_str}] Cá mập vừa quét râu vượt đỉnh ${smc_res.last_sweep.swept_price:,.1f} kích hoạt Stop Loss phe Short "
                    f"rồi đạp giá đóng cửa ngược lại. CHoCH đảo chiều Giảm!{macro_note} Ưu tiên Short tại Bearish OB ${resistance_price:,.1f}."
                ),
                risk_level="TRUNG BÌNH",
                support_price=support_price, resistance_price=resistance_price,
                suggested_limit_buy=suggested_limit_buy, suggested_limit_sell=suggested_limit_sell,
                updated_time=now_str, mtf_radar=radar, active_timeframe=active_timeframe,
                smc_structure=smc_res.market_structure, smc_bias=smc_res.institutional_bias,
                smc_rationale=smc_res.bias_rationale,
                vwap_fair_price=vwap_res.vwap if vwap_res else current_price,
                vwap_status=vwap_res.valuation_status if vwap_res else "EQUILIBRIUM_FAIR",
                demand_zone=smc_res.nearest_demand_zone, supply_zone=smc_res.nearest_supply_zone,
                active_fvgs=fvg_list, last_sweep_info=sweep_text
            )

        elif smc_res.last_sweep and smc_res.last_sweep.type == "BULLISH_SWEEP":
            is_macro_aligned = macro_cycle in ("SECULAR_BULL", "WEEKLY_BULL")
            bull_sc = 85 if is_macro_aligned else 70
            bear_sc = 100 - bull_sc
            conf = 95 if is_macro_aligned else 84
            macro_note = " 🚀 Thuận đại chu kỳ tăng trưởng vĩ mô." if is_macro_aligned else " ⚠️ Sóng hồi kỹ thuật ngược trend giảm tuần."

            verdict = AIRegimeVerdict(
                regime="TRENDING_BULL",
                regime_display=f"BẪY THANH KHOẢN ĐÁY (SMC {active_timeframe})",
                recommended_strategy="SMC_ORDER_BLOCK",
                strategy_display="Săn Bẫy Cá Mập (CHoCH Long)",
                bull_score=bull_sc, bear_score=bear_sc, confidence=conf,
                rationale=(
                    f"[{now_str}] Cá mập vừa quét râu thủng đáy ${smc_res.last_sweep.swept_price:,.1f} ép rũ bỏ phe Long "
                    f"rồi rút chân đóng nến mạnh mẽ. CHoCH đảo chiều Tăng!{macro_note} Ưu tiên Buy tại Bullish OB ${support_price:,.1f}."
                ),
                risk_level="TRUNG BÌNH",
                support_price=support_price, resistance_price=resistance_price,
                suggested_limit_buy=suggested_limit_buy, suggested_limit_sell=suggested_limit_sell,
                updated_time=now_str, mtf_radar=radar, active_timeframe=active_timeframe,
                smc_structure=smc_res.market_structure, smc_bias=smc_res.institutional_bias,
                smc_rationale=smc_res.bias_rationale,
                vwap_fair_price=vwap_res.vwap if vwap_res else current_price,
                vwap_status=vwap_res.valuation_status if vwap_res else "EQUILIBRIUM_FAIR",
                demand_zone=smc_res.nearest_demand_zone, supply_zone=smc_res.nearest_supply_zone,
                active_fvgs=fvg_list, last_sweep_info=sweep_text
            )

        # CASE 2: VWAP Extreme Overextension (Mean-Reversion)
        elif vwap_res and vwap_res.valuation_status == "DISCOUNT_CHEAP":
            verdict = AIRegimeVerdict(
                regime="RANGING_SIDEWAY",
                regime_display=f"VÙNG CHIẾT KHẤU TỔ CHỨC (-2σ VWAP)",
                recommended_strategy="TWO_WAY_RANGE",
                strategy_display="Mean-Reversion Long Về VWAP",
                bull_score=70, bear_score=30, confidence=88,
                rationale=f"[{now_str}] Giá lệch sâu {vwap_res.current_deviation:.1f}σ dưới VWAP ({macro_desc}). {vwap_res.rationale} Tuyệt đối không bán tháo, canh gom Long tại Demand Zone.",
                risk_level="THẤP",
                support_price=support_price, resistance_price=resistance_price,
                suggested_limit_buy=suggested_limit_buy, suggested_limit_sell=suggested_limit_sell,
                updated_time=now_str, mtf_radar=radar, active_timeframe=active_timeframe,
                smc_structure=smc_res.market_structure, smc_bias=smc_res.institutional_bias,
                smc_rationale=smc_res.bias_rationale,
                vwap_fair_price=vwap_res.vwap, vwap_status=vwap_res.valuation_status,
                demand_zone=smc_res.nearest_demand_zone, supply_zone=smc_res.nearest_supply_zone,
                active_fvgs=fvg_list, last_sweep_info=sweep_text
            )

        elif vwap_res and vwap_res.valuation_status == "PREMIUM_EXPENSIVE":
            verdict = AIRegimeVerdict(
                regime="RANGING_SIDEWAY",
                regime_display=f"VÙNG ĐỊNH GIÁ QUÁ CAO (+2σ VWAP)",
                recommended_strategy="TWO_WAY_RANGE",
                strategy_display="Mean-Reversion Short Về VWAP",
                bull_score=30, bear_score=70, confidence=88,
                rationale=f"[{now_str}] Giá lệch cao +{vwap_res.current_deviation:.1f}σ trên VWAP ({macro_desc}). {vwap_res.rationale} Cấm FOMO Mua đuổi, canh chốt lời và tìm điểm Short hồi.",
                risk_level="TRUNG BÌNH",
                support_price=support_price, resistance_price=resistance_price,
                suggested_limit_buy=suggested_limit_buy, suggested_limit_sell=suggested_limit_sell,
                updated_time=now_str, mtf_radar=radar, active_timeframe=active_timeframe,
                smc_structure=smc_res.market_structure, smc_bias=smc_res.institutional_bias,
                smc_rationale=smc_res.bias_rationale,
                vwap_fair_price=vwap_res.vwap, vwap_status=vwap_res.valuation_status,
                demand_zone=smc_res.nearest_demand_zone, supply_zone=smc_res.nearest_supply_zone,
                active_fvgs=fvg_list, last_sweep_info=sweep_text
            )

        # CASE 3: Confirmed Trend with SMC Structure
        elif smc_res.market_structure == "BULLISH_TREND" and bull_counts >= 2:
            is_macro_aligned = macro_cycle in ("SECULAR_BULL", "WEEKLY_BULL")
            if is_macro_aligned:
                reg_name = f"Sóng Tăng Đại Chu Kỳ (BOS Up {active_timeframe} + Macro 1W/1M)"
                bull_sc = 85
                bear_sc = 15
                conf = 94
                macro_msg = f"Đồng thuận Đại Chu Kỳ Tăng Vĩ Mô ({macro_desc}): Dòng tiền lớn duy trì đỉnh đáy nâng dần."
            else:
                reg_name = f"Sóng Hồi Ngắn Hạn (BOS Up {active_timeframe} | Cản Tuần 1W)"
                bull_sc = 68
                bear_sc = 32
                conf = 80
                macro_msg = f"Tăng ngắn hạn nhưng Chu kỳ Vĩ mô ({macro_desc}) đang có cản lớn: Khuyến nghị chốt lời từng phần."

            verdict = AIRegimeVerdict(
                regime="TRENDING_BULL",
                regime_display=reg_name,
                recommended_strategy="TREND_BREAKOUT",
                strategy_display="Bám Theo Cá Mập (Pullback Demand)",
                bull_score=bull_sc, bear_score=bear_sc, confidence=conf,
                rationale=f"[{now_str}] {macro_msg} Canh Long tại khối Order Block hỗ trợ ${support_price:,.1f}.",
                risk_level="TRUNG BÌNH",
                support_price=support_price, resistance_price=resistance_price,
                suggested_limit_buy=suggested_limit_buy, suggested_limit_sell=suggested_limit_sell,
                updated_time=now_str, mtf_radar=radar, active_timeframe=active_timeframe,
                smc_structure=smc_res.market_structure, smc_bias=smc_res.institutional_bias,
                smc_rationale=smc_res.bias_rationale,
                vwap_fair_price=vwap_res.vwap if vwap_res else current_price,
                vwap_status=vwap_res.valuation_status if vwap_res else "EQUILIBRIUM_FAIR",
                demand_zone=smc_res.nearest_demand_zone, supply_zone=smc_res.nearest_supply_zone,
                active_fvgs=fvg_list, last_sweep_info=sweep_text
            )

        elif smc_res.market_structure == "BEARISH_TREND" and bear_counts >= 2:
            is_macro_aligned = macro_cycle in ("SECULAR_BEAR", "WEEKLY_BEAR")
            if is_macro_aligned:
                reg_name = f"Sóng Giảm Đại Chu Kỳ (BOS Down {active_timeframe} + Macro 1W/1M)"
                bull_sc = 15
                bear_sc = 85
                conf = 94
                macro_msg = f"Đồng thuận Đại Chu Kỳ Giảm Vĩ Mô ({macro_desc}): Dòng tiền tổ chức xả hàng diện rộng."
            else:
                reg_name = f"Nhịp Rũ Điều Chỉnh (BOS Down {active_timeframe} | Nền Tuần 1W)"
                bull_sc = 32
                bear_sc = 68
                conf = 80
                macro_msg = f"Áp lực giảm ngắn hạn trên nền Chu kỳ Vĩ mô ({macro_desc}): Chú ý vùng gom giá chiết khấu."

            verdict = AIRegimeVerdict(
                regime="TRENDING_BEAR",
                regime_display=reg_name,
                recommended_strategy="TREND_BREAKOUT",
                strategy_display="Bám Theo Cá Mập (Retest Supply)",
                bull_score=bull_sc, bear_score=bear_sc, confidence=conf,
                rationale=f"[{now_str}] {macro_msg} Canh Short tại khối Order Block kháng cự ${resistance_price:,.1f}.",
                risk_level="TRUNG BÌNH",
                support_price=support_price, resistance_price=resistance_price,
                suggested_limit_buy=suggested_limit_buy, suggested_limit_sell=suggested_limit_sell,
                updated_time=now_str, mtf_radar=radar, active_timeframe=active_timeframe,
                smc_structure=smc_res.market_structure, smc_bias=smc_res.institutional_bias,
                smc_rationale=smc_res.bias_rationale,
                vwap_fair_price=vwap_res.vwap if vwap_res else current_price,
                vwap_status=vwap_res.valuation_status if vwap_res else "EQUILIBRIUM_FAIR",
                demand_zone=smc_res.nearest_demand_zone, supply_zone=smc_res.nearest_supply_zone,
                active_fvgs=fvg_list, last_sweep_info=sweep_text
            )

        # CASE 4: Sideway Choppiness (Two-Way Boundary Execution)
        else:
            verdict = AIRegimeVerdict(
                regime="RANGING_SIDEWAY",
                regime_display=f"Tích Lũy Vùng Biên (Two-Way SMC {active_timeframe})",
                recommended_strategy="TWO_WAY_RANGE",
                strategy_display="Đánh 2 Đầu Biên Order Block",
                bull_score=50, bear_score=50, confidence=88,
                rationale=(
                    f"[{now_str}] Thị trường tích lũy giằng co quanh VWAP ${vwap_res.vwap if vwap_res else current_price:,.1f} ({macro_desc}). "
                    f"Chiến lược tổ chức: Gom Long tại Demand ${support_price:,.1f} và Chốt/Short tại Supply ${resistance_price:,.1f}."
                ),
                risk_level="THẤP",
                support_price=support_price, resistance_price=resistance_price,
                suggested_limit_buy=suggested_limit_buy, suggested_limit_sell=suggested_limit_sell,
                updated_time=now_str, mtf_radar=radar, active_timeframe=active_timeframe,
                smc_structure=smc_res.market_structure, smc_bias=smc_res.institutional_bias,
                smc_rationale=smc_res.bias_rationale,
                vwap_fair_price=vwap_res.vwap if vwap_res else current_price,
                vwap_status=vwap_res.valuation_status if vwap_res else "EQUILIBRIUM_FAIR",
                demand_zone=smc_res.nearest_demand_zone, supply_zone=smc_res.nearest_supply_zone,
                active_fvgs=fvg_list, last_sweep_info=sweep_text
            )

        # Attach Quant Skills Intelligence metrics (from claude-trading-skills)
        verdict.hurst_exponent = primary_ind.get("hurst", 0.50)
        verdict.hurst_regime = primary_ind.get("hurst_regime", "RANDOM_WALK")
        verdict.garman_klass_vol = primary_ind.get("gk_vol", 0.0)
        verdict.volatility_regime = primary_ind.get("vol_regime", "NORMAL")

        # Real-time SMC & VWAP distance metrics
        verdict.dist_demand_pct = smc_res.dist_demand_pct
        verdict.dist_supply_pct = smc_res.dist_supply_pct
        verdict.is_testing_ob = smc_res.is_testing_ob
        verdict.tested_ob_type = smc_res.tested_ob_type
        if vwap_res:
            verdict.dist_vwap_pct = vwap_res.dist_vwap_pct
            verdict.dist_vwap_usd = vwap_res.dist_vwap_usd
            verdict.vwap_sigma_dev = vwap_res.current_deviation

        self.last_verdict = verdict
        return verdict
