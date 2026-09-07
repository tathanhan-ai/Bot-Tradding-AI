# -*- coding: utf-8 -*-
"""
TacticalRegimeDispatcher (Bộ Điều Phối Chiến Lược Thực Chiến Theo Thế Trận)
Tích hợp và phân bổ 6 thế trận giao dịch định lượng chuẩn mực từ Claude Trading Skills:
1. SWING_TREND: Bám sóng xu hướng trung-dài hạn 1h/4h (DailyTradingMode + Chandelier ATR Trailing Exit)
2. DEFENSIVE_SNIPER: Bắt đáy/đỉnh cản tích lũy Sideway (OctoBot Dip/Peak Analyser + SMC Order Blocks)
3. LIQUIDITY_SWEEP: Đón râu quét thanh khoản đảo chiều V-shape (SMC Sweeps + CVD Whale Absorption + DCA Ladder)
4. SQUEEZE_BREAKOUT: Đánh bung nén biên độ (Bollinger Squeeze + Garman-Klass Vol Expansion)
5. DEFICIT_PATIENCE: Chế độ bảo toàn vốn & bù thâm hụt (Anti-Martingale + Cản Cứng Đa Khung R:R >= 2.5)
6. TWO_WAY_GRID: Lưới trung lập tạo lập thị trường khi giá bất động (Geometric Hedge Grid)
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, List, Tuple
import math
import pandas as pd
import numpy as np

from strategy.quant_skills_brain import QuantSkillsBrain, HurstRegimeResult, HighEfficiencyVolatilityResult, ScaledExitTranche
from strategy.octobot_trading_modes import OctoBotTradingCoordinator, OctoBotTradeSetup, StagedTakeProfit
from strategy.smart_money_concepts import SmartMoneyEngine, SMCAnalysisResult
from strategy.institutional_vwap import InstitutionalVWAPEngine, VWAPBandResult


class TacticalFormation(str, Enum):
    SWING_TREND = "SWING_TREND"             # Thế 1: Bám sóng Trend 1h/4h
    DEFENSIVE_SNIPER = "DEFENSIVE_SNIPER"   # Thế 2: Bắt đáy/đỉnh cản Sideway
    LIQUIDITY_SWEEP = "LIQUIDITY_SWEEP"     # Thế 3: Quét thanh khoản & Hấp thụ
    SQUEEZE_BREAKOUT = "SQUEEZE_BREAKOUT"   # Thế 4: Bung nén biên độ Volatility
    DEFICIT_PATIENCE = "DEFICIT_PATIENCE"   # Thế 5: Bảo toàn vốn & Bù thâm hụt
    TWO_WAY_GRID = "TWO_WAY_GRID"           # Thế 6: Lưới trung lập Hedge Grid


@dataclass
class TacticalStrategyProposal:
    formation: TacticalFormation
    formation_name_vi: str
    recommended_side: str            # 'BUY', 'SELL', 'NEUTRAL'
    recommended_type: str            # 'LIMIT', 'POST_ONLY', 'SCALE_RATIO', 'MARKET', 'GRID'
    timeframe: str                   # '1m', '5m', '15m', '1h', '4h'
    entry_price: float
    stop_loss: float
    take_profit: float
    staged_exits: List[Dict[str, Any]]
    risk_reward_ratio: float
    stop_distance_usdt: float
    stop_distance_pct: float
    
    # Capital Sleeve & Position Sizing
    sleeve_type: str                 # 'SHORT_TERM', 'LONG_TERM', 'RESERVE'
    optimal_margin: float            # Margin in USDT ($600 - $1,500)
    optimal_leverage: int            # 2x - 4x
    optimal_quantity: float          # In crypto base units (BTC)
    dollar_risk: float               # Maximum dollar loss if SL hit ($40 - $75)
    
    # Execution & Risk Parameters
    fee_tier: str                    # 'MAKER (0.02%)', 'MAKER LADDER (0.02%)', 'TAKER (0.05%)'
    breakeven_trigger_price: float   # Price where SL moves to entry + fee
    trailing_exit_type: str          # 'CHANDELIER_ATR', 'PARABOLIC_SAR', 'STAGED_LIMIT', 'NONE'
    trailing_multiplier: float
    invalidation_rule: str
    thesis_rationale: str
    confidence_score: int            # 0 - 100


class TacticalRegimeDispatcher:
    """
    Bộ Tổng Chỉ Huy Thế Trận Giao Dịch
    Định vị chính xác trạng thái thị trường và xuất ra kế hoạch giao dịch chuẩn mực.
    """
    def __init__(self):
        self.quant_brain = QuantSkillsBrain()
        self.octobot_coord = OctoBotTradingCoordinator()
        self.smc_engine = SmartMoneyEngine()
        self.vwap_engine = InstitutionalVWAPEngine()

    def identify_formation(
        self,
        df_structure: Optional[pd.DataFrame],
        df_macro: Optional[pd.DataFrame],
        indicators: Dict[str, Any],
        active_timeframe: str = "15m",
        smc_res: Optional[SMCAnalysisResult] = None,
        order_flow_verdict: Optional[Any] = None,
        governor_status: Optional[Any] = None,
        octobot_consensus: Optional[Any] = None
    ) -> Tuple[TacticalFormation, str]:
        """
        Nhận diện chính xác 1 trong 6 thế trận dựa trên thống kê định lượng và cấu trúc thị trường.
        """
        gov_enabled = bool(governor_status and getattr(governor_status, "enabled", False))
        carried_deficit = float(getattr(governor_status, "carried_deficit_pct", 0.0) or 0.0) if gov_enabled else 0.0
        gov_regime = getattr(governor_status, "regime", "ON_TRACK") if gov_enabled else "ON_TRACK"
        
        # 1. Kiểm tra Thế 5: BẢO TOÀN VỐN & BÙ THÂM HỤT
        if gov_enabled and (gov_regime in ("DRAWDOWN_ALERT", "DEFENSIVE_PATIENCE") or carried_deficit >= 15.0):
            return TacticalFormation.DEFICIT_PATIENCE, "Chế độ Bảo Toàn Vốn Cấp Độ 1 (Ưu tiên bảo vệ vốn, chỉ đánh setup R:R >= 2.5 tại cản cứng)"

        # 2. Định lượng Hurst Exponent & Biến động Garman-Klass
        close_series = df_structure["close"] if df_structure is not None and len(df_structure) >= 30 else None
        hurst_res = self.quant_brain.calculate_hurst_exponent(close_series) if close_series is not None else HurstRegimeResult(0.50, "RANDOM_WALK", 0.5, "")
        
        gk_res = self.quant_brain.calculate_garman_klass_volatility(df_structure, timeframe=active_timeframe) if df_structure is not None else None
        vol_ratio = gk_res.volatility_ratio if gk_res else 1.0
        
        adx = float(indicators.get("adx") or 15.0)
        chop = float(indicators.get("chop") or 50.0)
        bb_width = float(indicators.get("bb_width") or 0.03)

        # 3. Kiểm tra Thế 3: QUÉT THANH KHOẢN & HẤP THỤ RÂU NẾN (Liquidity Sweep)
        delta_mom = getattr(order_flow_verdict, "delta_momentum", "BALANCED") if order_flow_verdict else "BALANCED"
        has_sweep = bool(smc_res and smc_res.last_sweep is not None)
        has_absorption = delta_mom in ("ABSORPTION_BUY", "ABSORPTION_SELL")
        if has_sweep or (has_absorption and adx < 25.0):
            return TacticalFormation.LIQUIDITY_SWEEP, "Quét Râu Thanh Khoản SMC & Cá Voi Hấp Thụ (Cơ hội ăn nảy chữ V tại cản)"

        # 4. Kiểm tra Thế 4: BÙNG NỔ NÉN BIÊN ĐỘ (Squeeze Breakout)
        is_extreme_squeeze = (bb_width < 0.016 or vol_ratio < 0.65) and (chop > 58.0 or adx < 16.0)
        vol_spike = bool(indicators.get("vol_ratio", 1.0) > 1.7)
        # Nới lỏng: xác nhận bùng nổ bằng xung lực CVD mạnh khi vol_ratio chưa kịp cập nhật
        cvd_breakout = delta_mom in ("STRONG_BUY_PRESSURE", "STRONG_SELL_PRESSURE")
        if is_extreme_squeeze and (vol_spike or cvd_breakout):
            return TacticalFormation.SQUEEZE_BREAKOUT, "Bùng Nổ Nén Biên Độ Bollinger Squeeze (Động lượng nén bung dải)"

        # 5. Kiểm tra Thế 1: BÁM SÓNG TREND SWING 1H/4H (Trend Riding)
        octo_state = getattr(octobot_consensus, "consensus_state", "NEUTRAL") if octobot_consensus else "NEUTRAL"
        is_trending_hurst = hurst_res.hurst_exponent >= 0.54
        is_trending_adx = adx >= 22.0
        is_octo_trending = octo_state in ("STRONG_BULLISH", "STRONG_BEARISH", "BULLISH", "BEARISH")
        
        if (is_trending_hurst and is_trending_adx) or (is_octo_trending and adx >= 25.0):
            return TacticalFormation.SWING_TREND, "Bám Sóng Xu Hướng Trung-Dài Hạn 1h/4h (Daily Trading Mode + Chandelier Trailing Exit)"

        # 6. Kiểm tra Thế 6: LƯỚI TRUNG LẬP HEDGE GRID (Khi thị trường bất động; nới lỏng để lưới còn kích hoạt được)
        if adx < 14.0 and chop > 60.0 and vol_ratio < 0.60:
            return TacticalFormation.TWO_WAY_GRID, "Lưới Trung Lập Vô Hướng Geometric Hedge Grid (Thị trường bất động, không xu hướng)"

        # 7. MẶC ĐỊNH: Thế 2 hoặc Thế 5
        return TacticalFormation.DEFICIT_PATIENCE if (gov_enabled and carried_deficit > 0) else TacticalFormation.DEFENSIVE_SNIPER,             "Sniper Bắt Đáy / Đỉnh Cản Kỹ Thuật (Mean Reversion tại Swing S/R & Demand Block)"

    def dispatch_strategy(
        self,
        current_price: float,
        best_bid: float,
        best_ask: float,
        df_structure: Optional[pd.DataFrame],
        df_macro: Optional[pd.DataFrame],
        indicators: Dict[str, Any],
        active_timeframe: str = "15m",
        current_balance: float = 5059.69,
        governor_status: Optional[Any] = None,
        octobot_consensus: Optional[Any] = None,
        order_flow_verdict: Optional[Any] = None,
        visual_hft_metrics: Optional[Any] = None
    ) -> TacticalStrategyProposal:
        """
        Xây dựng chi tiết kế hoạch vào lệnh, phân bổ vốn, cắt lỗ, chốt lời theo thế trận.
        """
        atr = float(indicators.get("atr") or current_price * 0.006)
        rsi = float(indicators.get("rsi") or 50.0)
        adx = float(indicators.get("adx") or 15.0)
        spread = max(0.1, best_ask - best_bid)
        
        smc_res = self.smc_engine.analyze(df_structure, current_price) if df_structure is not None and len(df_structure) >= 20 else None
        vwap_res = self.vwap_engine.calculate(df_structure, current_price) if df_structure is not None and len(df_structure) >= 15 else None
        vwap_val = vwap_res.vwap if vwap_res else current_price

        # Tìm Swing Low và Swing High thực tế (Khắc phục hoàn toàn lỗi SL ảo $720!)
        local_low, major_sup, local_high, major_res = self._find_real_swings(df_structure, lookback=25)
        if local_low <= 0 or local_low >= current_price:
            local_low = round(current_price - 1.5 * atr, 1)
        if local_high <= 0 or local_high <= current_price:
            local_high = round(current_price + 1.5 * atr, 1)

        demand_bottom, demand_top = (smc_res.nearest_demand_zone if (smc_res and smc_res.nearest_demand_zone) else (local_low - 0.2 * atr, local_low))
        supply_bottom, supply_top = (smc_res.nearest_supply_zone if (smc_res and smc_res.nearest_supply_zone) else (local_high, local_high + 0.2 * atr))

        formation, desc = self.identify_formation(
            df_structure=df_structure, df_macro=df_macro, indicators=indicators,
            active_timeframe=active_timeframe, smc_res=smc_res,
            order_flow_verdict=order_flow_verdict, governor_status=governor_status,
            octobot_consensus=octobot_consensus
        )

        gov_enabled = bool(governor_status and getattr(governor_status, "enabled", False))
        min_gov_rr = float(getattr(governor_status, "min_risk_reward_ratio", 2.0) or 2.0) if gov_enabled else 2.0

        if formation == TacticalFormation.SWING_TREND:
            delta_mom = getattr(order_flow_verdict, "delta_momentum", "BALANCED") if order_flow_verdict else "BALANCED"
            if delta_mom in ("STRONG_SELL_PRESSURE", "ABSORPTION_SELL"):
                side = "SELL"
            elif delta_mom in ("STRONG_BUY_PRESSURE", "ABSORPTION_BUY"):
                side = "BUY"
            else:
                octo_dir = getattr(octobot_consensus, "recommended_direction", 1) if octobot_consensus else 1
                side = "BUY" if octo_dir >= 0 else "SELL"
            
            if side == "BUY":
                ema20 = float(indicators.get("ema20") or current_price - 0.5 * atr)
                entry = round(min(best_bid, max(ema20, demand_top)), 1)
                sl = round(local_low - 0.25 * atr, 1)
                risk_dist = max(120.0, entry - sl)
                tp1 = round(entry + 1.8 * risk_dist, 1)
                tp2 = round(entry + 3.0 * risk_dist, 1)
                tp3 = round(entry + 4.5 * risk_dist, 1)
            else:
                ema20 = float(indicators.get("ema20") or current_price + 0.5 * atr)
                entry = round(max(best_ask, min(ema20, supply_bottom)), 1)
                sl = round(local_high + 0.25 * atr, 1)
                risk_dist = max(120.0, sl - entry)
                tp1 = round(entry - 1.8 * risk_dist, 1)
                tp2 = round(entry - 3.0 * risk_dist, 1)
                tp3 = round(entry - 4.5 * risk_dist, 1)

            weighted_tp = tp2
            rr = round(abs(weighted_tp - entry) / risk_dist, 2)
            sleeve = "LONG_TERM"
            margin = round(min(1400.0, max(900.0, current_balance * 0.24)), 0)
            leverage = 3
            dollar_risk = round(min(80.0, max(50.0, current_balance * 0.015)), 1)
            qty, dollar_risk = TacticalRegimeDispatcher._size_qty_from_risk(
                dollar_risk, risk_dist, entry, leverage, margin)

            staged = [
                {"tranche": 1, "ratio": 0.30, "price": tp1, "note": "Chốt 30% -> Dời SL về Hòa Vốn (Risk-Free)"},
                {"tranche": 2, "ratio": 0.30, "price": tp2, "note": "Chốt 30% -> Khóa lãi Fib 1.618"},
                {"tranche": 3, "ratio": 0.40, "price": tp3, "note": "Chandelier ATR Trailing Stop gồng hết sóng"}
            ]

            return TacticalStrategyProposal(
                formation=formation, formation_name_vi="Thế 1: Bám Sóng Xu Hướng 1h/4h (Swing Trend)",
                recommended_side=side, recommended_type="LIMIT", timeframe="1h",
                entry_price=entry, stop_loss=sl, take_profit=tp2, staged_exits=staged,
                risk_reward_ratio=rr, stop_distance_usdt=risk_dist, stop_distance_pct=round(risk_dist/entry*100, 2),
                sleeve_type=sleeve, optimal_margin=margin, optimal_leverage=leverage, optimal_quantity=qty, dollar_risk=dollar_risk,
                fee_tier="MAKER (0.02%)", breakeven_trigger_price=tp1, trailing_exit_type="CHANDELIER_ATR", trailing_multiplier=3.0,
                invalidation_rule="Đóng nến 1h gãy qua đáy Swing Low cấu trúc",
                thesis_rationale=f"🚀 [SWING TREND] Xu hướng chủ đạo vững chắc. Đặt Limit Maker đón Pullback tại ${entry:,.1f}. Cắt lỗ chặt dưới cản đáy ${sl:,.1f} (-{risk_dist:,.1f}$). TP đa tầng thả trôi Chandelier Trailing Stop.",
                confidence_score=88
            )

        elif formation in (TacticalFormation.DEFENSIVE_SNIPER, TacticalFormation.DEFICIT_PATIENCE):
            delta_mom = getattr(order_flow_verdict, "delta_momentum", "BALANCED") if order_flow_verdict else "BALANCED"
            if delta_mom in ("STRONG_SELL_PRESSURE", "ABSORPTION_SELL"):
                side = "SELL"
            elif delta_mom in ("STRONG_BUY_PRESSURE", "ABSORPTION_BUY"):
                side = "BUY"
            else:
                # Guard chống bắt dao rơi: trong trend rõ (ADX cao + OctoBot đồng thuận), không mean-revert ngược trend
                octo_state_guard = getattr(octobot_consensus, "consensus_state", "NEUTRAL") if octobot_consensus else "NEUTRAL"
                octo_dir_guard = int(getattr(octobot_consensus, "recommended_direction", 0) or 0) if octobot_consensus else 0
                strong_trend_guard = adx >= 25.0 and octo_state_guard in ("STRONG_BULLISH", "STRONG_BEARISH", "BULLISH", "BEARISH") and octo_dir_guard != 0
                if strong_trend_guard:
                    side = "BUY" if octo_dir_guard > 0 else "SELL"
                else:
                    is_oversold = rsi <= 45.0 or current_price <= vwap_val
                    side = "BUY" if is_oversold else "SELL"

            if side == "BUY":
                entry = round(min(best_bid, demand_top + 0.15 * atr), 1)
                if entry > current_price:
                    entry = round(best_bid, 1)
                sl = round(local_low - 0.20 * atr, 1)
                risk_dist = max(90.0, entry - sl)
                
                tp_target = round(major_res - 0.15 * atr, 1)
                if (tp_target - entry) / risk_dist < min_gov_rr:
                    tp_target = round(entry + risk_dist * max(min_gov_rr, 2.6), 1)
                
                reward_dist = tp_target - entry
                tp1 = round(entry + 0.45 * reward_dist, 1)
                tp2 = tp_target
                tp3 = round(tp_target + 1.5 * atr, 1)
            else:
                entry = round(max(best_ask, supply_bottom - 0.15 * atr), 1)
                if entry < current_price:
                    entry = round(best_ask, 1)
                sl = round(local_high + 0.20 * atr, 1)
                risk_dist = max(90.0, sl - entry)
                
                tp_target = round(major_sup + 0.15 * atr, 1)
                if (entry - tp_target) / risk_dist < min_gov_rr:
                    tp_target = round(entry - risk_dist * max(min_gov_rr, 2.6), 1)
                
                reward_dist = entry - tp_target
                tp1 = round(entry - 0.45 * reward_dist, 1)
                tp2 = tp_target
                tp3 = round(tp_target - 1.5 * atr, 1)

            rr = round(reward_dist / risk_dist, 2)
            sleeve = "SHORT_TERM"
            if formation == TacticalFormation.DEFICIT_PATIENCE:
                margin = round(min(850.0, max(550.0, current_balance * 0.15)), 0)
                leverage = 3
                dollar_risk = round(min(50.0, max(35.0, current_balance * 0.009)), 1)
                form_name = "Thế 5: Bảo Toàn Vốn & Bù Thâm Hụt (Deficit Patience)"
            else:
                margin = round(min(1100.0, max(750.0, current_balance * 0.18)), 0)
                leverage = 4
                dollar_risk = round(min(65.0, max(45.0, current_balance * 0.012)), 1)
                form_name = "Thế 2: Sniper Bắt Đáy/Đỉnh Cản Sideway (Defensive Sniper)"

            qty, dollar_risk = TacticalRegimeDispatcher._size_qty_from_risk(
                dollar_risk, risk_dist, entry, leverage, margin)

            staged = [
                {"tranche": 1, "ratio": 0.40, "price": tp1, "note": f"Chốt 40% tại Trục VWAP (${tp1:,.1f}) -> Dời SL về Breakeven"},
                {"tranche": 2, "ratio": 0.35, "price": tp2, "note": f"Chốt 35% tại Cản Đối Diện (${tp2:,.1f}) | R:R={rr:.1f}:1"},
                {"tranche": 3, "ratio": 0.25, "price": tp3, "note": "Moonbag 25% nếu bùng nổ phá vỡ range"}
            ]

            return TacticalStrategyProposal(
                formation=formation, formation_name_vi=form_name,
                recommended_side=side, recommended_type="POST_ONLY", timeframe="15m",
                entry_price=entry, stop_loss=sl, take_profit=tp2, staged_exits=staged,
                risk_reward_ratio=rr, stop_distance_usdt=risk_dist, stop_distance_pct=round(risk_dist/entry*100, 2),
                sleeve_type=sleeve, optimal_margin=margin, optimal_leverage=leverage, optimal_quantity=qty, dollar_risk=dollar_risk,
                fee_tier="MAKER (0.02%)", breakeven_trigger_price=tp1, trailing_exit_type="STAGED_LIMIT", trailing_multiplier=2.0,
                invalidation_rule="Nến 15m đóng thủng hỗ trợ/kháng cự cản",
                thesis_rationale=f"🎯 [{form_name.upper()}] Nhận diện cản kỹ thuật chính xác: Hỗ trợ ${local_low:,.1f} | Kháng cự ${local_high:,.1f}. Đặt Post-Only Maker đón râu tại ${entry:,.1f}. Cắt lỗ siêu chặt ${sl:,.1f} (-{risk_dist:,.1f}$). R:R={rr:.1f}:1 cực đẹp!",
                confidence_score=85
            )

        elif formation == TacticalFormation.LIQUIDITY_SWEEP:
            delta_mom = getattr(order_flow_verdict, "delta_momentum", "BALANCED") if order_flow_verdict else "BALANCED"
            is_bull_sweep = (smc_res and smc_res.last_sweep and smc_res.last_sweep.type == "BULLISH_SWEEP") or delta_mom in ("ABSORPTION_BUY", "STRONG_BUY_PRESSURE")
            side = "BUY" if is_bull_sweep else "SELL"
            
            sweep_price = float(getattr(smc_res.last_sweep, "swept_price", current_price) if (smc_res and smc_res.last_sweep) else current_price)
            if side == "BUY":
                entry = round(best_bid, 1)
                sl = round(min(sweep_price, local_low) - 0.15 * atr, 1)
                risk_dist = max(80.0, entry - sl)
                tp1 = round(entry + 1.5 * risk_dist, 1)
                tp2 = round(entry + 2.8 * risk_dist, 1)
            else:
                entry = round(best_ask, 1)
                sl = round(max(sweep_price, local_high) + 0.15 * atr, 1)
                risk_dist = max(80.0, sl - entry)
                tp1 = round(entry - 1.5 * risk_dist, 1)
                tp2 = round(entry - 2.8 * risk_dist, 1)

            rr = round(abs(tp2 - entry) / risk_dist, 2)
            sleeve = "SHORT_TERM"
            margin = round(min(950.0, max(600.0, current_balance * 0.15)), 0)
            leverage = 4
            dollar_risk = round(min(50.0, max(35.0, current_balance * 0.010)), 1)
            qty, dollar_risk = TacticalRegimeDispatcher._size_qty_from_risk(
                dollar_risk, risk_dist, entry, leverage, margin)

            staged = [
                {"tranche": 1, "ratio": 0.50, "price": tp1, "note": "Chốt 50% khi nến rút chân hoàn tất V-shape"},
                {"tranche": 2, "ratio": 0.50, "price": tp2, "note": "Chốt 50% tại vùng quét thanh khoản đỉnh đối diện"}
            ]

            return TacticalStrategyProposal(
                formation=formation, formation_name_vi="Thế 3: Quét Râu Thanh Khoản SMC (Liquidity Sweep)",
                recommended_side=side, recommended_type="SCALE_RATIO", timeframe="15m",
                entry_price=entry, stop_loss=sl, take_profit=tp2, staged_exits=staged,
                risk_reward_ratio=rr, stop_distance_usdt=risk_dist, stop_distance_pct=round(risk_dist/entry*100, 2),
                sleeve_type=sleeve, optimal_margin=margin, optimal_leverage=leverage, optimal_quantity=qty, dollar_risk=dollar_risk,
                fee_tier="MAKER LADDER (0.02%)", breakeven_trigger_price=tp1, trailing_exit_type="STAGED_LIMIT", trailing_multiplier=1.8,
                invalidation_rule="Giá xuyên phá đuôi râu quét thanh khoản",
                thesis_rationale=f"⚡ [LIQUIDITY SWEEP] Cá mập quét râu Stop-hunt tại ${sweep_price:,.1f}. Rải Thang 3 Tầng DCA Ladder đón sóng bật chữ V. Cắt lỗ cực an toàn sau râu nến tại ${sl:,.1f}.",
                confidence_score=82
            )

        elif formation == TacticalFormation.SQUEEZE_BREAKOUT:
            cvd_pressure = getattr(order_flow_verdict, "delta_momentum", "BALANCED") if order_flow_verdict else "BALANCED"
            side = "BUY" if cvd_pressure in ("STRONG_BUY_PRESSURE", "ABSORPTION_BUY") or current_price >= vwap_val else "SELL"
            
            entry = current_price
            if side == "BUY":
                sl = round(entry - 1.2 * atr, 1)
                risk_dist = max(110.0, entry - sl)
                tp1 = round(entry + 2.0 * risk_dist, 1)
                tp2 = round(entry + 3.5 * risk_dist, 1)
            else:
                sl = round(entry + 1.2 * atr, 1)
                risk_dist = max(110.0, sl - entry)
                tp1 = round(entry - 2.0 * risk_dist, 1)
                tp2 = round(entry - 3.5 * risk_dist, 1)

            rr = round(abs(tp2 - entry) / risk_dist, 2)
            sleeve = "SHORT_TERM"
            margin = round(min(1200.0, max(800.0, current_balance * 0.20)), 0)
            leverage = 3
            dollar_risk = round(min(70.0, max(50.0, current_balance * 0.013)), 1)
            qty, dollar_risk = TacticalRegimeDispatcher._size_qty_from_risk(
                dollar_risk, risk_dist, entry, leverage, margin)

            staged = [
                {"tranche": 1, "ratio": 0.40, "price": tp1, "note": "Chốt 40% tại 2.0R -> Dời SL về Breakeven"},
                {"tranche": 2, "ratio": 0.60, "price": tp2, "note": "Gồng sóng bung nén Parabolic SAR Trailing"}
            ]

            return TacticalStrategyProposal(
                formation=formation, formation_name_vi="Thế 4: Bùng Nổ Nén Biên Độ (Squeeze Breakout)",
                recommended_side=side, recommended_type="MARKET", timeframe="15m",
                entry_price=entry, stop_loss=sl, take_profit=tp2, staged_exits=staged,
                risk_reward_ratio=rr, stop_distance_usdt=risk_dist, stop_distance_pct=round(risk_dist/entry*100, 2),
                sleeve_type=sleeve, optimal_margin=margin, optimal_leverage=leverage, optimal_quantity=qty, dollar_risk=dollar_risk,
                fee_tier="TAKER (0.05%)", breakeven_trigger_price=tp1, trailing_exit_type="PARABOLIC_SAR", trailing_multiplier=2.5,
                invalidation_rule="Nến đóng đảo chiều quay ngược vào trong dải Bollinger",
                thesis_rationale=f"📦 [SQUEEZE BREAKOUT] Bollinger nén cực đại bung biên độ. Đánh theo xung lực mở dải với Margin ${margin:,.0f}. Stop-loss ${sl:,.1f}, thả trôi lợi nhuận Parabolic SAR.",
                confidence_score=80
            )

        else:
            entry = current_price
            sl = round(entry - 2.5 * atr, 1)
            tp = round(entry + 2.5 * atr, 1)
            return TacticalStrategyProposal(
                formation=TacticalFormation.TWO_WAY_GRID, formation_name_vi="Thế 6: Lưới Trung Lập Vô Hướng (Two-Way Hedge Grid)",
                recommended_side="NEUTRAL", recommended_type="GRID", timeframe="15m",
                entry_price=entry, stop_loss=sl, take_profit=tp, staged_exits=[],
                risk_reward_ratio=1.0, stop_distance_usdt=2.5*atr, stop_distance_pct=round(2.5*atr/entry*100, 2),
                sleeve_type="SHORT_TERM", optimal_margin=900.0, optimal_leverage=2, optimal_quantity=0.03, dollar_risk=50.0,
                fee_tier="MAKER (0.02%)", breakeven_trigger_price=entry, trailing_exit_type="NONE", trailing_multiplier=0.0,
                invalidation_rule="Giá bứt phá ra ngoài biên độ lưới +-3 ATR",
                thesis_rationale=f"⚖️ [HEDGE GRID] Thị trường bất động hoàn toàn. Rải lưới hình học đối xứng quanh VWAP ${vwap_val:,.1f} biên [${entry - 2.0*atr:,.1f} - ${entry + 2.0*atr:,.1f}].",
                confidence_score=75
            )

    @staticmethod
    def _size_qty_from_risk(dollar_risk: float, risk_dist: float, entry: float,
                            leverage: int, margin_floor: float) -> Tuple[float, float]:
        """
        Size chuan: qty = risk/risk_dist; margin = qty*entry/lev. Chi khi margin duoi floor
        moi nang qty len, va phai cap nhat lai dollar_risk thuc (= qty*risk_dist) de khong bao sai risk.
        Tra ve (qty, actual_dollar_risk).
        """
        qty = round(dollar_risk / max(1.0, risk_dist), 4)
        if qty * entry / max(1, leverage) < margin_floor:
            qty = round(margin_floor * leverage / max(1.0, entry), 4)
            dollar_risk = round(qty * risk_dist, 1)
        return qty, dollar_risk

    @staticmethod
    def _find_real_swings(df: Optional[pd.DataFrame], lookback: int = 25) -> Tuple[float, float, float, float]:
        if df is None or len(df) < 10:
            return 0.0, 0.0, 0.0, 0.0
        eff_len = min(len(df), max(10, lookback))
        recent = df.iloc[-eff_len:]
        local_low = float(recent["low"].min())
        local_high = float(recent["high"].max())
        major_len = min(len(df), 60)
        major = df.iloc[-major_len:]
        major_sup = float(major["low"].min())
        major_res = float(major["high"].max())
        return local_low, major_sup, local_high, major_res
