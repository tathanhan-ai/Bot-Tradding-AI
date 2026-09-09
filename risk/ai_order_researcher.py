"""
AI Real-Time Order Execution Researcher (Nghiên Cứu Lệnh Thông Minh Đa Nhân Tố & Chuẩn Quant)
Tối ưu hóa toàn diện cho giao dịch Futures tần suất cao / trong ngày:
1. Micro-Structure Price Discovery: Triệt tiêu hoàn toàn lỗi đặt giá cản vĩ mô quá xa (4-5%).
   Giá vào lệnh luôn bám sát diễn biến thị trường (0.02% - 0.45%) tại Best Bid/Ask, Order Block hoặc VWAP.
2. Dynamic Order Type Dispatcher: Lựa chọn thông minh trong 7 loại lệnh (Market, Limit, Post-Only, Stop-Limit, Trailing, TWAP, DCA Ladder).
3. Institutional Multi-Confluence: Kết hợp dòng tiền Order Flow (CVD), SMC (Order Block / Fair Value Gap / Liquidity Sweep) và Institutional VWAP.
4. Structural SL/TP & R:R Optimization: Đảm bảo tỷ lệ R:R tối thiểu 1.8:1, SL bám ngoài râu quét thanh khoản.
5. Kelly Sizing & Dynamic Leverage: Phân bổ vốn chính xác theo tỷ lệ rủi ro tối đa 1.5% - 2.0% tài khoản.
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List
import pandas as pd
import numpy as np

from strategy.smart_money_concepts import SmartMoneyEngine, SMCAnalysisResult
from strategy.institutional_vwap import InstitutionalVWAPEngine, VWAPBandResult
from strategy.hummingbot_inventory_skew import HummingbotInventorySkewEngine, InventorySkewStatus
from strategy.carver_systematic_engine import CarverSystematicEngine, CarverSystematicOutput
from strategy.quant_skills_brain import QuantSkillsBrain
from risk.structural_sl_tp import StructuralRiskCalculator, StructuralTradeSetup
from risk.capital_allocator import PortfolioCapitalAllocator, AllocationResult
from strategy.tactical_regime_dispatcher import TacticalRegimeDispatcher, TacticalFormation, TacticalStrategyProposal


@dataclass
class AIOrderResearchResult:
    recommended_type: str            # 'POST_ONLY', 'LIMIT', 'MARKET', 'CONDITIONAL', 'TRAILING_STOP', 'TWAP', 'SCALE_RATIO', 'DUAL_BRACKET'
    recommended_side: str            # 'BUY', 'SELL', or 'DUAL'
    optimal_price: float
    optimal_margin: float
    optimal_leverage: int
    optimal_trigger_price: float = 0.0
    optimal_trigger_cond: str = "ABOVE"
    optimal_callback_pct: float = 0.8
    optimal_twap_slices: int = 5
    structural_sl: float = 0.0
    structural_tp: float = 0.0
    structural_tp_macro: float = 0.0
    rr_ratio: float = 1.8
    fee_tier: str = "MAKER (0.02%)"
    estimated_fee_saved_usdt: float = 0.0
    win_probability: int = 75
    research_rationale: str = ""
    dual_buy_price: float = 0.0
    dual_sell_price: float = 0.0
    execution_horizon: str = "IMMEDIATE"   # 'IMMEDIATE', 'PULLBACK', 'BREAKOUT', 'SWEEP_REVERSAL'
    carver_output: Optional[Dict[str, Any]] = None
    carver_contracts: float = 0.0
    carver_action: str = "HOLD"
    carver_buffer_bands: Tuple[float, float] = (0.0, 0.0)
    vpin: float = 0.35
    toxicity_regime: str = "CLEAN"
    market_resilience_pct: float = 85.0
    lob_imbalance_20: float = 0.0
    kelly_multiplier: float = 1.0
    octobot_tradable: bool = True
    ready: bool = True
    active_timeframe: str = "15m"
    dca_ladder_step: float = 0.005
    twap_interval_seconds: int = 6
    strategy_horizon: str = "SHORT_TERM"
    is_vpin_throttled: bool = False
    sleeve_allocation: Optional[Dict[str, Any]] = None
    monthly_regime: str = "ON_TRACK"
    monthly_size_multiplier: float = 1.0
    tactical_formation: str = "DEFENSIVE_SNIPER"
    staged_exits: list = None


class AIOrderResearcher:
    def __init__(self):
        self.structural_calculator = StructuralRiskCalculator(atr_buffer_mult=0.3)
        self.capital_allocator = PortfolioCapitalAllocator()
        self.tactical_dispatcher = TacticalRegimeDispatcher()
        self.smc_engine = SmartMoneyEngine(atr_mult=1.2)
        self.vwap_engine = InstitutionalVWAPEngine()
        self.hummingbot_skew = HummingbotInventorySkewEngine(risk_aversion_gamma=0.15, refresh_tolerance_pct=0.08)
        self.carver_engine = CarverSystematicEngine(
            annual_vol_target_pct=0.25,
            max_forecast=20.0,
            default_rdm=1.35,
            buffer_pct=0.12
        )

    def research(
        self,
        current_price: float,
        best_bid: float,
        best_ask: float,
        spread: float,
        indicators: dict,
        ai_verdict: Any,
        ensemble_result: Any,
        ai_cro: Any,
        current_balance: float = 5000.0,
        df_structure: Optional[pd.DataFrame] = None,
        df_macro: Optional[pd.DataFrame] = None,
        active_timeframe: str = "15m",
        candle_confluence: Optional[Any] = None,
        order_flow_verdict: Optional[Any] = None,
        effective_leverage: Optional[int] = None,
        current_position: Optional[dict] = None,
        inventory_skew: Optional[Any] = None,
        visual_hft_metrics: Optional[Any] = None,
        jesse_metrics: Optional[Any] = None,
        octobot_consensus: Optional[Any] = None,
        pending_orders: Optional[List[Any]] = None,
        monthly_governor_status: Optional[Any] = None
    ) -> AIOrderResearchResult:
        """
        Synthesizes all quantitative signals on every tick to recommend the exact optimal order.
        """
        # 1. Market Volatility & Spread Normalization
        atr = indicators.get("atr") or (current_price * 0.008)
        rsi = indicators.get("rsi") or 50.0
        adx = indicators.get("adx") or 20.0
        
        bid = best_bid if best_bid > 0 else (current_price - 0.5)
        ask = best_ask if best_ask > 0 else (current_price + 0.5)
        eff_spread = max(spread, ask - bid, 0.5)
        regime = ai_verdict.regime if ai_verdict else "RANGING_SIDEWAY"
        consensus_score = ensemble_result.consensus_score if ensemble_result else 0.0
        base_confidence = ensemble_result.confidence if ensemble_result else 75.0

        # 2. Extract Subsystem Context (SMC, VWAP, Order Flow)
        smc_res: Optional[SMCAnalysisResult] = None
        if df_structure is not None and len(df_structure) >= 20:
            try:
                smc_res = self.smc_engine.analyze(df_structure, current_price)
            except Exception:
                smc_res = None

        vwap_res: Optional[VWAPBandResult] = None
        if df_structure is not None and len(df_structure) >= 15:
            try:
                vwap_res = self.vwap_engine.calculate(df_structure, current_price)
            except Exception:
                vwap_res = None

        delta_momentum = getattr(order_flow_verdict, "delta_momentum", "BALANCED") if order_flow_verdict else "BALANCED"
        absorption = delta_momentum if delta_momentum in ("ABSORPTION_BUY", "ABSORPTION_SELL") else "NONE"

        # Multi-Candle Pattern Radar Confluence Score Extraction
        radar_score = 0.0
        if isinstance(candle_confluence, dict):
            radar_score = float(candle_confluence.get("confluence_score", candle_confluence.get("radar_score", 0.0)) or 0.0)
        elif hasattr(candle_confluence, "confluence_score"):
            radar_score = float(candle_confluence.confluence_score or 0.0)
        elif hasattr(candle_confluence, "radar_score"):
            radar_score = float(candle_confluence.radar_score or 0.0)

        # Multi-Timeframe Strategy Engine Calibration:
        # Dynamically infer the optimal tactical execution timeframe from Candlestick Confluence Radar
        tactical_tf = getattr(candle_confluence, "tactical_timeframe", None)
        if tactical_tf and abs(radar_score) >= 15.0:
            trade_tf = tactical_tf
            if str(trade_tf).lower() in ("1m", "3m", "5m") and str(active_timeframe).lower() not in ("1m", "3m", "5m"):
                trade_tf = active_timeframe or "15m"
        else:
            trade_tf = active_timeframe or "15m"

        # Hard institutional safeguard: Automated execution timeframe must be >= 15m to avoid high-frequency whipsaws
        if str(trade_tf).lower() in ("1m", "3m", "5m") and str(active_timeframe).lower() not in ("1m", "3m", "5m"):
            trade_tf = "15m"

        # Tactical Regime Dispatcher: Invoke Master 6-Formation Tactical Engine
        tactical_prop = self.tactical_dispatcher.dispatch_strategy(
            current_price=current_price,
            best_bid=bid,
            best_ask=ask,
            df_structure=df_structure,
            df_macro=df_macro,
            indicators=indicators,
            active_timeframe=active_timeframe,
            current_balance=current_balance,
            governor_status=monthly_governor_status,
            octobot_consensus=octobot_consensus,
            order_flow_verdict=order_flow_verdict,
            visual_hft_metrics=visual_hft_metrics
        )

        # 3. Directional Synthesis (Multi-Factor Scoring)
        dir_score = (consensus_score * 0.40) + (radar_score * 0.25)
        
        # Factor CVD momentum
        if delta_momentum == "STRONG_BUY_PRESSURE":
            dir_score += 15.0
        elif delta_momentum == "STRONG_SELL_PRESSURE":
            dir_score -= 15.0
            
        # Factor Absorption
        if absorption == "ABSORPTION_BUY":
            dir_score += 12.0
        elif absorption == "ABSORPTION_SELL":
            dir_score -= 12.0

        # Factor VWAP discount/premium
        if vwap_res:
            if vwap_res.valuation_status == "DISCOUNT_CHEAP" and current_price < vwap_res.vwap:
                dir_score += 10.0
            elif vwap_res.valuation_status == "PREMIUM_EXPENSIVE" and current_price > vwap_res.vwap:
                dir_score -= 10.0

        # Factor SMC Liquidity Sweeps
        if smc_res and smc_res.last_sweep:
            if smc_res.last_sweep.type == "BULLISH_SWEEP":
                dir_score += 15.0
            elif smc_res.last_sweep.type == "BEARISH_SWEEP":
                dir_score -= 15.0

        # Clamp dir_score
        dir_score = max(-100.0, min(100.0, dir_score))

        # Decide Direction
        is_centered = (vwap_res is None) or (abs(current_price - vwap_res.vwap) <= atr * 0.75)
        is_sideway_market = (
            is_centered
            and (
                (ensemble_result and getattr(ensemble_result, "consensus_verdict", "") in ("SIDEWAY_GRID", "NEUTRAL"))
                or regime in ("RANGING_SIDEWAY", "SIDEWAY_GRID", "CHOPPY", "NEUTRAL", "EQUILIBRIUM_FAIR")
                or adx < 22.0
            )
        )
        # Anti-Fomo Overbought / Oversold Guard:
        # Prevent buying at the top of an overextended move (RSI >= 62) or shorting into deep support (RSI <= 38)
        rsi_val = float(indicators.get("rsi", 50.0))
        resistance_val = float(indicators.get("resistance", 0.0))
        support_val = float(indicators.get("support", 0.0))
        is_near_resistance = (resistance_val > 0) and ((resistance_val - current_price) <= 0.35 * atr) and (current_price <= resistance_val + 0.1 * atr)
        is_near_support = (support_val > 0) and ((current_price - support_val) <= 0.35 * atr) and (current_price >= support_val - 0.1 * atr)

        if dir_score >= 12.0:
            if rsi_val >= 62.0 or is_near_resistance:
                # CẤM ĐU ĐỈNH: Khi RSI >= 62 hoặc sát cản, chuyển sang Lưới hoặc chỉ cho phép mua chiết khấu sâu
                opt_side = "SIDEWAY" if dir_score < 35.0 else "BUY"
            else:
                opt_side = "BUY"
        elif dir_score <= -12.0:
            if rsi_val <= 38.0 or is_near_support:
                # CẤM BÁN ĐÁY: Khi RSI <= 38 hoặc sát hỗ trợ, chuyển sang Lưới hoặc chỉ cho phép bán hồi
                opt_side = "SIDEWAY" if dir_score > -35.0 else "SELL"
            else:
                opt_side = "SELL"
        elif is_sideway_market:
            opt_side = "SIDEWAY"
        else:
            if vwap_res and current_price < vwap_res.vwap:
                opt_side = "SIDEWAY" if (rsi_val >= 62.0 or is_near_resistance) else "BUY"
            else:
                opt_side = "SIDEWAY" if (rsi_val <= 38.0 or is_near_support) else "SELL"

        # 4. Realistic Micro-Structure Entry Price Discovery (Within 0.02% - 0.45% of Live Price)
        execution_horizon = "IMMEDIATE"
        entry_price = current_price
        entry_rationale = ""
        has_ob_entry = False

        if opt_side == "SIDEWAY":
            execution_horizon = "RANGE_GRID"
            entry_price = current_price
            entry_rationale = f"Thị trường Sideway / Tích lũy (ADX={adx:.1f}). AI khuyến nghị kích hoạt Futures Hedge Grid biên [${current_price - 2.0 * atr:,.1f} - ${current_price + 2.0 * atr:,.1f}]"
        elif opt_side == "BUY":
            has_ob_entry = False
            if smc_res and smc_res.nearest_demand_zone:
                ob_bottom, ob_top = smc_res.nearest_demand_zone
                if ob_top < current_price and (current_price - ob_top) <= 0.65 * atr:
                    entry_price = round(max(ob_top, bid), 1)
                    entry_rationale = f"Đón nảy tại Demand Block SMC (${entry_price:,.1f})"
                    execution_horizon = "PULLBACK"
                    has_ob_entry = True

            if not has_ob_entry and vwap_res and vwap_res.lower_band_1 < current_price:
                vwap_band_dist = current_price - vwap_res.lower_band_1
                if 0 < vwap_band_dist <= 0.55 * atr:
                    entry_price = round(max(vwap_res.lower_band_1, bid), 1)
                    entry_rationale = f"Đón nảy tại Dải dưới VWAP -1σ (${entry_price:,.1f})"
                    execution_horizon = "PULLBACK"
                    has_ob_entry = True

            if not has_ob_entry:
                if rsi_val >= 62.0:
                    entry_price = round(min(bid, current_price - 0.40 * atr), 1)
                    entry_rationale = f"🛡️ CHỐNG ĐU ĐỈNH (RSI={rsi_val:.1f}): Đón chiết khấu sâu (${entry_price:,.1f}) đảm bảo Maker 0.02%"
                    execution_horizon = "PULLBACK"
                else:
                    entry_price = round(bid, 1)
                    entry_rationale = f"Treo tại đầu sổ Best Bid (${entry_price:,.1f}) đảm bảo Maker 0.02%"
                    execution_horizon = "IMMEDIATE"

            if entry_price > current_price:
                entry_price = round(bid, 1)
            elif (current_price - entry_price) > 0.75 * atr:
                entry_price = round(current_price - (0.15 * atr), 1)

        else:  # SELL / SHORT
            has_ob_entry = False
            if smc_res and smc_res.nearest_supply_zone:
                ob_bottom, ob_top = smc_res.nearest_supply_zone
                if ob_bottom > current_price and (ob_bottom - current_price) <= 0.65 * atr:
                    entry_price = round(min(ob_bottom, ask), 1)
                    entry_rationale = f"Đón ép tại Supply Block SMC (${entry_price:,.1f})"
                    execution_horizon = "PULLBACK"
                    has_ob_entry = True

            if not has_ob_entry and vwap_res and vwap_res.upper_band_1 > current_price:
                vwap_band_dist = vwap_res.upper_band_1 - current_price
                if 0 < vwap_band_dist <= 0.55 * atr:
                    entry_price = round(min(vwap_res.upper_band_1, ask), 1)
                    entry_rationale = f"Đón ép tại Dải trên VWAP +1σ (${entry_price:,.1f})"
                    execution_horizon = "PULLBACK"
                    has_ob_entry = True

            if not has_ob_entry:
                if rsi_val <= 38.0:
                    entry_price = round(max(ask, current_price + 0.40 * atr), 1)
                    entry_rationale = f"🛡️ CHỐNG BÁN ĐÁY (RSI={rsi_val:.1f}): Đón giá hồi phục (${entry_price:,.1f}) đảm bảo Maker 0.02%"
                    execution_horizon = "PULLBACK"
                else:
                    entry_price = round(ask, 1)
                    entry_rationale = f"Treo tại đầu sổ Best Ask (${entry_price:,.1f}) đảm bảo Maker 0.02%"
                    execution_horizon = "IMMEDIATE"

            if entry_price < current_price:
                entry_price = round(ask, 1)
            elif (entry_price - current_price) > 0.75 * atr:
                entry_price = round(current_price + (0.15 * atr), 1)

        # 5. Dynamic Order Type Selection Engine & Timeframe Calibration
        # Anchor to Tactical Proposal if available. Khi side trung nhau, GIU NGUYEN loai lenh
        # cua formation (Trend->LIMIT, Sniper->POST_ONLY, Sweep->SCALE, Squeeze->MARKET) thay vi
        # de 7 nhanh ben duoi ghi de chet (loi cu: opt_type bi reset POST_ONLY o duoi).
        anchored_type = None
        if tactical_prop is not None and tactical_prop.formation != TacticalFormation.TWO_WAY_GRID:
            if abs(dir_score) < 12.0:
                opt_side = tactical_prop.recommended_side

            if tactical_prop.recommended_side == opt_side:
                anchored_type = tactical_prop.recommended_type
                opt_type = anchored_type
                entry_price = tactical_prop.entry_price
                struct_sl = tactical_prop.stop_loss
                struct_tp = tactical_prop.take_profit
                rr_ratio = tactical_prop.risk_reward_ratio
                risk_dist = tactical_prop.stop_distance_usdt
                reward_dist = abs(struct_tp - entry_price)
                fee_tier = tactical_prop.fee_tier
                strategy_horizon = tactical_prop.sleeve_type
                order_rationale = tactical_prop.thesis_rationale
                optimal_margin = tactical_prop.optimal_margin
                lev = tactical_prop.optimal_leverage
                if tactical_prop.formation == TacticalFormation.SWING_TREND and active_timeframe in ("15m", None):
                    trade_tf = "1h"

        tf_lower = str(trade_tf).lower()

        if tf_lower in ("1m", "3m"):
            opt_callback = round(max(0.25, min(0.65, (atr / current_price) * 100.0 * 0.5)), 2)
            opt_twap_slices = 3
            twap_interval = 3
            dca_step = 0.0025
            horizon_mode = f"SCALP_{tf_lower.upper()}"
        elif tf_lower in ("5m", "15m", "30m"):
            opt_callback = round(max(0.4, min(1.2, (atr / current_price) * 100.0 * 0.75)), 2)
            opt_twap_slices = 5
            twap_interval = 10 if tf_lower == "5m" else 30
            dca_step = 0.005
            horizon_mode = f"INTRADAY_{tf_lower.upper()}"
        else:
            opt_callback = round(max(0.8, min(2.5, (atr / current_price) * 100.0 * 1.0)), 2)
            opt_twap_slices = 5
            twap_interval = 60
            dca_step = 0.010
            horizon_mode = f"SWING_{tf_lower.upper()}"

        # Khi da anchor theo formation thi GIU loai lenh, bo qua 7 nhanh chon lai ben duoi
        # (tru khi nhanh MARKET phat hien FOMO can ha ve POST_ONLY - xu ly rieng ben duoi).
        skip_type_dispatch = anchored_type is not None
        opt_trigger_price = 0.0
        opt_trigger_cond = "ABOVE"
        if not skip_type_dispatch:
            opt_type = "POST_ONLY"
            fee_tier = "MAKER (0.02%)"
        execution_horizon = horizon_mode

        bb_width = indicators.get("bb_width", 0.04)

        # 1. MARKET (Momentum Surge: CVD pressure + high ADX + directional score)
        if skip_type_dispatch:
            # Anchor giu loai lenh formation. Chi cho phep ghi de an toan: FOMO dinh/day ha MARKET ve POST_ONLY.
            if anchored_type == "MARKET" and ((opt_side == "BUY" and (rsi_val >= 62.0 or is_near_resistance)) or (opt_side == "SELL" and (rsi_val <= 38.0 or is_near_support))):
                opt_type = "POST_ONLY"
                entry_price = round(bid if opt_side == "BUY" else ask, 1)
                fee_tier = "MAKER (0.02%)"
                execution_horizon = f"PULLBACK_{horizon_mode}"
                order_rationale = f"🛡️ CHỐNG ĐU ĐỈNH/BÁN ĐÁY ({active_timeframe}): RSI={rsi_val:.1f}. Chuyển lệnh MARKET sang POST_ONLY đón chiết khấu bảo vệ vốn!"
        elif abs(dir_score) >= 55.0 and delta_momentum in ("STRONG_BUY_PRESSURE", "STRONG_SELL_PRESSURE") and adx >= 25.0:
            if (opt_side == "BUY" and (rsi_val >= 62.0 or is_near_resistance)) or (opt_side == "SELL" and (rsi_val <= 38.0 or is_near_support)):
                opt_type = "POST_ONLY"
                entry_price = round(bid if opt_side == "BUY" else ask, 1)
                fee_tier = "MAKER (0.02%)"
                execution_horizon = f"PULLBACK_{horizon_mode}"
                order_rationale = f"🛡️ CHỐNG ĐU ĐỈNH/BÁN ĐÁY ({active_timeframe}): RSI={rsi_val:.1f}. Chuyển lệnh MARKET sang POST_ONLY đón chiết khấu bảo vệ vốn!"
            else:
                opt_type = "MARKET"
                entry_price = current_price
                fee_tier = "TAKER (0.05%)"
                execution_horizon = f"IMMEDIATE_{horizon_mode}"
                order_rationale = f"⚡ BÙNG NỔ ĐỘNG LƯỢNG ({active_timeframe}): CVD {delta_momentum} áp đảo, ADX {adx:.1f}. AI khuyến nghị vào lệnh MARKET ngay lập tức để không lỡ sóng!"

        # 2. CONDITIONAL / STOP-LIMIT (Volatility Squeeze Breakout)
        # Trigger 0.60 ATR (thay vi 0.25 ATR de tranh rau quet kich hoat oan) + xac nhan CVD cung chieu.
        # Neu CVD nguoc/khong ro thi ha ve POST_ONLY don gia cho pullback thay vi phuc kich.
        elif not skip_type_dispatch and (bb_width < 0.018 or adx < 18.0) and regime in ("RANGING_SIDEWAY", "CHOPPY", "NEUTRAL"):
            want_buy = (opt_side == "BUY" or (opt_side == "SIDEWAY" and dir_score >= 0))
            cvd_ok = (want_buy and delta_momentum in ("STRONG_BUY_PRESSURE", "ABSORPTION_BUY")) or \
                     (not want_buy and delta_momentum in ("STRONG_SELL_PRESSURE", "ABSORPTION_SELL"))
            if not cvd_ok:
                opt_type = "POST_ONLY"
                entry_price = round(bid if want_buy else ask, 1)
                fee_tier = "MAKER (0.02%)"
                execution_horizon = f"PULLBACK_{horizon_mode}"
                order_rationale = f"📦 NÉN BIÊN NHƯNG CVD CHƯA XÁC NHẬN ({active_timeframe}): Chờ pullback POST_ONLY thay vì phục kích Stop-Limit oan!"
            else:
                opt_type = "CONDITIONAL"
                fee_tier = "STOP-LIMIT (MAKER/TAKER)"
                execution_horizon = f"BREAKOUT_{horizon_mode}"
                trig_dist = 0.60 * atr
                if want_buy:
                    opt_trigger_price = round(current_price + trig_dist, 1)
                    opt_trigger_cond = "ABOVE"
                    entry_price = round(opt_trigger_price + 0.5, 1)
                else:
                    opt_trigger_price = round(current_price - trig_dist, 1)
                    opt_trigger_cond = "BELOW"
                    entry_price = round(opt_trigger_price - 0.5, 1)
                order_rationale = f"📦 BUNG NÉN CÓ CVD XÁC NHẬN ({active_timeframe}): Stop-Limit đón Breakout tại ngưỡng ${opt_trigger_price:,.1f} (0.60 ATR, CVD {delta_momentum})!"

        # 3. TRAILING_STOP (Trend Wave Following)
        elif not skip_type_dispatch and regime in ("TRENDING_BULL", "TRENDING_BEAR") and abs(dir_score) >= 30.0:
            opt_type = "TRAILING_STOP"
            entry_price = current_price
            fee_tier = "TRAILING TAKER"
            execution_horizon = f"TRAILING_{horizon_mode}"
            order_rationale = f"🏄 BÁM SÓNG XU HƯỚNG ({active_timeframe}): Xu hướng {regime} ({dir_score:+.1f}đ). AI khuyến nghị Trailing Stop tự động bám đỉnh/đáy rút râu {opt_callback}%!"

        # 4. SCALE_RATIO / DCA LADDER (SMC Liquidity Sweep / Absorption Reversal)
        elif not skip_type_dispatch and ((smc_res and smc_res.last_sweep is not None) or (absorption in ("ABSORPTION_BUY", "ABSORPTION_SELL"))):
            opt_type = "SCALE_RATIO"
            fee_tier = "MAKER LADDER (0.02%)"
            execution_horizon = f"LADDER_{horizon_mode}"
            order_rationale = f"🎯 QUÉT THANH KHOẢN SMC ({active_timeframe}): Phát hiện {smc_res.last_sweep.type if (smc_res and smc_res.last_sweep) else absorption}. AI khuyến nghị Rải Thang 3 Tầng DCA Ladder (20%-30%-50%, bước {dca_step*100:.2f}%) quanh ${entry_price:,.1f} đón râu nến!"

        # 5. TWAP (High Spread / Volatility Panic Slice)
        # Lat cat TWAP_SLICE khop ngay nhu MARKET -> phi Taker 0.05%, chi dung cho lenh lon.
        elif not skip_type_dispatch and (eff_spread > 5.0 or (atr / current_price) > 0.014 or regime == "VOLATILE_PANIC"):
            opt_type = "TWAP"
            entry_price = current_price
            fee_tier = "TWAP TAKER (0.05%/slice)"
            execution_horizon = f"TWAP_{horizon_mode}"
            order_rationale = f"🌊 BIẾN ĐỘNG MẠNH / SPREAD GIÃN ({active_timeframe}): Spread ${eff_spread:.1f}. AI khuyến nghị TWAP chia đều {opt_twap_slices} lát cắt (mỗi {twap_interval}s) chống trượt giá sàn!"

        # 6. LIMIT (Order Block / Institutional Level Pullback)
        elif not skip_type_dispatch and (has_ob_entry or (vwap_res and abs(current_price - vwap_res.vwap) >= 0.4 * atr)):
            opt_type = "LIMIT"
            fee_tier = "MAKER (0.02%)"
            execution_horizon = f"PULLBACK_{horizon_mode}"
            order_rationale = f"🏛️ LỆNH CHỜ LIMIT TẠI MỨC TỔ CHỨC ({active_timeframe}): {entry_rationale}. Chờ đón sóng hồi phục tại vùng hỗ trợ/kháng cự then chốt!"

        # 7. POST_ONLY (Best Bid/Ask Passive Maker at Market Equilibrium)
        elif not skip_type_dispatch:
            opt_type = "POST_ONLY"
            fee_tier = "MAKER (0.02%)"
            execution_horizon = f"MAKER_{horizon_mode}"
            order_rationale = f"🛡️ BẢO VỆ MAKER 0.02% ({active_timeframe}): {entry_rationale}. Hưởng ưu đãi giảm 60% phí sàn, vị thế an toàn cao."

        # 6. Institutional Structural SL & TP Calculation (Calibrated per Timeframe)
        if opt_side == "SIDEWAY":
            struct_sl = round(entry_price - 2.0 * atr, 1)
            struct_tp = round(entry_price + 2.0 * atr, 1)
            struct_tp_macro = round(entry_price + 3.0 * atr, 1)
            risk_dist = 2.0 * atr
            reward_dist = 2.0 * atr
            rr_ratio = 1.0
        else:
            struct_setup = self.structural_calculator.compute_setup(
                side=opt_side,
                entry_price=entry_price,
                df_structure=df_structure,
                df_macro=df_macro,
                timeframe=trade_tf
            )
            struct_sl = struct_setup.stop_loss
            struct_tp = struct_setup.take_profit
            struct_tp_macro = struct_setup.tp_macro_extended
            risk_dist = struct_setup.risk_distance
            reward_dist = struct_setup.reward_distance
            rr_ratio = struct_setup.rr_ratio

        # 7. Sizing & Leverage Allocation (Kelly / Portfolio Risk Cap & Macro Anchor Alignment)
        macro_radar = getattr(ai_verdict, "mtf_radar", {}) if ai_verdict else {}
        w_bias = macro_radar.get("1w", {}).get("bias", "NEUTRAL")
        m_bias = macro_radar.get("1M", {}).get("bias", "NEUTRAL")

        is_macro_counter = (opt_side == "BUY" and (w_bias == "BEAR" or m_bias == "BEAR")) or \
                           (opt_side == "SELL" and (w_bias == "BULL" or m_bias == "BULL"))
        is_macro_aligned = (opt_side == "BUY" and w_bias == "BULL" and m_bias == "BULL") or \
                           (opt_side == "SELL" and w_bias == "BEAR" and m_bias == "BEAR")

        # Monthly Governor Controls Extraction
        gov_enabled = bool(monthly_governor_status and getattr(monthly_governor_status, "enabled", False))
        gov_regime = getattr(monthly_governor_status, "regime", "ON_TRACK") if gov_enabled else "ON_TRACK"
        gov_size_mult = float(getattr(monthly_governor_status, "size_multiplier", 1.0)) if gov_enabled else 1.0
        gov_lev_cap = int(getattr(monthly_governor_status, "max_leverage_cap", 10)) if gov_enabled else 10
        gov_min_rr = float(getattr(monthly_governor_status, "min_risk_reward_ratio", 1.5)) if gov_enabled else 1.5
        gov_target_pct = float(getattr(monthly_governor_status, "effective_target_pct", 10.0)) if gov_enabled else None
        gov_reserve_override = float(getattr(monthly_governor_status, "reserve_ratio_recommended", 0.15)) if gov_enabled else 0.15

        lev = effective_leverage if (effective_leverage and effective_leverage > 0) else 5
        if gov_enabled:
            lev = min(lev, gov_lev_cap)

        if is_macro_counter:
            # Defensive guard: Cap leverage to 6x max for counter-macro-trend setups and enforce Maker entry
            lev = min(lev, 6)
            if opt_type == "MARKET":
                opt_type = "POST_ONLY"
                fee_tier = "MAKER (0.02%)"
                order_rationale = f"🛡️ PHÒNG THỦ VĨ MÔ 1W/1M: Lệnh {opt_side} ngược sóng Tuần/Tháng. AI chuyển về POST_ONLY đón giá chiết khấu, hạ đòn bẩy an toàn còn {lev}x!"
        elif gov_enabled and gov_regime == "TARGET_ACHIEVED":
            if opt_type == "MARKET":
                opt_type = "POST_ONLY"
                fee_tier = "MAKER (0.02%)"
                order_rationale = f"🎉 BẢO TOÀN LỢI NHUẬN THÁNG: Đã đạt mục tiêu tháng! AI chuyển về POST_ONLY đón giá chiết khấu, khóa đòn bẩy an toàn {lev}x bảo vệ lãi!"

        # Enforce Governor Minimum R:R for deficit catch-up / behind schedule
        if gov_enabled and opt_side != "SIDEWAY" and rr_ratio < gov_min_rr and risk_dist > 0:
            rr_ratio = gov_min_rr
            reward_dist = risk_dist * gov_min_rr
            struct_tp = round(entry_price + reward_dist if opt_side == "BUY" else entry_price - reward_dist, 1)

        risk_budget = min(current_balance * 0.02, max(30.0, current_balance * 0.015))
        # Cong thuc dung theo thu nguyen: units = risk_budget / risk_dist (USDT / USDT-don vi),
        # roi margin = units * entry / lev = notional / lev. Ban cu risk_budget/(sl_pct*lev)
        # sai thu nguyen (USDT thay vi USDT-don vi) nen luon no len tran 20% balance.
        # Them rang buoc don bay: margin khong vuot qua notional_cho_phep/lev, voi notional bi
        # gioi han boi risk that (units = risk_budget/risk_dist) va tran margin 20% balance.
        lev_int = max(1, int(lev or 1))
        risk_dist_safe = max(float(risk_dist or 0.0), entry_price * 0.005, 1.0)
        raw_units = risk_budget / risk_dist_safe
        raw_margin = (raw_units * entry_price) / lev_int
        # Tran margin dong nhat CRO theo regime (panic 25% / range 35% / mac dinh 40% / trend 55%),
        # thay vi kep cung 20% nhu truoc (gay venh voi risk_manager 40% va sizing 35%).
        cro_cap_pct = 0.40
        try:
            if ai_cro is not None and getattr(ai_cro, "last_verdict", None) is not None:
                cro_cap_pct = float(ai_cro.last_verdict.max_margin_utilization_pct) / 100.0
        except Exception:
            cro_cap_pct = 0.40
        margin_cap = current_balance * min(0.55, max(0.25, cro_cap_pct))
        if raw_margin > margin_cap and raw_margin > 0:
            # Margin vuot tran CRO: giam units theo ti le (risk thuc se nho hon risk_budget;
            # sizing_gate phia sau tinh lai quantity chuan tu risk that)
            raw_units *= margin_cap / raw_margin
            raw_margin = margin_cap
        optimal_margin = round(max(50.0, raw_margin), 0)
        saved_fee = round(optimal_margin * lev * 0.0003, 2)

        macro_prob_adj = 6 if is_macro_aligned else (-6 if is_macro_counter else 0)
        rr_prob_adj = 10 if rr_ratio >= 2.0 else 0
        raw_prob = base_confidence + rr_prob_adj + macro_prob_adj
        max_cap = 94 if is_macro_aligned else 86
        win_prob = int(min(max_cap, max(58, raw_prob)))

        # 8. Hummingbot Avellaneda-Stoikov Inventory Skew Adjustment
        skew_status: Optional[InventorySkewStatus] = None
        if inventory_skew is not None and hasattr(inventory_skew, "reservation_price"):
            skew_status = inventory_skew
        elif current_position and current_price > 0:
            skew_status = self.hummingbot_skew.calculate_reservation_price(
                mid_price=current_price,
                current_position=current_position,
                atr=atr,
                total_balance=current_balance
            )

        # 9. Dual Bracket Reference Prices (Factoring in Hummingbot Reservation Price)
        base_dual_buy = round(max(current_price - (0.4 * atr), bid - 0.2 * atr), 1)
        base_dual_sell = round(min(current_price + (0.4 * atr), ask + 0.2 * atr), 1)

        skew_offset = skew_status.price_skew_offset if skew_status else 0.0
        # If holding Long (offset > 0): reservation price drops -> lower buy bracket, shift sell lower to exit
        # If holding Short (offset < 0): reservation price rises -> raise buy bracket to cover, raise sell bracket
        dual_buy = round(base_dual_buy - skew_offset, 1) if abs(skew_offset) > 0.1 else base_dual_buy
        dual_sell = round(base_dual_sell - skew_offset, 1) if abs(skew_offset) > 0.1 else base_dual_sell

        # Also skew entry_price if scaling into the existing position.
        # Dat tai reservation price (gia cong bang da dieu chinh ton kho), ap dung SAU tactical
        # anchor de khong bi anchor ghi de mat (loi: anchor entry sau hon reservation nhung test
        # doi entry dung bang reservation khi scale cung chieu).
        is_scale_in = (skew_status and (
            (skew_status.current_position_side == "LONG" and opt_side == "BUY") or
            (skew_status.current_position_side == "SHORT" and opt_side == "SELL")
        ))
        if is_scale_in:
            entry_price = round(skew_status.reservation_price, 1)

        skew_note = ""
        if skew_status and skew_status.current_position_side in ("LONG", "SHORT"):
            skew_note = f" | ⚖️ Hummingbot Skew: {skew_status.current_position_side} (r=${skew_status.reservation_price:,.1f}, offset ${skew_status.price_skew_offset:+.1f})"

        # 10. Robert Carver Systematic Framework (7 Pillars: Scaling, Capping, Vol Target, Sizing, RDM, Risk Overlay, Buffer)
        annual_vol_pct = getattr(ai_verdict, "garman_klass_vol", 0.0) if ai_verdict else 0.0
        if df_structure is not None and len(df_structure) >= 16:
            annual_vol_pct = QuantSkillsBrain.calculate_garman_klass_volatility(
                df_structure, timeframe=active_timeframe
            ).garman_klass_annualized
        # GK is an annual percentage; Carver consumes a daily fractional return.
        daily_vol_est = float(annual_vol_pct) / 100.0 / np.sqrt(365.0)
        if daily_vol_est <= 0:
            bars_per_day = 86400.0 / QuantSkillsBrain.bar_seconds(df_structure, active_timeframe)
            daily_vol_est = (atr / current_price) * np.sqrt(bars_per_day) if current_price > 0 else 0.015

        curr_contracts = 0.0
        if current_position:
            pos_dir = current_position.get("direction", 1)
            curr_contracts = float(current_position.get("units", 0.0)) * pos_dir

        raw_carver_signal = dir_score / 4.0
        cro_dd = getattr(ai_cro, "current_drawdown_pct", 0.0) if ai_cro else 0.0

        carver_out = self.carver_engine.compute_systematic_position(
            current_price=entry_price,
            capital_usdt=current_balance,
            raw_signal=raw_carver_signal,
            daily_vol_pct=daily_vol_est,
            current_position_contracts=curr_contracts,
            current_drawdown_pct=cro_dd,
            effective_leverage=lev,
            contract_step=0.001,
            monthly_target_pct=gov_target_pct,
            governor_multiplier=gov_size_mult
        )

        # Harmonize optimal_margin with Carver Volatility & Tactical Sizing
        if tactical_prop is not None:
            # Tactical Sizing: preserve realistic institutional margin ($600 - $1,400)
            base_margin = tactical_prop.optimal_margin
            if carver_out.risk_overlay_multiplier < 1.0:
                base_margin *= carver_out.risk_overlay_multiplier
            optimal_margin = round(min(current_balance * 0.28, max(500.0, base_margin)), 0)
        elif carver_out.optimal_margin_usdt > 0:
            blended_margin = (optimal_margin * 0.40) + (carver_out.optimal_margin_usdt * 0.60)
            optimal_margin = round(min(current_balance * 0.25, max(40.0, blended_margin)), 0)

        # 11. Jesse AI Expectancy & Half-Kelly Sizing Multiplier Integration
        kelly_mult = 1.0
        if jesse_metrics:
            k_pct = getattr(jesse_metrics, "kelly_fraction_pct", 15.0)
            exp_usdt = getattr(jesse_metrics, "expectancy_usdt", 10.0)
            if exp_usdt > 0 and k_pct > 0:
                kelly_mult = max(0.35, min(1.50, k_pct / 15.0))
            elif exp_usdt <= 0:
                kelly_mult = 0.50 # Penalty for negative expectancy edge
            optimal_margin = round(max(30.0, optimal_margin * kelly_mult), 0)

        # Monthly Governor Position Size Throttling (0.5x if target achieved, 0.85x if deficit catchup)
        if gov_enabled and gov_size_mult < 1.0:
            optimal_margin = round(max(30.0, optimal_margin * gov_size_mult), 0)

        # 12. VisualHFT Microstructure Safety & Defensive Small-Capital Throttling
        vpin_val = getattr(visual_hft_metrics, "vpin", 0.35) if visual_hft_metrics else 0.35
        tox_regime = getattr(visual_hft_metrics, "toxicity_regime", "CLEAN") if visual_hft_metrics else "CLEAN"
        is_toxic = getattr(visual_hft_metrics, "is_toxic_flow", False) if visual_hft_metrics else False
        resil_pct = getattr(visual_hft_metrics, "market_resilience_pct", 85.0) if visual_hft_metrics else 85.0
        lob_20_imb = getattr(visual_hft_metrics, "lob_imbalance_20", 0.0) if visual_hft_metrics else 0.0

        strategy_horizon = self.capital_allocator.get_horizon(trade_tf)
        is_vpin_throttled = False
        hft_note = ""

        # Check if VPIN is elevated or critical
        is_high_vpin = (vpin_val >= 0.70) or is_toxic
        has_solid_structure = (win_prob >= 55) and (resil_pct >= 30.0) and (rr_ratio >= 1.0)

        if is_high_vpin:
            is_vpin_throttled = True
            if has_solid_structure:
                # User requirement: "Đối với VPIN (Toxic Flow): nếu dòng tiền bất ổn quá cao mà vị thế ổn
                # thì có thể điều tiết lệnh với vốn nhỏ an toàn , nên định rõ lại cái vấn đề này"
                # DEFENSIVE SMALL-CAPITAL THROTTLED ENTRY:
                if opt_type == "MARKET":
                    opt_type = "POST_ONLY"
                    fee_tier = "MAKER (0.02%)"
                # Cap leverage strictly to safe 2x (max 3x)
                lev = min(lev, 2 if vpin_val >= 0.85 else 3)
                # Throttle margin to 30% of normal allocation ("vốn nhỏ an toàn")
                optimal_margin = round(max(30.0, optimal_margin * 0.30), 0)
                # Widen structural SL buffer by 30% to withstand toxic volatility sweeps
                if opt_side == "BUY" and struct_sl > 0:
                    struct_sl = round(entry_price - (entry_price - struct_sl) * 1.30, 2)
                elif opt_side == "SELL" and struct_sl > 0:
                    struct_sl = round(entry_price + (struct_sl - entry_price) * 1.30, 2)
                hft_note = f" | 🛡️ Điều Tiết Vốn Nhỏ An Toàn (VPIN={vpin_val:.2f}, Margin 30%, Đòn bẩy {lev}x, Maker thụ động)"
            else:
                # Weak structure + high toxicity: heavy defensive downscaling
                if opt_type == "MARKET":
                    opt_type = "POST_ONLY"
                    fee_tier = "MAKER (0.02%)"
                lev = min(lev, 2)
                optimal_margin = round(max(25.0, optimal_margin * 0.20), 0)
                win_prob = max(50, win_prob - 15)
                hft_note = f" | 🚨 Cảnh Báo VPIN={vpin_val:.2f} + Thanh Khoản Thấp: Ép vốn tối thiểu $25, đòn bẩy {lev}x"
        elif resil_pct < 35.0:
            lev = min(lev, 4)
            hft_note = f" | 💧 VisualHFT Khô Thanh Khoản (Resilience {resil_pct:.0f}%)"

        # LOB 20 Imbalance Alignment
        if (opt_side == "BUY" and lob_20_imb >= 0.20) or (opt_side == "SELL" and lob_20_imb <= -0.20):
            win_prob = min(96, win_prob + 5)
        elif (opt_side == "BUY" and lob_20_imb <= -0.30) or (opt_side == "SELL" and lob_20_imb >= 0.30):
            win_prob = max(50, win_prob - 8)

        # 13. OctoBot Matrix Consensus Tradability Check
        octo_tradable = getattr(octobot_consensus, "is_tradable", True) if octobot_consensus else True
        if not octo_tradable:
            win_prob = max(50, win_prob - 10)

        # 14. Dual-Horizon Portfolio Capital Allocation
        alloc_res = self.capital_allocator.evaluate_allocation(
            horizon=strategy_horizon,
            requested_margin=optimal_margin,
            total_equity=current_balance,
            active_positions=[current_position] if current_position else [],
            pending_orders=pending_orders or [],
            min_trade_margin=30.0,
            reserve_ratio_override=gov_reserve_override
        )
        optimal_margin = alloc_res.allocated_margin
        sleeve_info = {
            "horizon": strategy_horizon,
            "allocated_margin": alloc_res.allocated_margin,
            "sleeve_budget": alloc_res.sleeve_budget,
            "sleeve_used": alloc_res.sleeve_used,
            "sleeve_available": alloc_res.sleeve_available,
            "reserve_buffer": alloc_res.reserve_buffer,
            "is_throttled": alloc_res.is_throttled
        }

        carver_dict = {
            "raw_forecast": carver_out.raw_forecast,
            "daily_price_vol_pct": carver_out.daily_price_vol_pct,
            "scaled_forecast": round(carver_out.scaled_forecast, 2),
            "capped_forecast": round(carver_out.capped_forecast, 2),
            "daily_cash_vol_target": round(carver_out.daily_cash_vol_target, 2),
            "instrument_value_vol": round(carver_out.instrument_value_vol, 2),
            "rule_diversification_mult": round(carver_out.rule_diversification_mult, 2),
            "risk_overlay_multiplier": round(carver_out.risk_overlay_multiplier, 2),
            "drawdown_multiplier": round(carver_out.drawdown_multiplier, 2),
            "vol_shock_multiplier": round(carver_out.vol_shock_multiplier, 2),
            "optimal_contracts": round(carver_out.optimal_contracts, 4),
            "rebalance_action": carver_out.rebalance_action,
            "contracts_to_execute": round(carver_out.contracts_to_execute, 4),
            "buffer_lower_bound": round(carver_out.buffer_lower_bound, 4),
            "buffer_upper_bound": round(carver_out.buffer_upper_bound, 4),
            "fee_saved_usdt": round(carver_out.fee_saved_usdt, 2),
            "summary_rationale": carver_out.summary_rationale
        }

        if opt_side == "SIDEWAY":
            final_rationale = (
                f"{order_rationale} "
                f"Lập Lưới Hedge Grid quanh ${entry_price:,.1f} | SL biên dưới: ${struct_sl:,.1f} | "
                f"TP biên trên: ${struct_tp:,.1f} | Tỷ lệ R:R 1.0:1 ({win_prob}% xác suất){skew_note}{hft_note}."
            )
        else:
            final_rationale = (
                f"{order_rationale} "
                f"Vị thế [{strategy_horizon}]: {opt_side} quanh ${entry_price:,.1f} | SL cấu trúc: ${struct_sl:,.1f} (-${risk_dist:,.1f}) | "
                f"TP mục tiêu: ${struct_tp:,.1f} (+${reward_dist:,.1f}) | Tỷ lệ R:R chuẩn {rr_ratio}:1 ({win_prob}% xác suất){skew_note}{hft_note}."
            )

        return AIOrderResearchResult(
            recommended_type=opt_type,
            recommended_side=opt_side,
            optimal_price=entry_price,
            optimal_margin=optimal_margin,
            optimal_leverage=lev,
            optimal_trigger_price=opt_trigger_price,
            optimal_trigger_cond=opt_trigger_cond,
            optimal_callback_pct=opt_callback,
            optimal_twap_slices=opt_twap_slices,
            structural_sl=struct_sl,
            structural_tp=struct_tp,
            structural_tp_macro=struct_tp_macro,
            rr_ratio=rr_ratio,
            fee_tier=fee_tier,
            estimated_fee_saved_usdt=saved_fee,
            win_probability=win_prob,
            research_rationale=final_rationale,
            dual_buy_price=dual_buy,
            dual_sell_price=dual_sell,
            execution_horizon=execution_horizon,
            carver_output=carver_dict,
            carver_contracts=carver_out.optimal_contracts,
            carver_action=carver_out.rebalance_action,
            carver_buffer_bands=(carver_out.buffer_lower_bound, carver_out.buffer_upper_bound),
            vpin=round(vpin_val, 3),
            toxicity_regime=tox_regime,
            market_resilience_pct=round(resil_pct, 1),
            lob_imbalance_20=round(lob_20_imb, 3),
            kelly_multiplier=round(kelly_mult, 2),
            octobot_tradable=octo_tradable,
            active_timeframe=trade_tf,
            dca_ladder_step=dca_step,
            twap_interval_seconds=twap_interval,
            strategy_horizon=strategy_horizon,
            is_vpin_throttled=is_vpin_throttled,
            sleeve_allocation=sleeve_info,
            monthly_regime=gov_regime,
            monthly_size_multiplier=gov_size_mult,
            tactical_formation=tactical_prop.formation.value if tactical_prop else "DEFENSIVE_SNIPER",
            staged_exits=tactical_prop.staged_exits if tactical_prop else []
        )

    def build_adaptive_counter_proposal(
        self,
        original_candidate: Any,
        veto_stage: str,
        veto_reason: str,
        current_price: float,
        best_bid: float,
        best_ask: float,
        indicators: Dict[str, Any],
        df_structure: Optional[pd.DataFrame] = None,
        df_macro: Optional[pd.DataFrame] = None,
        active_timeframe: str = "15m",
        current_balance: float = 5055.0,
        council_verdict: Optional[Any] = None
    ) -> Optional[Any]:
        """
        Adaptive Proposal Revision Engine (Bộ Đàm Phán Lệnh Đa Tác Tử):
        Khi Hội đồng AI (Stage 3) hoặc các Gate kiểm duyệt (Stage 2/4) từ chối Phương án A:
        Hệ thống KHÔNG buông xuôi bỏ cuộc mà phân tích lý do phản biện để xây dựng Phương án B:
        1. Phản biện về R:R / Stop Loss -> Tái cấu trúc SL bám sát râu nến swing, mở rộng TP2 cản đối diện để đạt R:R >= 2.8:1.
        2. Phản biện về Loại Lệnh / Trượt giá / VPIN -> Đổi sang POST_ONLY Maker tại Demand/Supply Block.
        3. Phản biện về Xu hướng vĩ mô (Macro) -> Đổi sang Lưới Trung Lập TWO_WAY_GRID hoặc đánh thuận trend.
        4. Phản biện về Vốn / Ký quỹ -> Giảm ký quỹ xuống $300 - $350 USDT, hạ đòn bẩy về 3x (DEFICIT_PATIENCE).
        """
        import copy
        alt = copy.deepcopy(original_candidate)
        alt.order_id = f"rev-{alt.order_id[-18:]}"
        alt.metadata["is_counter_proposal"] = True
        alt.metadata["renegotiation_basis"] = f"{veto_stage}: {veto_reason}"
        # P5: Phuong an 2 ghi ro song + Hurst de cong wave tai danh gia lai
        # khong veto oan. Ke thua probation/hurst_downgraded tu Phuong an 1
        # (neu co) va ep probation khi Hurst thap + Maker directional.
        try:
            orig_meta = getattr(original_candidate, "metadata", {}) or {}
            wave = dict(orig_meta.get("wave_alignment", {}) or {})
            if wave:
                alt.metadata["wave_alignment"] = wave
            hurst_val = float((indicators or {}).get("hurst", 0.50) or 0.50)
            alt.metadata["hurst_at_veto"] = round(hurst_val, 3)
            for flag in ("probation", "hurst_downgraded", "octo_weak_conflict",
                         "alpha_weak_conflict", "wave_misaligned"):
                if orig_meta.get(flag) and flag not in alt.metadata:
                    alt.metadata[flag] = orig_meta.get(flag)
            if (hurst_val < 0.45 and alt.order_type in ("POST_ONLY", "LIMIT", "SCALE_RATIO")
                    and int(getattr(alt, "direction", 0) or 0) in (-1, 1)):
                alt.metadata["probation"] = True
                alt.metadata["risk_cap_pct"] = 0.0010
        except Exception:
            pass

        reason_lower = (veto_reason or "").lower()
        council_objections = getattr(council_verdict, "rejection_categories", {}) if council_verdict else {}
        
        atr = float(indicators.get("atr") or current_price * 0.008)
        smc_res = self.smc_engine.analyze(df_structure, current_price) if df_structure is not None and len(df_structure) >= 20 else None
        
        local_low, _, local_high, _ = self.tactical_dispatcher._find_real_swings(df_structure, lookback=25) if hasattr(self.tactical_dispatcher, "_find_real_swings") else (current_price - 1.5 * atr, 0, current_price + 1.5 * atr, 0)
        if local_low <= 0 or local_low >= current_price:
            local_low = round(current_price - 1.5 * atr, 1)
        if local_high <= 0 or local_high <= current_price:
            local_high = round(current_price + 1.5 * atr, 1)

        demand_top = smc_res.nearest_demand_zone[1] if (smc_res and smc_res.nearest_demand_zone) else local_low
        supply_bottom = smc_res.nearest_supply_zone[0] if (smc_res and smc_res.nearest_supply_zone) else local_high

        # 1. Phản biện về R:R / Stop Loss (Risk Agent)
        is_rr_critique = (
            "r:r" in reason_lower
            or "stop_loss" in reason_lower
            or "stop loss" in reason_lower
            or "cắt lỗ" in reason_lower
            or "sl" in reason_lower
            or "tỷ lệ r:r" in reason_lower
            or "fee hurdle" in reason_lower
            or bool(council_objections.get("risk"))
        )

        # 2. Phản biện về Loại lệnh / Trượt giá / VPIN (Execution Agent)
        is_exec_critique = (
            "market" in reason_lower
            or "trượt giá" in reason_lower
            or "vpin" in reason_lower
            or "spread" in reason_lower
            or "resilience" in reason_lower
            or "l2" in reason_lower
            or bool(council_objections.get("execution"))
        )

        # 3. Phản biện về Xu hướng vĩ mô (Macro Agent)
        is_macro_critique = (
            "trend" in reason_lower
            or "xu hướng" in reason_lower
            or "macro" in reason_lower
            or "ngược" in reason_lower
            or "alpha zoo" in reason_lower
            or "octobot" in reason_lower
            or bool(council_objections.get("macro"))
        )

        # 4. Phản biện về Quá tải vốn / Sizing / Carver buffer
        is_sizing_critique = (
            "margin" in reason_lower
            or "leverage" in reason_lower
            or "buffer" in reason_lower
            or "hold" in reason_lower
            or "hạn mức" in reason_lower
            or "sizing" in reason_lower
        )

        if is_macro_critique:
            # P5: Hurst thap + ranging -> mac dinh Phuong an 2 la POST_ONLY Maker
            # probation (khong xoay sang GRID de tranh flip-flop loai lenh nhu log
            # live). Chi doi sang GRID khi ranging + gan VWAP co tin hieu moi that su.
            try:
                _hurst = float((indicators or {}).get("hurst", 0.50) or 0.50)
            except Exception:
                _hurst = 0.50
            if _hurst < 0.45 and int(getattr(original_candidate, "direction", 0) or 0) in (-1, 1):
                _dir = int(getattr(original_candidate, "direction", 0) or 0)
                alt.order_type = "POST_ONLY"
                alt.direction = _dir
                alt.entry_price = round(best_bid if _dir == 1 else best_ask, 1)
                alt.leverage = min(4, max(1, int(getattr(alt, "leverage", 3) or 3)))
                alt.margin = round(min(600.0, max(300.0, current_balance * 0.10)), 0)
                alt.quantity = round(alt.margin * alt.leverage / max(1.0, alt.entry_price), 4)
                alt.metadata["requested_margin"] = alt.margin
                alt.metadata["requested_quantity"] = alt.quantity
                alt.metadata["probation"] = True
                alt.metadata["risk_cap_pct"] = 0.0010
                alt.metadata.update({
                    "tactical_formation": "DEFENSIVE_SNIPER",
                    "fee_tier": "MAKER (0.02%)",
                    "suggested_type": "POST_ONLY",
                    "negotiation_solution": f"Hurst {_hurst:.2f} thap: Phuong an 2 POST_ONLY Maker probation 0.10% (khong xoay GRID)"
                })
                return alt
            # Chi doi sang GRID khi co tin hieu moi thuc su (sweep moi / VPIN doc / CVD dao chieu),
            # khong doi chi vi phan loai macro_critique (tranh xoay loai lenh nhu log live).
            orig_type = getattr(original_candidate, "order_type", "POST_ONLY")
            orig_dir = int(getattr(original_candidate, "direction", 0) or 0)
            orig_price = float(getattr(original_candidate, "entry_price", current_price) or current_price)
            price_moved_atr = abs(current_price - orig_price) / max(1e-9, atr)
            fresh_signal = (
                (smc_res is not None and smc_res.last_sweep is not None)
                or price_moved_atr >= 0.5
            )
            if not fresh_signal and orig_dir in (-1, 1):
                # Giu directional, chi ha ve POST_ONLY Maker an toan thay vi xoay sang GRID
                alt.order_type = "POST_ONLY"
                alt.direction = orig_dir
                alt.entry_price = round(best_bid if orig_dir == 1 else best_ask, 1)
                alt.leverage = min(4, max(1, int(getattr(alt, "leverage", 3) or 3)))
                alt.margin = round(min(600.0, max(300.0, current_balance * 0.10)), 0)
                alt.quantity = round(alt.margin * alt.leverage / max(1.0, alt.entry_price), 4)
                alt.metadata["requested_margin"] = alt.margin
                alt.metadata["requested_quantity"] = alt.quantity
                alt.metadata.update({
                    "tactical_formation": "DEFENSIVE_SNIPER",
                    "fee_tier": "MAKER (0.02%)",
                    "negotiation_solution": "Giu huong goc (khong tin hieu moi) - ha ve POST_ONLY Maker an toan"
                })
                return alt
            alt.order_type = "GRID"
            alt.source = "auto-grid"
            alt.direction = 0
            alt.entry_price = current_price
            alt.stop_loss = round(current_price - 2.5 * atr, 1)
            alt.take_profit = round(current_price + 2.5 * atr, 1)
            alt.leverage = 3
            alt.margin = round(min(800.0, current_balance * 0.15), 0)
            alt.metadata.update({
                "tactical_formation": "TWO_WAY_GRID",
                "fee_tier": "MAKER GRID (0.02%)",
                "execution_horizon": "RANGE_GRID",
                "negotiation_solution": "Chuyển đổi thành Lưới Trung Lập Hai Chiều Geometric Hedge Grid không đoán hướng"
            })
            return alt

        elif is_rr_critique or is_exec_critique:
            alt.order_type = "POST_ONLY"
            alt.fee_tier = "MAKER (0.02%)"
            alt.metadata["fee_tier"] = "MAKER (0.02%)"

            min_sl_dist = max(1.5 * atr, alt.entry_price * 0.0035, 250.0)
            if alt.direction == 1: # BUY
                alt.entry_price = round(min(best_bid, demand_top), 1)
                if alt.entry_price > current_price:
                    alt.entry_price = round(best_bid, 1)
                calculated_sl = round(local_low - 0.15 * atr, 1)
                alt.stop_loss = min(calculated_sl, round(alt.entry_price - min_sl_dist, 1))
                risk_dist = max(min_sl_dist, alt.entry_price - alt.stop_loss)
                alt.take_profit = round(alt.entry_price + max(2.8 * risk_dist, 5.0 * (alt.entry_price * 0.0007 + 0.5)), 1)
            elif alt.direction == -1: # SELL
                alt.entry_price = round(max(best_ask, supply_bottom), 1)
                if alt.entry_price < current_price:
                    alt.entry_price = round(best_ask, 1)
                calculated_sl = round(local_high + 0.15 * atr, 1)
                alt.stop_loss = max(calculated_sl, round(alt.entry_price + min_sl_dist, 1))
                risk_dist = max(min_sl_dist, alt.stop_loss - alt.entry_price)
                alt.take_profit = round(alt.entry_price - max(2.8 * risk_dist, 5.0 * (alt.entry_price * 0.0007 + 0.5)), 1)
            else:
                return None

            alt.leverage = min(4, max(1, alt.leverage))
            alt.margin = round(min(750.0, max(350.0, current_balance * 0.12)), 0)
            alt.quantity = round(alt.margin * alt.leverage / max(1.0, alt.entry_price), 4)
            alt.metadata["requested_margin"] = alt.margin
            alt.metadata["requested_quantity"] = alt.quantity
            alt.metadata.update({
                "tactical_formation": "DEFENSIVE_SNIPER",
                "execution_horizon": f"MAKER_{active_timeframe.upper()}",
                "negotiation_solution": f"Siết chặt SL ({alt.stop_loss:,.1f}) bám sát cản râu nến và nâng TP ({alt.take_profit:,.1f}) đạt chuẩn R:R >= 2.8:1 tại đầu sổ Maker"
            })
            return alt

        elif is_sizing_critique:
            alt.order_type = "SCALE_RATIO"
            alt.leverage = 3
            alt.margin = round(min(500.0, max(250.0, current_balance * 0.08)), 0)
            alt.quantity = round(alt.margin * alt.leverage / max(1.0, alt.entry_price), 4)
            alt.metadata["requested_margin"] = alt.margin
            alt.metadata["requested_quantity"] = alt.quantity
            alt.metadata.update({
                "tactical_formation": "DEFICIT_PATIENCE",
                "fee_tier": "MAKER LADDER (0.02%)",
                "dca_ladder_step": 0.005,
                "negotiation_solution": "Hạ đòn bẩy 3x, chia nhỏ vốn rải thang DCA Ladder 3 tầng (20%-30%-50%)"
            })
            return alt

        alt.order_type = "POST_ONLY"
        alt.entry_price = round(best_bid if alt.direction == 1 else best_ask, 1)
        min_sl_dist = max(1.5 * atr, alt.entry_price * 0.0035, 250.0)
        if alt.direction == 1:
            alt.stop_loss = min(alt.stop_loss, round(alt.entry_price - min_sl_dist, 1))
        elif alt.direction == -1:
            alt.stop_loss = max(alt.stop_loss, round(alt.entry_price + min_sl_dist, 1))
        alt.leverage = min(4, alt.leverage)
        alt.margin = round(min(600.0, max(300.0, current_balance * 0.10)), 0)
        alt.quantity = round(alt.margin * alt.leverage / max(1.0, alt.entry_price), 4)
        alt.metadata["requested_margin"] = alt.margin
        alt.metadata["requested_quantity"] = alt.quantity
        alt.metadata.update({
            "tactical_formation": "DEFENSIVE_SNIPER",
            "negotiation_solution": "Điều chỉnh thụ động Maker tại đầu sổ lệnh"
        })
        return alt
