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
from typing import Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np

from strategy.smart_money_concepts import SmartMoneyEngine, SMCAnalysisResult
from strategy.institutional_vwap import InstitutionalVWAPEngine, VWAPBandResult
from strategy.hummingbot_inventory_skew import HummingbotInventorySkewEngine, InventorySkewStatus
from strategy.carver_systematic_engine import CarverSystematicEngine, CarverSystematicOutput
from risk.structural_sl_tp import StructuralRiskCalculator, StructuralTradeSetup


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


class AIOrderResearcher:
    def __init__(self):
        self.structural_calculator = StructuralRiskCalculator(atr_buffer_mult=0.3)
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
        octobot_consensus: Optional[Any] = None
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
        absorption = getattr(order_flow_verdict, "absorption_divergence", "NONE") if order_flow_verdict else "NONE"

        radar_score = 0.0
        if isinstance(candle_confluence, dict):
            radar_score = candle_confluence.get("radar_score", 0.0)
        elif hasattr(candle_confluence, "radar_score"):
            radar_score = candle_confluence.radar_score

        # 3. Directional Synthesis (Multi-Factor Scoring)
        dir_score = (consensus_score * 0.40) + (radar_score * 0.25)
        
        # Factor CVD momentum
        if delta_momentum == "AGGRESSIVE_BUYING":
            dir_score += 15.0
        elif delta_momentum == "AGGRESSIVE_SELLING":
            dir_score -= 15.0
            
        # Factor Absorption
        if absorption == "BULLISH_ABSORPTION":
            dir_score += 12.0
        elif absorption == "BEARISH_ABSORPTION":
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
        if dir_score >= 12.0:
            opt_side = "BUY"
        elif dir_score <= -12.0:
            opt_side = "SELL"
        else:
            if vwap_res and current_price < vwap_res.vwap:
                opt_side = "BUY"
            else:
                opt_side = "SELL"

        # 4. Realistic Micro-Structure Entry Price Discovery (Within 0.02% - 0.45% of Live Price)
        execution_horizon = "IMMEDIATE"
        entry_price = current_price
        entry_rationale = ""

        if opt_side == "BUY":
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
                entry_price = round(ask, 1)
                entry_rationale = f"Treo tại đầu sổ Best Ask (${entry_price:,.1f}) đảm bảo Maker 0.02%"
                execution_horizon = "IMMEDIATE"

            if entry_price < current_price:
                entry_price = round(ask, 1)
            elif (entry_price - current_price) > 0.75 * atr:
                entry_price = round(current_price + (0.15 * atr), 1)

        # 5. Dynamic Order Type Selection Engine
        opt_type = "POST_ONLY"
        opt_trigger_price = 0.0
        opt_trigger_cond = "ABOVE"
        opt_callback = round(max(0.4, min(1.8, (atr / current_price) * 100.0 * 0.75)), 2)
        opt_twap_slices = 5
        fee_tier = "MAKER (0.02%)"

        bb_width = indicators.get("bb_width", 0.04)

        if abs(dir_score) >= 60.0 and delta_momentum in ("AGGRESSIVE_BUYING", "AGGRESSIVE_SELLING") and adx >= 28.0:
            opt_type = "MARKET"
            entry_price = current_price
            fee_tier = "TAKER (0.05%)"
            execution_horizon = "IMMEDIATE"
            order_rationale = f"⚡ BÙNG NỔ ĐỘNG LƯỢNG: CVD {delta_momentum} áp đảo, ADX {adx:.1f}. AI khuyến nghị vào lệnh MARKET ngay lập tức để không lỡ sóng!"

        elif (bb_width < 0.015 or adx < 16.0) and regime == "RANGING_SIDEWAY":
            opt_type = "CONDITIONAL"
            fee_tier = "STOP-LIMIT (MAKER/TAKER)"
            execution_horizon = "BREAKOUT"
            if opt_side == "BUY":
                opt_trigger_price = round(current_price + (0.25 * atr), 1)
                opt_trigger_cond = "ABOVE"
                entry_price = round(opt_trigger_price + 0.5, 1)
            else:
                opt_trigger_price = round(current_price - (0.25 * atr), 1)
                opt_trigger_cond = "BELOW"
                entry_price = round(opt_trigger_price - 0.5, 1)
            order_rationale = f"📦 THỊ TRƯỜNG NÉN BIÊN (Squeeze): Bollinger co thắt. AI khuyến nghị Stop-Limit đón Breakout tại ngưỡng ${opt_trigger_price:,.1f}!"

        elif regime in ("TRENDING_BULL", "TRENDING_BEAR") and abs(dir_score) >= 35.0:
            opt_type = "TRAILING_STOP"
            entry_price = current_price
            fee_tier = "TRAILING TAKER"
            execution_horizon = "IMMEDIATE"
            order_rationale = f"🏄 BÁM SÓNG XU HƯỚNG: Xu hướng {regime} mạnh ({dir_score:+.1f}đ). AI khuyến nghị Trailing Stop tự động bám đỉnh/đáy rút râu {opt_callback}%!"

        elif smc_res and smc_res.last_sweep is not None:
            opt_type = "SCALE_RATIO"
            fee_tier = "MAKER LADDER (0.02%)"
            execution_horizon = "SWEEP_REVERSAL"
            order_rationale = f"🎯 QUÉT THANH KHOẢN SMC: Phát hiện {smc_res.last_sweep.type}. AI khuyến nghị Rải Thang 3 Tầng (20%-30%-50%) quanh ${entry_price:,.1f} đón râu nến!"

        elif eff_spread > 5.0 or (atr / current_price) > 0.014 or regime == "VOLATILE_PANIC":
            opt_type = "TWAP"
            opt_twap_slices = 5
            entry_price = current_price
            fee_tier = "TWAP SLICES"
            execution_horizon = "IMMEDIATE"
            order_rationale = f"🌊 BIẾN ĐỘNG MẠNH / SPREAD GIÃN: Spread ${eff_spread:.1f}. AI khuyến nghị TWAP chia đều 5 lát cắt chống trượt giá sàn!"

        else:
            opt_type = "POST_ONLY"
            fee_tier = "MAKER (0.02%)"
            order_rationale = f"🛡️ BẢO VỆ MAKER 0.02%: {entry_rationale}. Hưởng ưu đãi giảm 60% phí sàn, vị thế an toàn cao."

        # 6. Institutional Structural SL & TP Calculation (Calibrated per Timeframe)
        struct_setup = self.structural_calculator.compute_setup(
            side=opt_side,
            entry_price=entry_price,
            df_structure=df_structure,
            df_macro=df_macro,
            timeframe=active_timeframe
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

        lev = effective_leverage if (effective_leverage and effective_leverage > 0) else 5
        if is_macro_counter:
            # Defensive guard: Cap leverage to 6x max for counter-macro-trend setups and enforce Maker entry
            lev = min(lev, 6)
            if opt_type == "MARKET":
                opt_type = "POST_ONLY"
                fee_tier = "MAKER (0.02%)"
                order_rationale = f"🛡️ PHÒNG THỦ VĨ MÔ 1W/1M: Lệnh {opt_side} ngược sóng Tuần/Tháng. AI chuyển về POST_ONLY đón giá chiết khấu, hạ đòn bẩy an toàn còn {lev}x!"

        risk_budget = min(current_balance * 0.02, max(30.0, current_balance * 0.015))
        sl_pct = max(risk_dist / entry_price, 0.005)
        
        raw_margin = risk_budget / (sl_pct * lev)
        optimal_margin = round(min(current_balance * 0.20, max(50.0, raw_margin)), 0)
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

        # Also skew entry_price if scaling into the existing position
        if skew_status and skew_status.current_position_side == opt_side:
            if opt_side == "BUY" and entry_price > skew_status.reservation_price:
                entry_price = round(min(entry_price, skew_status.reservation_price), 1)
            elif opt_side == "SELL" and entry_price < skew_status.reservation_price:
                entry_price = round(max(entry_price, skew_status.reservation_price), 1)

        skew_note = ""
        if skew_status and skew_status.current_position_side in ("LONG", "SHORT"):
            skew_note = f" | ⚖️ Hummingbot Skew: {skew_status.current_position_side} (r=${skew_status.reservation_price:,.1f}, offset ${skew_status.price_skew_offset:+.1f})"

        # 10. Robert Carver Systematic Framework (7 Pillars: Scaling, Capping, Vol Target, Sizing, RDM, Risk Overlay, Buffer)
        daily_vol_est = getattr(ai_verdict, "garman_klass_vol", 0.0) if ai_verdict else 0.0
        if daily_vol_est <= 0.002:
            daily_vol_est = (atr / current_price) if current_price > 0 else 0.015

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
            contract_step=0.001
        )

        # Harmonize optimal_margin with Carver Volatility-Targeted Position Sizing
        if carver_out.optimal_margin_usdt > 0:
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

        # 12. VisualHFT Microstructure Safety & Order Gating
        vpin_val = getattr(visual_hft_metrics, "vpin", 0.35) if visual_hft_metrics else 0.35
        tox_regime = getattr(visual_hft_metrics, "toxicity_regime", "CLEAN") if visual_hft_metrics else "CLEAN"
        is_toxic = getattr(visual_hft_metrics, "is_toxic_flow", False) if visual_hft_metrics else False
        resil_pct = getattr(visual_hft_metrics, "market_resilience_pct", 85.0) if visual_hft_metrics else 85.0
        lob_20_imb = getattr(visual_hft_metrics, "lob_imbalance_20", 0.0) if visual_hft_metrics else 0.0

        hft_note = ""
        if is_toxic:
            # Dangerous informed toxic flow sweep in progress:
            # Downgrade market taker orders to post-only maker to avoid adverse slippage
            if opt_type == "MARKET":
                opt_type = "POST_ONLY"
                fee_tier = "MAKER (0.02%)"
            lev = min(lev, 3) # Cap leverage to 3x max
            optimal_margin = round(optimal_margin * 0.50, 0) # Cut position in half
            win_prob = max(50, win_prob - 12)
            hft_note = f" | 🚨 VisualHFT Toxic Flow VPIN={vpin_val:.2f} (Hạ đòn bẩy {lev}x, giảm 50% margin)"
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

        carver_dict = {
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

        final_rationale = (
            f"{order_rationale} "
            f"Vị thế: {opt_side} quanh ${entry_price:,.1f} | SL cấu trúc: ${struct_sl:,.1f} (-${risk_dist:,.1f}) | "
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
            octobot_tradable=octo_tradable
        )
