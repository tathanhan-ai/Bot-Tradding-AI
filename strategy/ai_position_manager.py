"""
AI Dynamic Position Coordinator & Smart Exit Optimizer
Continuously coordinates open positions on every price tick:
1. Dynamic Breakeven & Risk-Free Trade Lock (Dời SL về Entry + Phí Sàn khi lãi >= +0.8R)
2. Dynamic TP Expansion (Nới TP theo cản khung lớn 1h/4h khi sóng mạnh để gồng lời dài hơi)
3. Trailing Stop along Swing Highs/Lows with ATR buffer
4. AI Early Take-Profit (Chốt lời khi RSI quá mua/quá bán cực đại cạn kiệt lực đẩy)
5. AI Early Cut-Loss (Cắt lỗ sớm ở -0.4R khi cấu trúc thị trường bị bẻ gãy)
"""
import time
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

from strategy.freqtrade_roi import FreqtradeROIEngine


@dataclass
class AIPositionDecision:
    action: str             # 'HOLD', 'LOCK_BREAKEVEN', 'UPDATE_TRAILING', 'EXPAND_TAKE_PROFIT', 'AI_TAKE_PROFIT', 'AI_CUT_LOSS'
    new_stop_loss: Optional[float] = None
    new_take_profit: Optional[float] = None
    reason: str = ""
    status_display: str = ""


class AIPositionCoordinator:
    def __init__(self):
        self.roi_engine = FreqtradeROIEngine(default_timeframe="15m")

    def evaluate_position(
        self,
        pos: dict,
        current_price: float,
        indicators: dict,
        ai_verdict: Any,
        fee_engine: Any,
        df_macro: Optional[Any] = None
    ) -> AIPositionDecision:
        decision = self._evaluate_position_logic(pos, current_price, indicators, ai_verdict, fee_engine, df_macro)

        # Enforce strict Freqtrade Monotonic Ratchet invariant: Stoploss can NEVER regress backwards
        if decision.new_stop_loss is not None:
            direction = pos.get("direction", 1)
            current_sl = pos.get("stop_loss", 0.0)
            if direction == 1 and current_sl > 0:
                decision.new_stop_loss = round(max(decision.new_stop_loss, current_sl), 2)
            elif direction == -1 and current_sl > 0:
                decision.new_stop_loss = round(min(decision.new_stop_loss, current_sl), 2)
        return decision

    def _evaluate_position_logic(
        self,
        pos: dict,
        current_price: float,
        indicators: dict,
        ai_verdict: Any,
        fee_engine: Any,
        df_macro: Optional[Any] = None
    ) -> AIPositionDecision:
        direction = pos["direction"]
        entry = pos["entry_price"]
        breakeven = pos.get("breakeven_price", entry)
        initial_risk = pos.get("initial_risk", max(abs(entry - pos["stop_loss"]), current_price * 0.005))
        current_sl = pos["stop_loss"]
        current_tp = pos["take_profit"]
        is_risk_free = pos.get("is_risk_free", False)
        tp_expanded = pos.get("tp_expanded", False)

        rsi = indicators.get("rsi", 50.0)
        atr = indicators.get("atr", current_price * 0.006)

        gain_usdt = (current_price - entry) * pos["units"] * direction
        r_multiple = (current_price - entry) / initial_risk if direction == 1 else (entry - current_price) / initial_risk

        # =========================================================================
        # 0. KHÓA CHỈNH TAY THỦ CÔNG (MANUAL TP/SL LOCK)
        # =========================================================================
        if pos.get("is_manual_tpsl", False):
            return AIPositionDecision(
                action="HOLD",
                reason="Khóa TP/SL Chỉnh Tay (AI tôn trọng mục tiêu Stop Loss & Take Profit của người dùng)",
                status_display="🔒 TP/SL Chỉnh Tay (Manual)"
            )

        # =========================================================================
        # 0. FREQTRADE MINIMAL ROI TIME-DECAYING EXIT (Chốt lời suy giảm theo thời gian)
        # =========================================================================
        should_exit_roi, roi_reason, profit_pct, target_roi = self.roi_engine.evaluate_roi_exit(pos, current_price)
        if should_exit_roi and gain_usdt > 0:
            return AIPositionDecision(
                action="AI_TAKE_PROFIT",
                reason=roi_reason,
                status_display=f"💰 Chốt Lời Freqtrade ROI (+{profit_pct:.2f}%)"
            )

        # =========================================================================
        # FREQTRADE MONOTONIC RATCHET TRAILING STOP (Khóa Lãi Bậc Thang Bất Biến)
        # =========================================================================
        if r_multiple >= 2.0:
            target_ratchet_sl = (entry + 1.0 * initial_risk) if direction == 1 else (entry - 1.0 * initial_risk)
            if r_multiple >= 3.0:
                trail_atr_sl = (current_price - 0.8 * atr) if direction == 1 else (current_price + 0.8 * atr)
                target_ratchet_sl = max(target_ratchet_sl, trail_atr_sl) if direction == 1 else min(target_ratchet_sl, trail_atr_sl)

            target_ratchet_sl = round(target_ratchet_sl, 2)
            # Monotonic rule: SL can strictly only move towards profit
            if (direction == 1 and target_ratchet_sl > current_sl and target_ratchet_sl < current_price) or \
               (direction == -1 and target_ratchet_sl < current_sl and target_ratchet_sl > current_price):
                return AIPositionDecision(
                    action="UPDATE_TRAILING",
                    new_stop_loss=target_ratchet_sl,
                    reason=f"FREQTRADE RATCHET KHÓA LÃI 🛡️ (Dời SL lên ${target_ratchet_sl:,.2f} để bảo vệ +{r_multiple:.1f}R lợi nhuận)",
                    status_display=f"🚀 Freqtrade Ratchet: ${target_ratchet_sl:,.1f}"
                )

        # =========================================================================
        # 1. MILESTONE 1: DỜI SL VỀ ENTRY (KHÓA HÒA VỐN + PHÍ SÀN = RISK-FREE TRADE)
        # =========================================================================
        if not is_risk_free:
            # Trigger ONLY when confirmed in profit >= +1.0R (never on tiny 1-tick fluctuations)
            if r_multiple >= 1.0 and gain_usdt > 0:
                if direction == 1 and current_sl < breakeven:
                    # SL must always be strictly below current price with at least 0.3 ATR breathing room
                    new_sl = round(min(breakeven, current_price - (0.3 * atr)), 2)
                    if new_sl > current_sl and new_sl < current_price:
                        pos["is_risk_free"] = True
                        return AIPositionDecision(
                            action="LOCK_BREAKEVEN",
                            new_stop_loss=new_sl,
                            reason="DỜI SL VỀ ENTRY 🛡️ (Khóa Hòa Vốn Sau Phí - Lệnh Trở Thành Risk-Free)",
                            status_display="🛡️ ĐÃ KHÓA HÒA VỐN (RISK-FREE TRADE)"
                        )
                elif direction == -1 and current_sl > breakeven:
                    new_sl = round(max(breakeven, current_price + (0.3 * atr)), 2)
                    if new_sl < current_sl and new_sl > current_price:
                        pos["is_risk_free"] = True
                        return AIPositionDecision(
                            action="LOCK_BREAKEVEN",
                            new_stop_loss=new_sl,
                            reason="DỜI SL VỀ ENTRY 🛡️ (Khóa Hòa Vốn Sau Phí - Lệnh Trở Thành Risk-Free)",
                            status_display="🛡️ ĐÃ KHÓA HÒA VỐN (RISK-FREE TRADE)"
                        )

        # =========================================================================
        # 1.5 MILESTONE 1.5: CHỐT LỜI TỪNG PHẦN 50/50 (PARTIAL TAKE PROFIT) TẠI TP1
        # =========================================================================
        if not pos.get("partial_tp_done", False):
            dist_to_tp = abs(current_tp - entry)
            progress_pct = (abs(current_price - entry) / max(dist_to_tp, 1.0)) if dist_to_tp > 0 else 0.0
            if r_multiple >= 1.2 or progress_pct >= 0.60:
                pos["partial_tp_done"] = True
                pos["is_risk_free"] = True
                safe_sl = max(current_sl, breakeven) if direction == 1 else min(current_sl, breakeven)
                return AIPositionDecision(
                    action="PARTIAL_TAKE_PROFIT",
                    new_stop_loss=safe_sl,
                    reason=f"CHỐT LỜI 50% TẠI TP1 💰 (${current_price:,.2f} | +{r_multiple:.2f}R)",
                    status_display="💰 ĐÃ CHỐT 50% | 50% CÒN LẠI GỒNG RISK-FREE"
                )

        # =========================================================================
        # 2. MILESTONE 2: NỚI RỘNG TP THEO CẢN LỚN ĐỂ GỒNG LÃI DÀI HƠI (TP EXPANSION)
        # =========================================================================
        # When price reaches >= 85% of initial TP, check if momentum supports riding big trend
        dist_total = abs(current_tp - entry)
        curr_progress = (current_price - entry) / dist_total if dist_total > 0 else 0.0

        if not tp_expanded and curr_progress >= 0.85:
            # Check momentum: Not extremely exhausted yet (RSI between 55-72 for Long, 28-45 for Short)
            momentum_healthy = (55.0 <= rsi <= 74.0) if direction == 1 else (26.0 <= rsi <= 45.0)
            macro_trend_aligned = (ai_verdict and ai_verdict.regime == ("TRENDING_BULL" if direction == 1 else "TRENDING_BEAR"))

            if momentum_healthy or macro_trend_aligned:
                # Find higher timeframe target
                macro_target = 0.0
                if ai_verdict:
                    macro_target = ai_verdict.resistance_price if direction == 1 else ai_verdict.support_price

                if direction == 1:
                    new_tp = round(max(current_tp + (2.5 * atr), macro_target * 0.998), 2)
                    locked_sl = round(entry + (0.7 * initial_risk), 2)  # Lock in at least +0.7R profit
                else:
                    new_tp = round(min(current_tp - (2.5 * atr), macro_target * 1.002), 2) if macro_target > 0 else round(current_tp - (2.5 * atr), 2)
                    locked_sl = round(entry - (0.7 * initial_risk), 2)

                pos["tp_expanded"] = True
                return AIPositionDecision(
                    action="EXPAND_TAKE_PROFIT",
                    new_stop_loss=locked_sl,
                    new_take_profit=new_tp,
                    reason=f"NỚI RỘNG TP GỒNG SÓNG DÀI HƠI 🚀 (Dời SL khóa lãi lên ${locked_sl:,.1f}, Nới TP lên ${new_tp:,.1f})",
                    status_display=f"🚀 NỚI TP GỒNG LÃI DÀI: ${new_tp:,.1f}"
                )

        # =========================================================================
        # 3. AI TAKE PROFIT (Chốt lời khi đạt target hoặc RSI kiệt sức cực đại)
        # =========================================================================
        if direction == 1:  # LONG
            # Extreme exhaustion at/above resistance
            if gain_usdt > 0 and (rsi >= 78.0 or current_price >= current_tp):
                return AIPositionDecision(
                    action="AI_TAKE_PROFIT",
                    reason=f"CHỐT LỜI ĐẠT MỤC TIÊU 🎯 (${current_price:,.2f} | RSI={rsi:.1f})",
                    status_display="🎯 Đã Chốt Lời Thành Công"
                )

        else:  # SHORT
            if gain_usdt > 0 and (rsi <= 22.0 or current_price <= current_tp):
                return AIPositionDecision(
                    action="AI_TAKE_PROFIT",
                    reason=f"CHỐT LỜI ĐẠT MỤC TIÊU 🎯 (${current_price:,.2f} | RSI={rsi:.1f})",
                    status_display="🎯 Đã Chốt Lời Thành Công"
                )

        # =========================================================================
        # 4. AI EARLY CUT LOSS (Cắt lỗ sớm khi cấu trúc sóng bị bẻ gãy ngược hướng)
        # =========================================================================
        if gain_usdt < 0:
            loss_pct = abs(gain_usdt) / pos["margin"] * 100.0 if pos["margin"] > 0 else 0.0

            # Cut loss early only if position has matured past noise window of its timeframe
            pos_tf = pos.get("timeframe", "15m")
            min_age = 60.0 if pos_tf in ("1m", "3m") else (120.0 if pos_tf == "5m" else 180.0)
            pos_age = time.time() - pos.get("open_timestamp", 0)
            if pos_age >= min_age and r_multiple <= -0.92 and loss_pct >= 1.2:
                if (direction == 1 and rsi < 32.0 and ai_verdict and ai_verdict.regime == "TRENDING_BEAR") or \
                   (direction == -1 and rsi > 68.0 and ai_verdict and ai_verdict.regime == "TRENDING_BULL"):
                    return AIPositionDecision(
                        action="AI_CUT_LOSS",
                        reason="AI CẮT LỖ SỚM BẢO TOÀN VỐN 🛑 (Cấu trúc đảo chiều ngược vị thế, cắt ở -0.85R)",
                        status_display="🛑 AI Cắt Lỗ Sớm Giảm Thiểu Rủi Ro"
                    )

            # Hit Stop Loss
            if (direction == 1 and current_price <= current_sl) or (direction == -1 and current_price >= current_sl):
                reason = "STOP_LOSS 🛑" if (current_sl < entry if direction == 1 else current_sl > entry) else "TRAILING_STOP 🛡️"
                return AIPositionDecision(
                    action="AI_CUT_LOSS",
                    reason=reason,
                    status_display="🛑 Chạm Cắt Lỗ"
                )

        # =========================================================================
        # 5. DYNAMIC TRAILING STOP THEO SÓNG (Nếu đã có lãi tốt)
        # =========================================================================
        if r_multiple >= 1.2:
            if direction == 1:
                trail_sl = round(current_price - (1.2 * atr), 2)
                if trail_sl > current_sl:
                    return AIPositionDecision(
                        action="UPDATE_TRAILING",
                        new_stop_loss=trail_sl,
                        reason=f"Nâng Trailing Stop lên ${trail_sl:,.2f} để khóa lợi nhuận",
                        status_display=f"🚀 AI Trailing Khóa Lãi: ${trail_sl:,.1f}"
                    )
            else:
                trail_sl = round(current_price + (1.2 * atr), 2)
                if trail_sl < current_sl:
                    return AIPositionDecision(
                        action="UPDATE_TRAILING",
                        new_stop_loss=trail_sl,
                        reason=f"Hạ Trailing Stop xuống ${trail_sl:,.2f} để khóa lợi nhuận",
                        status_display=f"🚀 AI Trailing Khóa Lãi: ${trail_sl:,.1f}"
                    )

        return AIPositionDecision(action="HOLD", status_display="AI Giám Sát Realtime 👁️")
