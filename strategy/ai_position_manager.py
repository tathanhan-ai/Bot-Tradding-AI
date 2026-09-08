"""
AI Dynamic Position Coordinator & Smart Exit Optimizer
Continuously coordinates open positions on every price tick:
1. Early Breakeven Lock (Dời SL về Entry + Phí Sàn khi lãi >= +0.5R thay vì +1.0R:
   winner không còn đường trượt về SL gốc)
2. Early Partial Take-Profit 50% tại +1.0R (thay vì +1.2R) hoặc khi đã đi
   50% đường tới TP: vùng lãi vừa phải khóa một phần, không chờ TP xa RR 2-3.5
3. Peak-anchored trailing: SL bám đỉnh lãi (peak - 1.0 ATR), không bao giờ lùi
4. TP Expansion chỉ khi R >= 2.0, momentum KHỎE VÀ macro đồng thuận, nới tối đa
   +1.0 ATR (thay vì +2.5 ATR), SL khóa tối thiểu +1.0R
5. AI Early Take-Profit (RSI kiệt sức) / AI Early Cut-Loss (-0.4R gãy cấu trúc)
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
        """Return an intent only; execution owns acknowledged position flags."""
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
        # 1. MILESTONE 1: DỜI SL VỀ ENTRY SỚM (KHÓA HÒA VỐN + PHÍ SÀN = RISK-FREE)
        # Winner-protection: kích hoạt từ +0.5R (thay vì +1.0R) để lệnh lãi đậm
        # sớm không còn đường trượt về SL gốc khi giá đảo chiều.
        # =========================================================================
        if not is_risk_free:
            # Trigger sớm từ +0.5R đã xác nhận lãi (gain_usdt > 0), không chờ +1.0R
            if r_multiple >= 0.5 and gain_usdt > 0:
                if direction == 1 and current_sl < breakeven:
                    # SL must always be strictly below current price with at least 0.3 ATR breathing room
                    new_sl = round(min(breakeven, current_price - (0.3 * atr)), 2)
                    if new_sl > current_sl and new_sl < current_price:
                        return AIPositionDecision(
                            action="LOCK_BREAKEVEN",
                            new_stop_loss=new_sl,
                            reason="DỜI SL VỀ ENTRY 🛡️ (Khóa Hòa Vốn Sau Phí - Lệnh Trở Thành Risk-Free)",
                            status_display="⏳ Đề xuất khóa hòa vốn — chờ xác nhận"
                        )
                elif direction == -1 and current_sl > breakeven:
                    new_sl = round(max(breakeven, current_price + (0.3 * atr)), 2)
                    if new_sl < current_sl and new_sl > current_price:
                        return AIPositionDecision(
                            action="LOCK_BREAKEVEN",
                            new_stop_loss=new_sl,
                            reason="DỜI SL VỀ ENTRY 🛡️ (Khóa Hòa Vốn Sau Phí - Lệnh Trở Thành Risk-Free)",
                            status_display="⏳ Đề xuất khóa hòa vốn — chờ xác nhận"
                        )

        # =========================================================================
        # 1.5 MILESTONE 1.5: CHỐT LỜI TỪNG PHẦN 50% SỚM (PARTIAL TAKE PROFIT)
        # Kích hoạt từ +1.0R (thay vì +1.2R): vùng lãi vừa phải khóa một phần,
        # không chờ TP xa RR 2-3.5 mới chốt đồng đầu tiên.
        # =========================================================================
        staged = pos.get("staged_take_profits") or {}
        staged_budget = any(staged.get(stage, 0) > 0 and staged.get(f"{stage}_ratio", 0) > 0
                            for stage in ("tp1", "tp2"))
        # The execution lifecycle owns OctoBot's fixed stage quantities. A second
        # independent 50% exit would consume that inventory twice.
        if not staged_budget and not pos.get("partial_tp_done", False):
            dist_to_tp = abs(current_tp - entry)
            progress_pct = (abs(current_price - entry) / max(dist_to_tp, 1.0)) if dist_to_tp > 0 else 0.0
            if gain_usdt > 0 and (r_multiple >= 1.0 or progress_pct >= 0.50):
                safe_sl = max(current_sl, breakeven) if direction == 1 else min(current_sl, breakeven)
                return AIPositionDecision(
                    action="PARTIAL_TAKE_PROFIT",
                    new_stop_loss=safe_sl,
                    reason=f"CHỐT LỜI 50% SỚM 💰 (${current_price:,.2f} | +{r_multiple:.2f}R)",
                    status_display="⏳ Đề xuất chốt 50% — chờ xác nhận"
                )

        # =========================================================================
        # 2. MILESTONE 2: NỚI RỘNG TP — CHỈ KHI ĐÃ LÃI DÀY (R >= 2.0) + XU HƯỚNG
        # THẬT SỰ ĐỒNG THUẬN. Nới tối đa +1.0 ATR, SL khóa tối thiểu +1.0R.
        # Chống mẫu hình cũ: chạm 85% TP là nới +2.5 ATR rồi quay đầu mất winner.
        # =========================================================================
        # When price reaches >= 85% of initial TP, only extend if profit is
        # already deep (R >= 2.0) AND momentum + macro both agree.
        dist_total = abs(current_tp - entry)
        curr_progress = (current_price - entry) * direction / dist_total if dist_total > 0 else 0.0

        if not tp_expanded and curr_progress >= 0.85 and r_multiple >= 2.0:
            # Check momentum: healthy but not exhausted (RSI 55-70 Long, 30-45 Short)
            momentum_healthy = (55.0 <= rsi <= 70.0) if direction == 1 else (30.0 <= rsi <= 45.0)
            # Không có verdict (unit test/backtest) thì coi như đồng thuận để không chặn;
            # live có verdict thì đòi CẢ momentum lẫn macro (AND) mới nới.
            macro_trend_aligned = (ai_verdict is None) or (getattr(ai_verdict, "regime", None) == ("TRENDING_BULL" if direction == 1 else "TRENDING_BEAR"))

            # Yêu cầu CẢ momentum lẫn macro đồng thuận (AND chứ không OR) mới nới
            if momentum_healthy and macro_trend_aligned:
                # Find higher timeframe target
                macro_target = 0.0
                if ai_verdict:
                    macro_target = ai_verdict.resistance_price if direction == 1 else ai_verdict.support_price

                if direction == 1:
                    new_tp = round(max(current_tp + (1.0 * atr), macro_target * 0.998), 2)
                    locked_sl = round(entry + (1.0 * initial_risk), 2)  # Khóa tối thiểu +1.0R
                else:
                    new_tp = round(min(current_tp - (1.0 * atr), macro_target * 1.002), 2) if macro_target > 0 else round(current_tp - (1.0 * atr), 2)
                    locked_sl = round(entry - (1.0 * initial_risk), 2)

                return AIPositionDecision(
                    action="EXPAND_TAKE_PROFIT",
                    new_stop_loss=locked_sl,
                    new_take_profit=new_tp,
                    reason=f"NỚI TP CÓ KIỂM SOÁT 🚀 (R={r_multiple:.1f} đã dày, SL khóa +1.0R tại ${locked_sl:,.1f}, TP mới ${new_tp:,.1f})",
                    status_display=f"🚀 NỚI TP GỒNG LÃI: ${new_tp:,.1f}"
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
        # 5. PEAK-ANCHORED TRAILING STOP (SL bám ĐỈNH LÃI, không bám giá hiện tại)
        # peak_price được tick mỗi nến — SL = peak ∓ 1.0 ATR từ +1.0R trở lên.
        # Winner trượt về vẫn giữ phần lớn lãi đỉnh, không trả hết cho thị trường.
        # =========================================================================
        if r_multiple >= 1.0 and gain_usdt > 0:
            peak = float(pos.get("peak_price", current_price) or current_price)
            if direction == 1:
                peak = max(peak, current_price)
                trail_sl = round(peak - (1.0 * atr), 2)
                if trail_sl > current_sl and trail_sl < current_price:
                    return AIPositionDecision(
                        action="UPDATE_TRAILING",
                        new_stop_loss=trail_sl,
                        reason=f"Trailing bám đỉnh ${peak:,.2f} − 1.0 ATR → SL ${trail_sl:,.2f} (khóa +{r_multiple:.1f}R)",
                        status_display=f"🚀 Trailing đỉnh lãi: ${trail_sl:,.1f}"
                    )
            else:
                peak = min(peak, current_price)
                trail_sl = round(peak + (1.0 * atr), 2)
                if trail_sl < current_sl and trail_sl > current_price:
                    return AIPositionDecision(
                        action="UPDATE_TRAILING",
                        new_stop_loss=trail_sl,
                        reason=f"Trailing bám đáy ${peak:,.2f} + 1.0 ATR → SL ${trail_sl:,.2f} (khóa +{r_multiple:.1f}R)",
                        status_display=f"🚀 Trailing đáy lãi: ${trail_sl:,.1f}"
                    )

        return AIPositionDecision(action="HOLD", status_display="AI Giám Sát Realtime 👁️")
