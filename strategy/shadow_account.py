# -*- coding: utf-8 -*-
"""
HKUDS Vibe-Trading: Shadow Account & Behavioral Bias Scanner
(Tài Khoản Bóng Ma & Soi Lỗi Tâm Lý Trader)
Phân tích nhật ký lệnh từ SQLite/Memory Bank để bóc tách 3 tật xấu tâm lý kinh điển:
1. Chốt non (Premature Exit / Disposition Effect): Đóng lệnh sớm khi chưa chạm TP cấu trúc
2. Gồng lỗ (Loss Aversion): Dời SL hoặc gồng lỗ quá thời gian quy định
3. Vào lệnh trả thù (Revenge Trading): Mở lệnh gấp gáp ngay sau lệnh thua
Tính toán Điểm Kỷ Luật (Discipline Score: 0-100%) và kịch bản đối chiếu What-If.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional
import math


@dataclass
class BehavioralBiasReport:
    discipline_score: int = 0               # Interpret only when analysis_available=True.
    discipline_grade: str = "UNAVAILABLE"
    
    # Biases count
    premature_exits_count: int = 0          # Số lần chốt non
    loss_aversion_count: int = 0            # Số lần gồng lỗ vượt quá kế hoạch
    revenge_trades_count: int = 0           # Số lần vào lệnh trả thù
    
    # What-If Comparative Analysis
    actual_pnl_usdt: float = 0.0            # PnL thực tế
    shadow_systematic_pnl_usdt: float = 0.0 # PnL nếu tuân thủ 100% thuật toán
    missed_alpha_usdt: float = 0.0          # Lợi nhuận bị bỏ lỡ do can thiệp tay
    
    behavioral_diagnostics: str = ""        # Lời khuyên tâm lý & điều chỉnh hành vi
    analysis_available: bool = False
    counterfactual_available: bool = False
    counterfactual_reason: str = "Thiếu replay đường giá và kế hoạch SL/TP gốc; chưa đo được missed alpha."
    sample_size: int = 0


class ShadowAccountAnalyzer:
    def __init__(self):
        self.latest_report = BehavioralBiasReport()

    @staticmethod
    def _timestamp(trade: Dict[str, Any], *keys: str) -> Optional[float]:
        for key in keys:
            value = trade.get(key)
            if value is None or value == "":
                continue
            try:
                ts = float(value)
                ts = ts / 1000.0 if ts > 1e11 else ts
            except (TypeError, ValueError):
                value = str(value)
                if len(value) >= 5 and value[2] == "-":
                    reference = str(trade.get("created_at") or trade.get("exit_time") or "")
                    if len(reference) < 10 or reference[4] != "-":
                        continue
                    value = reference[:4] + "-" + value
                try:
                    ts = datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
                except (ValueError, OverflowError, OSError):
                    continue
            if math.isfinite(ts) and ts > 0:
                return ts
        return None

    def analyze_trade_history(
        self,
        trades: List[Dict[str, Any]],
        initial_balance: float = 5000.0
    ) -> BehavioralBiasReport:
        if not trades:
            self.latest_report = BehavioralBiasReport(behavioral_diagnostics="Chưa có lệnh đóng để đánh giá.")
            return self.latest_report

        try:
            premature_exits = 0
            loss_aversions = 0
            revenge_trades = 0
            actual_pnl = 0.0
            last_loss_time = 0.0

            for t in sorted(trades, key=lambda trade: self._timestamp(trade, "closed_at_ts", "timestamp", "exit_time", "created_at") or 0.0):
                pnl = float(t.get("pnl", 0.0))
                if not math.isfinite(pnl):
                    raise ValueError("non-finite realized PnL")
                actual_pnl += pnl
                reason = str(t.get("reason", "")).lower()
                close_time = self._timestamp(t, "closed_at_ts", "timestamp", "closed_at", "exit_time", "created_at")
                open_time = self._timestamp(t, "open_time", "open_timestamp", "opened_at", "entry_time")

                # 1. Chốt non (Premature Exit)
                # Lãi nhỏ (0 < pnl < 15 USD hoặc pnl_pct < 0.6%) nhưng đóng thủ công (không phải TP cản)
                if 0.0 < pnl < 15.0 and "thủ công" in reason:
                    premature_exits += 1

                # 2. Gồng lỗ (Loss Aversion)
                if pnl < -45.0:
                    loss_aversions += 1

                # 3. Vào lệnh trả thù (Revenge Trading)
                # Mở lệnh trong vòng 90 giây ngay sau 1 lệnh bị lỗ
                if last_loss_time and open_time is not None and 0 <= (open_time - last_loss_time) < 90.0:
                    revenge_trades += 1

                if pnl < 0:
                    last_loss_time = close_time or 0.0

            # Compute Discipline Score: Base 100 minus penalties
            penalty = (premature_exits * 6) + (loss_aversions * 10) + (revenge_trades * 14)
            disc_score = max(25, min(100, 100 - penalty))

            if disc_score >= 85:
                grade = "A"
                advice = "Kỷ luật thép! Bạn tuân thủ chặt chẽ kế hoạch StopLoss & TakeProfit của thuật toán."
            elif disc_score >= 70:
                grade = "B"
                advice = f"Khá tốt. Phát hiện {premature_exits} lần chốt non, hãy để lệnh chạy hết target TP."
            elif disc_score >= 50:
                grade = "C"
                advice = f"Cảnh báo: Có {revenge_trades} lệnh vào trả thù sau khi thua. Cần kích hoạt cooldown."
            else:
                grade = "D"
                advice = "Tâm lý FOMO/Revenge chi phối! Đề xuất bật chế độ khóa tay (Pure Autonomous Mode)."

            self.latest_report = BehavioralBiasReport(
                discipline_score=disc_score,
                discipline_grade=grade,
                premature_exits_count=premature_exits,
                loss_aversion_count=loss_aversions,
                revenge_trades_count=revenge_trades,
                actual_pnl_usdt=round(actual_pnl, 2),
                # No counterfactual PnL can be inferred from realized PnL alone.
                shadow_systematic_pnl_usdt=0.0,
                missed_alpha_usdt=0.0,
                behavioral_diagnostics=f"Chỉ báo hành vi heuristic: {advice}",
                analysis_available=True,
                sample_size=len(trades),
            )
            return self.latest_report
        except (TypeError, ValueError, OverflowError):
            self.latest_report = BehavioralBiasReport(behavioral_diagnostics="Dữ liệu lịch sử không hợp lệ; chưa thể đánh giá.")
            return self.latest_report

    def get_report(self) -> BehavioralBiasReport:
        return self.latest_report

    def get_latest_analysis(self) -> BehavioralBiasReport:
        return self.latest_report
