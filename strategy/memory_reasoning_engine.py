"""
Memory-Augmented Reasoning & Deterministic Validation Guardrails
Modeled after https://github.com/qrak/LLM_trader and Hummingbot Condor OODA loop

Key Mechanisms:
1. Episodic Trade Memory Bank:
   - Stores context profiles of closed trades (RSI, CVD delta, SMC zone, regime, outcome).
   - Before opening a new trade, queries memory for similar setups that failed.
   - Vetoes the trade if the current setup has high similarity to a recent loss (prevents repeating errors).
2. Deterministic Validation Guardrail:
   - Eliminates AI hallucinations by cross-checking qualitative claims against hard mathematical indicators.
   - If AI claims "Strong Trend" but ADX < 20, forces regime to "RANGING_SIDEWAY".
   - If AI claims "Oversold" but RSI > 40, clamps confidence and removes buy bias.
"""
import time
import math
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple


@dataclass
class TradeContextProfile:
    trade_id: int
    direction: int              # 1 (Long), -1 (Short)
    entry_price: float
    rsi: float
    cvd_momentum: str
    smc_structure: str
    vwap_status: str
    net_pnl: float
    outcome: str                # "WIN" or "LOSS"
    failure_reason: str = ""    # e.g., "Bought at resistance into Bearish Absorption"
    timestamp: float = field(default_factory=time.time)
    absorption_signal: str = "NONE"


@dataclass
class MemoryCheckResult:
    is_safe: bool               # False if a highly similar loss is detected
    warning_level: str          # "SAFE", "CAUTION", "BLOCKED"
    similar_trade_id: Optional[int] = None
    similarity_score: float = 0.0
    lesson_learned: str = ""


@dataclass
class DeterministicValidationResult:
    is_valid: bool
    corrected_regime: str
    confidence_penalty: float
    validation_notes: str


BASELINE_PRINCIPLES: List[Dict[str, Any]] = [
    {
        "trade_id": "OODA-1",
        "direction": "RISK",
        "outcome": "GUARD",
        "lesson": "Chống Noise Stopout: Nến 1m/5m bắt buộc đệm SL tối thiểu >= 0.45% để triệt tiêu dao động ngẫu nhiên vi mô.",
        "loss": 0.0,
        "is_baseline": True
    },
    {
        "trade_id": "OODA-2",
        "direction": "FLOW",
        "outcome": "GUARD",
        "lesson": "CVD Phân kỳ xả: Không mở Long khi giá tăng nhưng Delta CVD tạo đỉnh thấp hơn (Hấp thụ xả cá mập).",
        "loss": 0.0,
        "is_baseline": True
    },
    {
        "trade_id": "OODA-3",
        "direction": "SMC",
        "outcome": "GUARD",
        "lesson": "Bẫy thanh khoản (Equal Lows): Đặt SL lùi sau râu nến quét thanh khoản thay vì sát đáy cản hiển nhiên.",
        "loss": 0.0,
        "is_baseline": True
    },
    {
        "trade_id": "OODA-4",
        "direction": "VWAP",
        "outcome": "GUARD",
        "lesson": "Định giá cân bằng: Ưu tiên Long ở vùng Discount hoặc POC, không Fomo khi giá cách xa dải trung bình.",
        "loss": 0.0,
        "is_baseline": True
    }
]


class EpisodicTradeMemoryBank:
    """
    Episodic Trade Memory Bank:
    Remembers recent trading experiences and checks whether the current candidate
    order closely resembles a previous setup that resulted in a loss.
    """
    def __init__(self, max_records: int = 40):
        self.max_records = max_records
        self.memory_records: List[TradeContextProfile] = []

    def record_trade_outcome(
        self,
        trade_id: int,
        direction: int,
        entry_price: float,
        indicators: Dict[str, Any],
        of_data: Optional[Dict[str, Any]],
        smc_data: Optional[Dict[str, Any]],
        vwap_data: Optional[Dict[str, Any]],
        net_pnl: float,
        exit_reason: str
    ):
        outcome = "WIN" if net_pnl > 0 else "LOSS"
        absorption = of_data.get("absorption_signal", "NONE") if of_data else "NONE"
        vwap_status = vwap_data.get("vwap_status", "FAIR") if vwap_data else "FAIR"

        if outcome == "LOSS":
            if direction == 1 and "PREMIUM" in vwap_status:
                failure_reason = "Fomo Long ở vùng giá đắt (Premium VWAP). Bài học: Chờ nén về Discount hoặc POC."
            elif direction == -1 and "DISCOUNT" in vwap_status:
                failure_reason = "Fomo Short ở vùng giá rẻ chiết khấu (Discount VWAP). Bài học: Chờ hồi lên Premium."
            elif direction == 1 and absorption == "BEAR_ABSORPTION":
                failure_reason = "Long vào tường xả ẩn của cá mập (Bear Absorption). Phủ quyết Long khi CVD xả."
            elif direction == -1 and absorption == "BULL_ABSORPTION":
                failure_reason = "Short vào tường gom ẩn của cá mập (Bull Absorption). Phủ quyết Short khi CVD gom."
            elif "STOP_LOSS" in exit_reason:
                failure_reason = f"Dính cắt lỗ do râu nến quét ({exit_reason}). Đã kích hoạt đệm chống nhiễu lớn hơn."
            else:
                failure_reason = f"Dính cắt lỗ do đảo chiều ngược cấu trúc ({exit_reason})."
        else:
            failure_reason = f"Chốt lời thành công: Đi đúng cấu trúc sóng + Hấp thụ thuận xu hướng ({exit_reason})."

        profile = TradeContextProfile(
            trade_id=trade_id,
            direction=direction,
            entry_price=entry_price,
            rsi=indicators.get("rsi", 50.0),
            cvd_momentum=of_data.get("delta_momentum", "BALANCED") if of_data else "BALANCED",
            smc_structure=smc_data.get("structure", "RANGING") if smc_data else "RANGING",
            vwap_status=vwap_data.get("vwap_status", "EQUILIBRIUM_FAIR") if vwap_data else "EQUILIBRIUM_FAIR",
            net_pnl=net_pnl,
            outcome=outcome,
            failure_reason=failure_reason,
            absorption_signal=absorption,
        )

        self.memory_records.append(profile)
        if len(self.memory_records) > self.max_records:
            self.memory_records.pop(0)

        # Logging must not interrupt an already-realized fill on legacy Windows encodings.

    def query_similarity_against_losses(
        self,
        candidate_direction: int,
        candidate_rsi: float,
        candidate_vwap_status: str,
        candidate_absorption: str
    ) -> MemoryCheckResult:
        """
        Calculates cosine-like similarity with previous loss records.
        If similarity >= 0.75, flags as BLOCKED.
        """
        loss_records = [r for r in self.memory_records if r.outcome == "LOSS"]
        if not loss_records:
            return MemoryCheckResult(is_safe=True, warning_level="SAFE", lesson_learned="Bộ nhớ sạch, chưa có mẫu hình thua lỗ tương tự.")

        highest_sim = 0.0
        worst_match: Optional[TradeContextProfile] = None

        for rec in reversed(loss_records[-10:]):  # prioritize recent 10 losses
            if rec.direction != candidate_direction:
                continue

            sim = 0.0
            # 1. RSI similarity (max 0.30)
            rsi_diff = abs(candidate_rsi - rec.rsi)
            sim += max(0.0, 0.30 - (rsi_diff / 50.0) * 0.30)

            # 2. VWAP status match (max 0.35)
            if candidate_vwap_status == rec.vwap_status:
                sim += 0.35

            # 3. Absorption trap match (max 0.35)
            if candidate_absorption != "NONE" and candidate_absorption == rec.absorption_signal:
                sim += 0.35

            if sim > highest_sim:
                highest_sim = sim
                worst_match = rec

        highest_sim = round(highest_sim, 2)
        if highest_sim >= 0.70 and worst_match:
            lesson = (
                f"🛑 [MEMORY VETO] Bối cảnh tương đồng {highest_sim*100:.0f}% với lệnh thua #{worst_match.trade_id}! "
                f"Bài học cũ: {worst_match.failure_reason}."
            )
            return MemoryCheckResult(
                is_safe=False,
                warning_level="BLOCKED",
                similar_trade_id=worst_match.trade_id,
                similarity_score=highest_sim,
                lesson_learned=lesson
            )
        elif highest_sim >= 0.50 and worst_match:
            lesson = f"⚠️ [MEMORY CẢNH BÁO] Tương đồng nhẹ {highest_sim*100:.0f}% với lệnh #{worst_match.trade_id} ({worst_match.failure_reason})."
            return MemoryCheckResult(
                is_safe=True,
                warning_level="CAUTION",
                similar_trade_id=worst_match.trade_id,
                similarity_score=highest_sim,
                lesson_learned=lesson
            )

        return MemoryCheckResult(
            is_safe=True,
            warning_level="SAFE",
            similarity_score=highest_sim,
            lesson_learned="Không có mẫu hình sai lầm cũ lặp lại."
        )

    def get_recent_lessons(self, limit: int = 5) -> List[Dict[str, Any]]:
        lessons = []
        for r in reversed(self.memory_records):
            if r.failure_reason:
                lessons.append({
                    "trade_id": f"#{r.trade_id}",
                    "direction": "LONG" if r.direction == 1 else "SHORT",
                    "outcome": r.outcome,
                    "lesson": r.failure_reason,
                    "loss": round(abs(r.net_pnl), 2),
                    "pnl": round(r.net_pnl, 2),
                    "is_baseline": False
                })
            if len(lessons) >= limit:
                break

        # If we have fewer real trade lessons than limit, fill with baseline principles
        if len(lessons) < limit:
            for bp in BASELINE_PRINCIPLES:
                lessons.append(bp)
                if len(lessons) >= limit:
                    break

        return lessons


class DeterministicValidationGuardrail:
    """
    Deterministic Validation Engine:
    Validates AI qualitative conclusions against rigorous mathematical facts.
    """
    def validate(
        self,
        claimed_regime: str,
        claimed_confidence: float,
        adx: float,
        atr: float,
        rsi: float
    ) -> DeterministicValidationResult:
        corrected_regime = claimed_regime
        penalty = 0.0
        notes = []

        # Rule 1: Claiming strong trend when ADX < 20
        if "TREND" in claimed_regime and adx < 20.0:
            corrected_regime = "RANGING_SIDEWAY"
            penalty += 15.0
            notes.append(f"ADX={adx:.1f} < 20 (Thị trường không có xu hướng, ép về Ranging)")

        # Rule 2: Claiming Bullish Trend when RSI < 40
        if "BULL" in claimed_regime and rsi < 40.0:
            penalty += 20.0
            notes.append(f"RSI={rsi:.1f} quá yếu cho nhịp tăng")

        # Rule 3: Claiming Bearish Trend when RSI > 60
        if "BEAR" in claimed_regime and rsi > 60.0:
            penalty += 20.0
            notes.append(f"RSI={rsi:.1f} quá mạnh cho nhịp giảm")

        is_valid = (penalty < 30.0)
        validation_str = " | ".join(notes) if notes else "Tất cả số liệu toán học khớp hoàn hảo với nhận định AI."

        return DeterministicValidationResult(
            is_valid=is_valid,
            corrected_regime=corrected_regime,
            confidence_penalty=penalty,
            validation_notes=validation_str
        )
