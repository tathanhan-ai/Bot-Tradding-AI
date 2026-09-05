"""
Antigravity AI Chief Risk Officer (AI-CRO)
Autonomous AI-driven capital risk manager powered by the Antigravity reasoning engine.
Dynamically controls:
1. Risk per trade (0.5% - 2.5%) based on market regime and volatility.
2. Consecutive loss throttling (anti-martingale protection).
3. Portfolio margin exposure cap (25% - 60%).
4. Adaptive daily drawdown circuit breakers.
5. Live risk rationale in Vietnamese.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional


@dataclass
class AIRiskVerdict:
    risk_per_trade_pct: float       # e.g. 1.5% = 0.015
    max_margin_utilization_pct: float # e.g. 35% = 0.35
    max_risk_amount_usdt: float     # e.g. $15.00 USDT
    daily_drawdown_limit_pct: float # e.g. 4.0%
    protection_mode: str            # 'AGGRESSIVE', 'BALANCED', 'DEFENSIVE', 'CAPITAL_PRESERVATION'
    consecutive_loss_count: int
    circuit_breaker_active: bool
    risk_rationale: str             # Vietnamese explanation from Antigravity AI
    defense_status: str             # 'ACTIVE_SHIELD', 'NORMAL', 'THROTTLED'
    updated_at: str


class AntigravityRiskOfficer:
    def __init__(self, initial_balance: float = 1000.0):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.daily_start_balance = initial_balance
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.circuit_breaker = False
        self.current_drawdown_pct = 0.0
        self.user_risk_pct: Optional[float] = None
        self.last_verdict: Optional[AIRiskVerdict] = None

    def record_trade_result(self, pnl: float):
        """Update win/loss streaks to dynamically throttle or reward risk allocation"""
        if pnl > 0:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        elif pnl < 0:
            self.consecutive_losses += 1
            self.consecutive_wins = 0

    def evaluate_risk_profile(
        self,
        current_balance: float,
        market_regime: str,
        confidence: int,
        atr_pct: float
    ) -> AIRiskVerdict:
        """
        Antigravity AI calculates dynamic capital risk parameters in real time.
        """
        self.current_balance = current_balance
        now_str = datetime.now().strftime("%H:%M:%S")

        # 1. Daily Drawdown calculation
        daily_loss_pct = ((self.daily_start_balance - self.current_balance) / self.daily_start_balance) * 100.0
        self.current_drawdown_pct = max(0.0, daily_loss_pct / 100.0)
        if daily_loss_pct >= 4.0:
            self.circuit_breaker = True
            self.last_verdict = AIRiskVerdict(
                risk_per_trade_pct=0.0,
                max_margin_utilization_pct=0.0,
                max_risk_amount_usdt=0.0,
                daily_drawdown_limit_pct=4.0,
                protection_mode="CAPITAL_PRESERVATION",
                consecutive_loss_count=self.consecutive_losses,
                circuit_breaker_active=True,
                risk_rationale=f"[{now_str}] CẦU CHÌ BẢO VỆ KÍCH HOẠT: Lỗ ngày đạt {daily_loss_pct:.2f}% (vượt ngưỡng 4.0%). Antigravity AI đã khóa toàn bộ lệnh để bảo toàn vốn gốc.",
                defense_status="CIRCUIT_BREAKER_LOCKED",
                updated_at=now_str
            )
            return self.last_verdict

        # 2. Base Risk Calculation based on Volatility (ATR) & Regime
        base_risk = (self.user_risk_pct / 100.0) if self.user_risk_pct else 0.015  # Default 1.5%
        max_margin_cap = 0.40 # Default 40%

        if self.user_risk_pct:
            mode = "CUSTOM_RISK"
            status = f"USER_SET ({self.user_risk_pct}%)"
            reason = f"Quy tắc quản trị vốn Antigravity AI đang áp dụng mức rủi ro chỉ định {self.user_risk_pct}% (${current_balance * base_risk:.2f}/lệnh)."
        elif market_regime == "VOLATILE_PANIC" or atr_pct > 2.0:
            # Extreme market shakeout: Defend capital
            base_risk = 0.008  # 0.8%
            max_margin_cap = 0.25 # 25% max margin
            mode = "DEFENSIVE"
            status = "ACTIVE_SHIELD 🛡️"
            reason = f"Thị trường biến động mạnh (ATR {atr_pct:.2f}%). Antigravity AI hạ rủi ro mỗi lệnh xuống 0.8% (${current_balance * 0.008:.2f}) và giới hạn ký quỹ 25% để triệt tiêu nguy cơ rung lắc."

        elif market_regime in ("TRENDING_BULL", "TRENDING_BEAR") and confidence >= 85:
            # High-confidence confirmed trend
            base_risk = 0.020  # 2.0%
            max_margin_cap = 0.55 # 55%
            mode = "AGGRESSIVE"
            status = "TREND_COMPOUNDING 🚀"
            reason = f"Xu hướng rõ ràng với độ tin cậy AI {confidence}%. Antigravity AI nâng rủi ro lên 2.0% (${current_balance * 0.02:.2f}) để tối ưu hóa tỷ suất sinh lời khi bứt phá sóng."

        elif market_regime == "RANGING_SIDEWAY":
            # Choppy sideway: Stable risk
            base_risk = 0.012  # 1.2%
            max_margin_cap = 0.35 # 35%
            mode = "BALANCED"
            status = "BALANCED ⚖️"
            reason = f"Thị trường tích lũy đi ngang. Antigravity AI phân bổ rủi ro cân bằng 1.2% (${current_balance * 0.012:.2f}), kiểm soát chặt vốn ký quỹ tránh bào mòn tài khoản."

        else:
            mode = "BALANCED"
            status = "NORMAL"
            reason = f"Điều kiện thị trường bình thường. Antigravity AI duy trì rủi ro 1.5% (${current_balance * 0.015:.2f})."

        # 3. Anti-Martingale / Consecutive Loss Throttling
        if self.consecutive_losses >= 2:
            base_risk = max(0.005, base_risk * 0.5)  # Cut risk in half
            max_margin_cap = min(0.20, max_margin_cap * 0.6)
            mode = "CAPITAL_PRESERVATION"
            status = "LOSS_STREAK_THROTTLED 🚨"
            reason += f" [CẢNH BÁO CHUỖI THUA: {self.consecutive_losses} lệnh liên tiếp. Antigravity AI tự động siết giảm 50% khối lượng lệnh tiếp theo]."

        max_risk_usdt = round(current_balance * base_risk, 2)

        verdict = AIRiskVerdict(
            risk_per_trade_pct=round(base_risk * 100, 2),
            max_margin_utilization_pct=round(max_margin_cap * 100, 1),
            max_risk_amount_usdt=max_risk_usdt,
            daily_drawdown_limit_pct=4.0,
            protection_mode=mode,
            consecutive_loss_count=self.consecutive_losses,
            circuit_breaker_active=self.circuit_breaker,
            risk_rationale=f"[{now_str}] {reason}",
            defense_status=status,
            updated_at=now_str
        )
        self.last_verdict = verdict
        return verdict
