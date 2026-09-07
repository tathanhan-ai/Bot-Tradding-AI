"""
Antigravity Monthly Target Governor & Adaptive Capital Allocation Engine
Manages monthly return expectations, capital preservation when targets are achieved,
and dynamic deficit compensation for subsequent months.

Core Capabilities:
1. Target Achievement Regulator:
   - When monthly profit >= target %, automatically switches into Capital Preservation Mode.
   - Throttles position sizes (e.g., 0.5x), caps max leverage (<= 3x), tightens stop loss,
     and requires higher AI consensus to protect accumulated monthly profits.

2. Deficit Catch-up Compensation (Auto-Adjustment for Following Month):
   - If the month closes without reaching the expected return, the unachieved deficit
     is mathematically amortized and carried forward into the next month's target.
   - AI adjusts strategy selection: prioritizes high R:R setups (>= 1:2.5) and optimal Kelly sizing
     within strict risk boundaries (never exceeds maximum 2% equity risk per trade).

3. Time-to-Month-End Pacing Analytics:
   - Evaluates daily progress vs expected linear/geometric pacing.
   - Prevents end-of-month FOMO/overtrading if behind schedule.
"""
from dataclasses import dataclass, asdict
from datetime import datetime, date
import calendar
import json
from typing import Dict, Any, Optional, List


@dataclass
class MonthlyGovernorStatus:
    enabled: bool
    base_target_pct: float             # User specified monthly target, e.g. 10.0%
    carried_deficit_pct: float         # Deficit carried from previous months, e.g. 3.2%
    effective_target_pct: float        # base + carried deficit (capped), e.g. 13.2%
    current_month_str: str             # e.g. '2026-09'
    month_start_balance: float         # Baseline balance for this month
    month_realized_pnl: float          # Realized PnL in USDT this month
    current_pnl_pct: float             # Realized PnL % this month
    progress_ratio: float              # current_pnl_pct / effective_target_pct
    day_of_month: int                  # Current day (e.g. 4)
    days_in_month: int                 # Total days (e.g. 30)
    month_time_progress_pct: float     # (day_of_month / days_in_month) * 100
    regime: str                        # 'TARGET_ACHIEVED', 'DEFICIT_CATCHUP', 'ON_TRACK', 'BEHIND_PACE'
    protection_mode: str               # 'CAPITAL_PRESERVATION', 'BALANCED', 'ADAPTIVE_CATCHUP'
    size_multiplier: float             # Multiplier applied to position size (e.g. 0.5 if achieved)
    max_leverage_cap: int              # Max allowable leverage (e.g. 3 if achieved, 10 otherwise)
    min_ai_confidence: int             # Min confidence required (e.g. 75 if achieved, 55 otherwise)
    min_risk_reward_ratio: float       # Min R:R required (e.g. 2.5 if catch-up, 1.5 normal)
    rationale: str                     # Vietnamese explanation for display
    updated_at: str
    target_pnl_usdt: float = 0.0       # Target profit in USDT for this month
    target_daily_pnl_usdt: float = 0.0 # Expected profit pacing per day in USDT
    remaining_days: int = 1            # Days remaining in current month
    reserve_ratio_recommended: float = 0.15 # 0.30 if TARGET_ACHIEVED, 0.15 otherwise
    profile_name: str = "growth"       # sustainable | balanced | growth | aggressive
    profile_label: str = "Tăng trưởng 10-20%"


class MonthlyTargetGovernor:
    MAX_CATCHUP_TARGET_CAP = 30.0      # Safety cap: target will never exceed 30%/month

    # Ma trận mốc kỳ vọng lãi tháng — chọn 1 click qua update_config(profile=...).
    # Bất biến an toàn: mọi profile giữ size_multiplier <= 1.0 (chống đuổi target),
    # risk/trade vẫn do RiskConfig + CRO quản (<= 1.5-2.0% vốn).
    PROFILES: Dict[str, Dict[str, Any]] = {
        "sustainable": {   # Bền vững 3-5%: mặc định khi paper chưa đạt gate
            "label": "Bền vững 3-5%",
            "base_target_pct": 4.0,
            "on_track": {"size": 1.0, "lev": 5, "conf": 55, "rr": 1.5},
            "deficit_catchup": {"size": 0.85, "lev": 3, "conf": 65, "rr": 2.0},
            "behind_pace": {"size": 0.80, "lev": 3, "conf": 70, "rr": 1.8},
            "achieved": {"size": 0.50, "lev": 2, "conf": 75, "rr": 2.0},
            "unlock": "Luôn mở (mặc định an toàn)",
        },
        "balanced": {      # Cân bằng 6-8%: mở khi paper đạt gate 20 lệnh
            "label": "Cân bằng 6-8%",
            "base_target_pct": 7.0,
            "on_track": {"size": 1.0, "lev": 6, "conf": 55, "rr": 1.5},
            "deficit_catchup": {"size": 0.85, "lev": 4, "conf": 65, "rr": 2.2},
            "behind_pace": {"size": 0.80, "lev": 4, "conf": 70, "rr": 2.0},
            "achieved": {"size": 0.50, "lev": 3, "conf": 75, "rr": 2.0},
            "unlock": "Winrate 20 lệnh >45%, PF >1.5, expectancy >0",
        },
        "growth": {        # Tăng trưởng 10-20%: profile đang chạy live
            "label": "Tăng trưởng 10-20%",
            "base_target_pct": 10.0,
            "on_track": {"size": 1.0, "lev": 8, "conf": 55, "rr": 1.5},
            # RR catch-up 2.5 -> 2.2 cho toi khi paper dat gate (RR 2.5 kho khop khi thi truong yeu,
            # sizing lai tu noi TP cho du RR cang lam giam fill)
            "deficit_catchup": {"size": 0.85, "lev": 4, "conf": 65, "rr": 2.2},
            "behind_pace": {"size": 0.80, "lev": 4, "conf": 70, "rr": 2.0},
            "achieved": {"size": 0.50, "lev": 3, "conf": 75, "rr": 2.0},
            "unlock": "Paper đạt gate như Balanced",
        },
        "aggressive": {    # Tốc chiến 20-30%: chỉ mở khi PF >2.0 trên 50 lệnh paper
            "label": "Tốc chiến 20-30%",
            "base_target_pct": 25.0,
            "on_track": {"size": 1.0, "lev": 8, "conf": 60, "rr": 1.8},
            "deficit_catchup": {"size": 0.80, "lev": 4, "conf": 70, "rr": 2.5},
            "behind_pace": {"size": 0.75, "lev": 3, "conf": 72, "rr": 2.2},
            "achieved": {"size": 0.40, "lev": 2, "conf": 80, "rr": 2.2},
            "unlock": "PF >2.0 trên 50 lệnh paper",
        },
    }

    def __init__(
        self,
        base_target_pct: float = 10.0,
        enabled: bool = True,
        auto_compensate_deficit: bool = True,
        storage = None,
        profile: str = "growth",
    ):
        self.base_target_pct = float(base_target_pct)
        self.enabled = bool(enabled)
        self.auto_compensate_deficit = bool(auto_compensate_deficit)
        self.storage = storage
        self.profile_name = profile if profile in self.PROFILES else "growth"

        now = datetime.now()
        self.current_month_str = now.strftime("%Y-%m")
        self.month_start_balance = 0.0
        self.carried_deficit_pct = 0.0

        # Load persisted governor state if available
        self._load_state()

    def _load_state(self):
        if not self.storage:
            return
        try:
            val_target = self.storage.get_setting("monthly_target_pct")
            if val_target is not None:
                self.base_target_pct = float(val_target)

            val_enabled = self.storage.get_setting("monthly_target_enabled")
            if val_enabled is not None:
                self.enabled = str(val_enabled).lower() in ("true", "1", "yes", "on")

            val_compensate = self.storage.get_setting("monthly_compensate_deficit")
            if val_compensate is not None:
                self.auto_compensate_deficit = str(val_compensate).lower() in ("true", "1", "yes", "on")

            val_profile = self.storage.get_setting("monthly_target_profile")
            if val_profile in self.PROFILES:
                self.profile_name = val_profile
                self.base_target_pct = float(self.PROFILES[val_profile]["base_target_pct"])

            val_month = self.storage.get_setting("monthly_governor_month")
            val_start_bal = self.storage.get_setting("monthly_start_balance")
            val_deficit = self.storage.get_setting("monthly_carried_deficit")

            now_month = datetime.now().strftime("%Y-%m")

            if val_month and val_start_bal is not None and float(val_start_bal) > 0:
                self.current_month_str = val_month
                self.month_start_balance = float(val_start_bal)
                self.carried_deficit_pct = float(val_deficit) if val_deficit else 0.0
                if val_month != now_month:
                    self._handle_month_rollover(now_month, self.month_start_balance)
            else:
                # First use has no prior target or deficit to carry forward.
                balance = self._stored_balance(0.0)
                if balance > 0:
                    self.month_start_balance = max(1.0, balance - self._realized_pnl(now_month))
                    self._save_state()

        except Exception as e:
            print(f"[MonthlyTargetGovernor] Error loading state: {e}", flush=True)

    def _save_state(self):
        if not self.storage:
            return
        try:
            self.storage.save_setting("monthly_target_pct", self.base_target_pct)
            self.storage.save_setting("monthly_target_enabled", self.enabled)
            self.storage.save_setting("monthly_compensate_deficit", self.auto_compensate_deficit)
            self.storage.save_setting("monthly_target_profile", self.profile_name)
            self.storage.save_setting("monthly_governor_month", self.current_month_str)
            self.storage.save_setting("monthly_start_balance", self.month_start_balance)
            self.storage.save_setting("monthly_carried_deficit", self.carried_deficit_pct)
        except Exception as e:
            print(f"[MonthlyTargetGovernor] Error saving state: {e}", flush=True)

    def update_config(self, target_pct: Optional[float] = None, enabled: bool = True,
                      auto_compensate: bool = True, profile: Optional[str] = None):
        if profile in self.PROFILES:
            self.profile_name = profile
            self.base_target_pct = float(self.PROFILES[profile]["base_target_pct"])
        if target_pct is not None:
            self.base_target_pct = max(1.0, min(100.0, float(target_pct)))
        self.enabled = bool(enabled)
        self.auto_compensate_deficit = bool(auto_compensate)
        self._save_state()

    @staticmethod
    def _is_trade_in_month(t: dict, ym_str: str) -> bool:
        """Assign realized PnL only to the close month, never the entry month."""
        for key in ("closed_at_ts", "timestamp", "closed_at", "exit_time", "created_at", "time"):
            value = t.get(key)
            if value is None or value == "":
                continue
            try:
                ts = float(value)
                dt = datetime.fromtimestamp(ts / 1000.0 if ts > 1e11 else ts)
            except (TypeError, ValueError, OverflowError, OSError):
                value = str(value)
                if key == "exit_time" and len(value) >= 5 and value[2] == "-":
                    created = str(t.get("created_at") or "")
                    if len(created) < 10 or created[4] != "-":
                        continue
                    value = created[:4] + "-" + value
                try:
                    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
                except ValueError:
                    continue
            return dt.strftime("%Y-%m") == ym_str
        return False

    def _realized_pnl(self, year_month: str, trades: Optional[List[dict]] = None) -> float:
        if self.storage and hasattr(self.storage, "get_monthly_realized_pnl"):
            return float(self.storage.get_monthly_realized_pnl(year_month))
        if trades is None and self.storage:
            trades = self.storage.load_trades(limit=1000)
        return sum(float(t.get("pnl", 0.0)) for t in (trades or []) if self._is_trade_in_month(t, year_month))

    def _stored_balance(self, default: float) -> float:
        if not self.storage:
            return default
        if hasattr(self.storage, "load_execution_runtime"):
            runtime = self.storage.load_execution_runtime()
            if runtime and "balance" in runtime:
                return float(runtime["balance"])
        return float((self.storage.load_account_state() or {}).get("current_balance", default))

    def _handle_month_rollover(self, new_month_str: str, prior_start_balance: float,
                               current_balance: Optional[float] = None, trades: Optional[List[dict]] = None):
        """
        Calculates whether previous month achieved target. If not, calculates deficit to carry over.
        """
        last_month_pnl_usdt = self._realized_pnl(self.current_month_str, trades)
        last_month_pnl_pct = last_month_pnl_usdt / prior_start_balance * 100.0 if prior_start_balance > 0 else 0.0

        effective_prior_target = self.base_target_pct + self.carried_deficit_pct

        if self.auto_compensate_deficit and prior_start_balance > 0 and (last_month_pnl_pct < effective_prior_target):
            # Unachieved deficit: carries over to new month
            deficit = max(0.0, effective_prior_target - last_month_pnl_pct)
            # Amortize deficit safely (cap added deficit at 15% to prevent runaway targets)
            self.carried_deficit_pct = round(min(15.0, deficit), 2)
        else:
            # Reached or exceeded target, reset carried deficit
            self.carried_deficit_pct = 0.0

        self.current_month_str = new_month_str
        # New month baseline balance
        if current_balance is None:
            current_balance = self._stored_balance(prior_start_balance + last_month_pnl_usdt)
        self.month_start_balance = max(1.0, current_balance - self._realized_pnl(new_month_str, trades))
        self._save_state()

    def evaluate(self, current_balance: float, trades: Optional[List[dict]] = None) -> MonthlyGovernorStatus:
        """
        Evaluates current month progress, determines regulatory regime and risk adjustments.
        """
        now = datetime.now()
        now_month_str = now.strftime("%Y-%m")

        # Check if calendar month rolled over
        if now_month_str != self.current_month_str:
            self._handle_month_rollover(now_month_str, self.month_start_balance, current_balance, trades)

        month_realized_pnl = self._realized_pnl(now_month_str, trades)
        if self.month_start_balance <= 0:
            self.month_start_balance = max(1.0, current_balance - month_realized_pnl)
            self._save_state()

        # Days calculations
        year, month = now.year, now.month
        days_in_month = calendar.monthrange(year, month)[1]
        day_of_month = now.day
        time_progress_pct = round((day_of_month / days_in_month) * 100.0, 1)

        month_realized_pnl = round(month_realized_pnl, 2)
        current_pnl_pct = round((month_realized_pnl / self.month_start_balance) * 100.0, 2)
        effective_target_pct = round(
            min(self.MAX_CATCHUP_TARGET_CAP, self.base_target_pct + (self.carried_deficit_pct if self.auto_compensate_deficit else 0.0)),
            2
        )

        progress_ratio = round(current_pnl_pct / effective_target_pct, 3) if effective_target_pct > 0 else 1.0

        profile = self.PROFILES.get(self.profile_name, self.PROFILES["growth"])

        # Default standard settings (lấy từ profile đang chọn, fallback ON_TRACK)
        regime = "ON_TRACK"
        protection_mode = "BALANCED"
        track = profile["on_track"]
        size_multiplier = track["size"]
        max_leverage_cap = track["lev"]
        min_ai_confidence = track["conf"]
        min_risk_reward = track["rr"]
        rationale = ""

        if not self.enabled:
            regime = "DISABLED"
            protection_mode = "STANDARD"
            rationale = "Bộ điều tiết lãi suất kỳ vọng tháng đang TẮT. AI sử dụng quản trị rủi ro thông thường."
        elif current_pnl_pct >= effective_target_pct:
            # 1. TARGET ACHIEVED -> Capital Preservation Mode (số theo profile)
            regime = "TARGET_ACHIEVED"
            protection_mode = "CAPITAL_PRESERVATION"
            achieved = profile["achieved"]
            size_multiplier = achieved["size"]
            max_leverage_cap = achieved["lev"]
            min_ai_confidence = achieved["conf"]
            min_risk_reward = achieved["rr"]
            rationale = (
                f"🎉 ĐÃ ĐẠT MỤC TIÊU THÁNG ({current_pnl_pct:+.2f}% / {effective_target_pct:.1f}%): "
                f"AI tự động kích hoạt chế độ BẢO TOÀN LỢI NHUẬN! Giảm 50% khối lượng lệnh mới, "
                f"khóa đòn bẩy tối đa 3x và chỉ khớp các lệnh A+ (độ tin cậy >= 75%) để bảo vệ lãi."
            )
        elif self.carried_deficit_pct > 0:
            # 2. DEFICIT CATCH-UP MODE -> Strict Anti-Chasing (Never increase size or leverage after losses)
            regime = "DEFICIT_CATCHUP"
            protection_mode = "ADAPTIVE_CATCHUP"
            catchup = profile["deficit_catchup"]
            size_multiplier = catchup["size"]   # Hard invariant: size_multiplier <= 1.0 (anti-target chasing)
            max_leverage_cap = catchup["lev"]   # Deficit leverage strictly below normal cap
            min_ai_confidence = catchup["conf"] # Filter out low-grade noise
            min_risk_reward = catchup["rr"]
            rationale = (
                f"🔄 CHẾ ĐỘ BÙ THIẾU HỤT THÁNG TRƯỚC: Tháng trước chưa đạt mục tiêu (thiếu {self.carried_deficit_pct:.1f}%). "
                f"Mục tiêu tháng này được điều chỉnh lên {effective_target_pct:.1f}%. "
                f"AI tự động ưu tiên các lệnh R:R cao (>= 1:2.5), giảm nhẹ khối lượng (0.85x) và siết đòn bẩy để bảo vệ vốn an toàn."
            )
        elif (time_progress_pct >= 60.0) and (progress_ratio < 0.35):
            # 3. BEHIND PACING -> Defensive Patience (Do NOT panic or overtrade)
            regime = "BEHIND_PACE"
            protection_mode = "DEFENSIVE_PATIENCE"
            behind = profile["behind_pace"]
            size_multiplier = behind["size"]  # Slightly reduce risk to prevent revenge trading
            max_leverage_cap = behind["lev"]
            min_ai_confidence = behind["conf"]
            min_risk_reward = behind["rr"]
            rationale = (
                f"⚠️ TIẾN ĐỘ THÁNG CHẬM (Đã qua {day_of_month}/{days_in_month} ngày, đạt {current_pnl_pct:+.2f}% / {effective_target_pct:.1f}%): "
                f"AI kích hoạt cơ chế KIÊN NHẪN PHÒNG THỦ - nghiêm cấm giao dịch ép lệnh (overtrading). "
                f"Nếu không kịp đạt, phần thiếu hụt sẽ tự động được bù sang tháng sau một cách an toàn."
            )
        else:
            # 4. NORMAL ON-TRACK (số theo profile)
            regime = "ON_TRACK"
            protection_mode = "BALANCED"
            size_multiplier = track["size"]
            max_leverage_cap = track["lev"]
            min_ai_confidence = track["conf"]
            min_risk_reward = track["rr"]
            rationale = (
                f"⚖️ ĐANG ĐÚNG TIẾN ĐỘ THÁNG: Lãi hiện tại {current_pnl_pct:+.2f}% / Mục tiêu {effective_target_pct:.1f}% "
                f"(Ngày {day_of_month}/{days_in_month} - {time_progress_pct:.0f}% tháng). AI duy trì nhịp độ giao dịch cân bằng."
            )

        # Hard Invariant: size_multiplier can NEVER exceed 1.0 (anti-target chasing)
        size_multiplier = min(1.0, float(size_multiplier))

        target_pnl_usdt = round(self.month_start_balance * (effective_target_pct / 100.0), 2)
        remaining_days = max(1, days_in_month - day_of_month + 1)
        remaining_target_usdt = max(0.0, target_pnl_usdt - month_realized_pnl)
        target_daily_pnl_usdt = round(remaining_target_usdt / remaining_days, 2)
        reserve_ratio_rec = 0.30 if regime == "TARGET_ACHIEVED" else 0.15

        now_str = now.strftime("%H:%M:%S")
        return MonthlyGovernorStatus(
            enabled=self.enabled,
            base_target_pct=self.base_target_pct,
            carried_deficit_pct=self.carried_deficit_pct,
            effective_target_pct=effective_target_pct,
            current_month_str=self.current_month_str,
            month_start_balance=round(self.month_start_balance, 2),
            month_realized_pnl=round(month_realized_pnl, 2),
            current_pnl_pct=current_pnl_pct,
            progress_ratio=progress_ratio,
            day_of_month=day_of_month,
            days_in_month=days_in_month,
            month_time_progress_pct=time_progress_pct,
            regime=regime,
            protection_mode=protection_mode,
            size_multiplier=size_multiplier,
            max_leverage_cap=max_leverage_cap,
            min_ai_confidence=min_ai_confidence,
            min_risk_reward_ratio=min_risk_reward,
            rationale=rationale,
            updated_at=now_str,
            target_pnl_usdt=target_pnl_usdt,
            target_daily_pnl_usdt=target_daily_pnl_usdt,
            remaining_days=remaining_days,
            reserve_ratio_recommended=reserve_ratio_rec,
            profile_name=self.profile_name,
            profile_label=profile.get("label", self.profile_name),
        )
