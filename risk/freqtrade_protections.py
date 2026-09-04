"""
Freqtrade-Grade Capital Protection Framework for Binance Futures
Modeled after https://github.com/freqtrade/freqtrade/blob/develop/freqtrade/plugins/protections/

Key Mechanisms:
1. StoplossGuard: Locks trading if X stoplosses occur within a lookback window (prevents revenge trading in bad regimes).
2. MaxDrawdownGuard: Pauses trading if portfolio drawdown from peak exceeds safety threshold (e.g. 3.5%).
3. CooldownPeriod: Enforces mandatory resting period after trade closure before opening a new position.
4. FeeDragFilter: Guarantees that the expected net profit after fees and slippage exceeds exchange fee hurdles by >= 3x.
"""
import sys
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


@dataclass
class ProtectionStatus:
    status: str              # 'NORMAL', 'COOLDOWN', 'STOPLOSS_GUARD', 'MAX_DRAWDOWN', 'FEE_DRAG'
    is_locked: bool          # True if trading is currently halted
    lock_reason: str         # Human-readable explanation in Vietnamese
    locked_until: float      # Timestamp when lock expires (0.0 if not locked)
    remaining_seconds: int   # Remaining seconds of lock
    recent_stoploss_count: int = 0
    current_drawdown_pct: float = 0.0
    peak_balance: float = 1000.0


class StoplossGuard:
    """
    Stops trading if trade_limit stoplosses occur within lookback_period_seconds.
    In Freqtrade: StoplossGuard(lookback_period_candles, trade_limit, stop_duration_candles).
    """
    def __init__(
        self,
        trade_limit: int = 2,
        lookback_seconds: float = 3600.0,       # 60 minutes
        stop_duration_seconds: float = 1800.0   # 30 minutes lock
    ):
        self.trade_limit = trade_limit
        self.lookback_seconds = lookback_seconds
        self.stop_duration_seconds = stop_duration_seconds
        self.stoploss_timestamps: List[float] = []
        self.locked_until: float = 0.0
        self.lock_reason: str = ""

    def record_loss(self, loss_amount: float, reason: str, timestamp: Optional[float] = None):
        ts = timestamp or time.time()
        self.stoploss_timestamps.append(ts)
        self._prune_old_events(ts)

        # If stoplosses in lookback window reach limit -> Trigger Lock
        if len(self.stoploss_timestamps) >= self.trade_limit:
            self.locked_until = ts + self.stop_duration_seconds
            mins = int(self.stop_duration_seconds // 60)
            self.lock_reason = (
                f"🛡️ [FREQTRADE STOPLOSS GUARD KÍCH HOẠT] Đã chạm {len(self.stoploss_timestamps)} lệnh cắt lỗ "
                f"trong {int(self.lookback_seconds // 60)} phút vừa qua. Tự động đóng băng mở lệnh {mins} phút "
                f"để bảo toàn vốn và triệt tiêu tâm lý giao dịch trả thù (Revenge Trading)!"
            )
            print(self.lock_reason, flush=True)

    def _prune_old_events(self, now: float):
        cutoff = now - self.lookback_seconds
        self.stoploss_timestamps = [t for t in self.stoploss_timestamps if t >= cutoff]

    def is_locked(self, now: Optional[float] = None) -> Tuple[bool, str, float]:
        ts = now or time.time()
        self._prune_old_events(ts)
        if ts < self.locked_until:
            return True, self.lock_reason, self.locked_until
        return False, "", 0.0

    def get_recent_count(self, now: Optional[float] = None) -> int:
        ts = now or time.time()
        self._prune_old_events(ts)
        return len(self.stoploss_timestamps)


class MaxDrawdownGuard:
    """
    Stops trading if portfolio drawdown from peak balance exceeds max_allowed_drawdown.
    In Freqtrade: MaxDrawdown(max_allowed_drawdown, lookback_period_candles, stop_duration_candles).
    """
    def __init__(
        self,
        max_allowed_drawdown: float = 0.035,     # 3.5% portfolio drawdown limit
        stop_duration_seconds: float = 3600.0   # 60 minutes cooling off
    ):
        self.max_allowed_drawdown = max_allowed_drawdown
        self.stop_duration_seconds = stop_duration_seconds
        self.peak_balance: float = 1000.0
        self.locked_until: float = 0.0
        self.lock_reason: str = ""

    def update_balance(self, current_balance: float, current_equity: float, now: Optional[float] = None):
        ts = now or time.time()
        effective = max(current_balance, current_equity)
        if effective > self.peak_balance:
            self.peak_balance = effective

        if self.peak_balance > 0:
            dd = (self.peak_balance - min(current_balance, current_equity)) / self.peak_balance
            if dd >= self.max_allowed_drawdown and ts >= self.locked_until:
                self.locked_until = ts + self.stop_duration_seconds
                self.lock_reason = (
                    f"🛑 [FREQTRADE MAX DRAWDOWN GUARD] Sụt giảm vốn {dd*100:.2f}% đã chạm trần cho phép "
                    f"({self.max_allowed_drawdown*100:.1f}% từ đỉnh ${self.peak_balance:,.2f}). "
                    f"Tạm dừng mở lệnh {int(self.stop_duration_seconds // 60)} phút để bảo toàn tài sản!"
                )
                print(self.lock_reason, flush=True)

    def is_locked(self, now: Optional[float] = None) -> Tuple[bool, str, float]:
        ts = now or time.time()
        if ts < self.locked_until:
            return True, self.lock_reason, self.locked_until
        return False, "", 0.0

    def get_drawdown_pct(self, current_balance: float) -> float:
        if self.peak_balance <= 0:
            return 0.0
        dd = (self.peak_balance - current_balance) / self.peak_balance
        return max(0.0, round(dd * 100.0, 2))


class CooldownPeriod:
    """
    Enforces a resting cooldown after closing any trade before taking another trade.
    In Freqtrade: CooldownPeriod(stop_duration_candles).
    """
    def __init__(self, default_cooldown_seconds: float = 30.0):
        self.default_cooldown_seconds = default_cooldown_seconds
        self.last_trade_closed_time: float = 0.0

    def on_trade_closed(self, timestamp: Optional[float] = None):
        self.last_trade_closed_time = timestamp or time.time()

    def is_locked(self, now: Optional[float] = None) -> Tuple[bool, str, float]:
        ts = now or time.time()
        passed = ts - self.last_trade_closed_time
        if passed < self.default_cooldown_seconds:
            remaining = int(self.default_cooldown_seconds - passed)
            return True, f"⏳ [FREQTRADE COOLDOWN] Đang trong thời gian tĩnh tâm ({remaining}s còn lại)", self.last_trade_closed_time + self.default_cooldown_seconds
        return False, "", 0.0


class FreqtradeProtectionEngine:
    """
    Unified Capital Protection Coordinator combining:
    - StoplossGuard
    - MaxDrawdownGuard
    - CooldownPeriod
    - FeeDragFilter
    """
    def __init__(self, initial_balance: float = 1000.0):
        self.initial_balance = initial_balance
        self.stoploss_guard = StoplossGuard(trade_limit=2, lookback_seconds=3600.0, stop_duration_seconds=1800.0)
        self.max_drawdown_guard = MaxDrawdownGuard(max_allowed_drawdown=0.035, stop_duration_seconds=3600.0)
        self.cooldown_guard = CooldownPeriod(default_cooldown_seconds=30.0)
        self.max_drawdown_guard.peak_balance = initial_balance

    def on_trade_closed(self, trade: dict):
        pnl = trade.get("pnl", 0.0)
        reason = trade.get("reason", "")
        now = time.time()

        self.cooldown_guard.on_trade_closed(now)

        # If loss
        if pnl <= 0:
            self.stoploss_guard.record_loss(abs(pnl), reason, now)

    def update_balance(self, balance: float, equity: float):
        self.max_drawdown_guard.update_balance(balance, equity)

    def validate_new_trade(
        self,
        entry_price: float,
        target_price: float,
        direction: int,
        balance: float,
        equity: float
    ) -> Tuple[bool, str, ProtectionStatus]:
        """
        Validates all Freqtrade protections before allowing a new order.
        Returns (approved, rejection_reason, status_object).
        """
        now = time.time()
        self.update_balance(balance, equity)

        # 1. Check Max Drawdown Guard
        is_dd_locked, dd_reason, dd_until = self.max_drawdown_guard.is_locked(now)
        if is_dd_locked:
            status = ProtectionStatus(
                status="MAX_DRAWDOWN",
                is_locked=True,
                lock_reason=dd_reason,
                locked_until=dd_until,
                remaining_seconds=max(0, int(dd_until - now)),
                recent_stoploss_count=self.stoploss_guard.get_recent_count(now),
                current_drawdown_pct=self.max_drawdown_guard.get_drawdown_pct(balance),
                peak_balance=self.max_drawdown_guard.peak_balance
            )
            return False, dd_reason, status

        # 2. Check Stoploss Guard
        is_sl_locked, sl_reason, sl_until = self.stoploss_guard.is_locked(now)
        if is_sl_locked:
            status = ProtectionStatus(
                status="STOPLOSS_GUARD",
                is_locked=True,
                lock_reason=sl_reason,
                locked_until=sl_until,
                remaining_seconds=max(0, int(sl_until - now)),
                recent_stoploss_count=self.stoploss_guard.get_recent_count(now),
                current_drawdown_pct=self.max_drawdown_guard.get_drawdown_pct(balance),
                peak_balance=self.max_drawdown_guard.peak_balance
            )
            return False, sl_reason, status

        # 3. Check Cooldown Period
        is_cd_locked, cd_reason, cd_until = self.cooldown_guard.is_locked(now)
        if is_cd_locked:
            status = ProtectionStatus(
                status="COOLDOWN",
                is_locked=True,
                lock_reason=cd_reason,
                locked_until=cd_until,
                remaining_seconds=max(0, int(cd_until - now)),
                recent_stoploss_count=self.stoploss_guard.get_recent_count(now),
                current_drawdown_pct=self.max_drawdown_guard.get_drawdown_pct(balance),
                peak_balance=self.max_drawdown_guard.peak_balance
            )
            return False, cd_reason, status

        # 4. Fee Drag Filter: Target profit must be >= 0.35% (> 3.5x Binance round-trip fee)
        if entry_price > 0 and target_price > 0:
            target_pct = abs(target_price - entry_price) / entry_price
            if target_pct < 0.0035:
                fee_reason = (
                    f"⚠️ [FREQTRADE FEE DRAG] Biên lợi nhuận mục tiêu ({target_pct*100:.3f}%) quá hẹp. "
                    f"Không đủ bù phí sàn và trượt giá (yêu cầu >= 0.35%). Từ chối mở lệnh!"
                )
                status = ProtectionStatus(
                    status="FEE_DRAG",
                    is_locked=False,
                    lock_reason=fee_reason,
                    locked_until=0.0,
                    remaining_seconds=0,
                    recent_stoploss_count=self.stoploss_guard.get_recent_count(now),
                    current_drawdown_pct=self.max_drawdown_guard.get_drawdown_pct(balance),
                    peak_balance=self.max_drawdown_guard.peak_balance
                )
                return False, fee_reason, status

        # All protections passed!
        status = ProtectionStatus(
            status="NORMAL",
            is_locked=False,
            lock_reason="Bình thường 🟢 (Tất cả chốt chặn Freqtrade sẵn sàng)",
            locked_until=0.0,
            remaining_seconds=0,
            recent_stoploss_count=self.stoploss_guard.get_recent_count(now),
            current_drawdown_pct=self.max_drawdown_guard.get_drawdown_pct(balance),
            peak_balance=self.max_drawdown_guard.peak_balance
        )
        return True, "Approved", status

    def get_status(self, balance: float = 1000.0, equity: float = 1000.0) -> ProtectionStatus:
        now = time.time()
        self.update_balance(balance, equity)

        is_dd_locked, dd_reason, dd_until = self.max_drawdown_guard.is_locked(now)
        if is_dd_locked:
            return ProtectionStatus(
                status="MAX_DRAWDOWN",
                is_locked=True,
                lock_reason=dd_reason,
                locked_until=dd_until,
                remaining_seconds=max(0, int(dd_until - now)),
                recent_stoploss_count=self.stoploss_guard.get_recent_count(now),
                current_drawdown_pct=self.max_drawdown_guard.get_drawdown_pct(balance),
                peak_balance=self.max_drawdown_guard.peak_balance
            )

        is_sl_locked, sl_reason, sl_until = self.stoploss_guard.is_locked(now)
        if is_sl_locked:
            return ProtectionStatus(
                status="STOPLOSS_GUARD",
                is_locked=True,
                lock_reason=sl_reason,
                locked_until=sl_until,
                remaining_seconds=max(0, int(sl_until - now)),
                recent_stoploss_count=self.stoploss_guard.get_recent_count(now),
                current_drawdown_pct=self.max_drawdown_guard.get_drawdown_pct(balance),
                peak_balance=self.max_drawdown_guard.peak_balance
            )

        is_cd_locked, cd_reason, cd_until = self.cooldown_guard.is_locked(now)
        if is_cd_locked:
            return ProtectionStatus(
                status="COOLDOWN",
                is_locked=True,
                lock_reason=cd_reason,
                locked_until=cd_until,
                remaining_seconds=max(0, int(cd_until - now)),
                recent_stoploss_count=self.stoploss_guard.get_recent_count(now),
                current_drawdown_pct=self.max_drawdown_guard.get_drawdown_pct(balance),
                peak_balance=self.max_drawdown_guard.peak_balance
            )

        return ProtectionStatus(
            status="NORMAL",
            is_locked=False,
            lock_reason="Bình thường 🟢 (Tất cả chốt chặn Freqtrade sẵn sàng)",
            locked_until=0.0,
            remaining_seconds=0,
            recent_stoploss_count=self.stoploss_guard.get_recent_count(now),
            current_drawdown_pct=self.max_drawdown_guard.get_drawdown_pct(balance),
            peak_balance=self.max_drawdown_guard.peak_balance
        )
