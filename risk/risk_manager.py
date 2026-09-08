"""
Binance Futures Risk Management Engine
Controls position sizing, leverage limits, liquidation distance, and daily drawdown circuit breakers.
"""
from dataclasses import dataclass
import math
from datetime import datetime, timezone
from typing import Optional, Tuple
from config.settings import RiskConfig, DEFAULT_RISK
from risk.antigravity_risk_officer import AntigravityRiskOfficer, AIRiskVerdict


@dataclass
class PositionProposal:
    symbol: str
    direction: int            # 1 for LONG, -1 for SHORT
    entry_price: float
    stop_loss: float
    take_profit: float
    units: float              # Number of coins (e.g. 0.15 BTC)
    notional_value: float     # Total position value in USDT (units * entry_price)
    required_margin: float    # Margin required = notional / leverage
    leverage: int
    risk_amount: float        # Total USDT risked if hit SL
    est_liquidation_price: float
    approved: bool
    rejection_reason: Optional[str] = None


class FuturesRiskManager:
    def __init__(self, config: RiskConfig = DEFAULT_RISK):
        self.config = config
        self.daily_start_balance: float = config.initial_balance
        self.current_balance: float = config.initial_balance
        self.circuit_breaker_active: bool = False
        self.ai_cro = AntigravityRiskOfficer(initial_balance=config.initial_balance)
        self.risk_day = datetime.now(timezone.utc).date().isoformat()

    def reset_daily_stats(self, new_balance: float):
        """Call at 00:00 UTC each day"""
        self.daily_start_balance = new_balance
        self.current_balance = new_balance
        self.circuit_breaker_active = False
        self.ai_cro.daily_start_balance = new_balance
        self.ai_cro.current_balance = new_balance
        self.ai_cro.circuit_breaker = False
        self.ai_cro.last_verdict = None
        self.risk_day = datetime.now(timezone.utc).date().isoformat()

    def sync_day(self):
        if datetime.now(timezone.utc).date().isoformat() != self.risk_day:
            self.reset_daily_stats(self.current_balance)

    def export_state(self):
        return {"risk_day": self.risk_day, "daily_start_balance": self.daily_start_balance,
                "circuit_breaker_active": self.circuit_breaker_active, "cro_breaker": self.ai_cro.circuit_breaker}

    def restore_state(self, saved):
        if saved:
            self.risk_day = saved["risk_day"]
            self.daily_start_balance = float(saved["daily_start_balance"])
            self.ai_cro.daily_start_balance = self.daily_start_balance
            self.circuit_breaker_active = bool(saved["circuit_breaker_active"])
            self.ai_cro.circuit_breaker = bool(saved["cro_breaker"])
        self.sync_day()

    def update_balance(self, new_balance: float):
        self.current_balance = new_balance
        # Check Daily Drawdown Circuit Breaker
        daily_loss = (self.daily_start_balance - self.current_balance) / self.daily_start_balance
        if daily_loss >= self.config.daily_max_loss_pct:
            self.circuit_breaker_active = True

    # Binance USD(S)-M position brackets (BTCUSDT, notional USDT):
    # (upper_notional, mmr). Tier 1 ~0.40%, tang dan theo bac de tranh danh gia thap liq khi lenh lon.
    MMR_BRACKETS = (
        (5_000_000, 0.004),
        (25_000_000, 0.005),
        (50_000_000, 0.010),
        (250_000_000, 0.025),
        (float("inf"), 0.050),
    )

    @classmethod
    def mmr_for_notional(cls, notional_usdt: float) -> float:
        for upper, mmr in cls.MMR_BRACKETS:
            if notional_usdt <= upper:
                return mmr
        return 0.050

    def calculate_liquidation_price(
        self,
        entry_price: float,
        direction: int,
        leverage: int,
        mmr: float = 0.0,  # 0 = tu tra theo bac notional (entry * leverage)
        notional_usdt: float = 0.0,
        taker_fee_rate: float = 0.0005,  # tru phi dong/mo uoc tinh de liq bao thu hon
    ) -> float:
        """
        Estimate Isolated liquidation price on Binance Futures.
        Liq = Entry * (1 -/+ 1/leverage + MMR + phi) de phan anh chi phi that.
        """
        lev = max(1, int(leverage))
        if mmr <= 0:
            base_notional = notional_usdt if notional_usdt > 0 else entry_price * lev
            mmr = self.mmr_for_notional(base_notional)
        cushion = mmr + taker_fee_rate
        if direction == 1:  # LONG
            # Liq = Entry * (1 - 1/leverage + mmr + phi)
            return max(0.0, entry_price * (1.0 - (1.0 / lev) + cushion))
        else:  # SHORT
            # Liq = Entry * (1 + 1/leverage - mmr - phi)
            return entry_price * (1.0 + (1.0 / lev) - cushion)

    def evaluate_order(
        self,
        symbol: str,
        direction: int,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        leverage: Optional[int] = None
    ) -> PositionProposal:
        """
        Evaluate and calculate safe position size based on account risk and Stop Loss distance.
        """
        used_leverage = leverage or self.config.default_leverage
        used_leverage = max(1, min(used_leverage, self.config.max_leverage))
        if direction not in (-1, 1) or any(not math.isfinite(v) or v <= 0 for v in (entry_price, stop_loss, take_profit, self.current_balance)):
            return PositionProposal(symbol, direction, entry_price, stop_loss, take_profit, 0, 0, 0, used_leverage, 0, 0, False, "Invalid price, balance or direction")
        if (take_profit - entry_price) * direction <= 0:
            return PositionProposal(symbol, direction, entry_price, stop_loss, take_profit, 0, 0, 0, used_leverage, 0, 0, False, "Take profit is on the wrong side")

        # 1. Check Circuit Breaker
        if self.circuit_breaker_active or self.ai_cro.circuit_breaker or (self.ai_cro.last_verdict and self.ai_cro.last_verdict.circuit_breaker_active):
            return PositionProposal(
                symbol=symbol, direction=direction, entry_price=entry_price,
                stop_loss=stop_loss, take_profit=take_profit, units=0.0,
                notional_value=0.0, required_margin=0.0, leverage=used_leverage,
                risk_amount=0.0, est_liquidation_price=0.0, approved=False,
                rejection_reason=f"Circuit Breaker kích hoạt: Lỗ ngày >= {self.config.daily_max_loss_pct*100}%"
            )

        # 2. Check Stop Loss validity
        stop_distance = abs(entry_price - stop_loss)
        if stop_distance <= 0 or (direction == 1 and stop_loss >= entry_price) or (direction == -1 and stop_loss <= entry_price):
            return PositionProposal(
                symbol=symbol, direction=direction, entry_price=entry_price,
                stop_loss=stop_loss, take_profit=take_profit, units=0.0,
                notional_value=0.0, required_margin=0.0, leverage=used_leverage,
                risk_amount=0.0, est_liquidation_price=0.0, approved=False,
                rejection_reason="Mức Stop Loss không hợp lệ so với giá vào lệnh."
            )

        # 3. Position Sizing based on Antigravity AI Dynamic Risk Allocation
        if self.ai_cro.last_verdict and not self.ai_cro.last_verdict.circuit_breaker_active:
            risk_pct = self.ai_cro.last_verdict.risk_per_trade_pct / 100.0
            margin_cap_pct = self.ai_cro.last_verdict.max_margin_utilization_pct / 100.0
        else:
            risk_pct = self.config.risk_per_trade_pct
            margin_cap_pct = 0.40

        risk_budget = self.current_balance * risk_pct
        units = risk_budget / stop_distance
        notional_value = units * entry_price
        required_margin = notional_value / used_leverage

        # 4. Check if required margin exceeds Antigravity AI margin cap
        max_allowed_margin = self.current_balance * margin_cap_pct
        if required_margin > max_allowed_margin:
            # Scale down units to match AI safe margin budget
            required_margin = max_allowed_margin
            notional_value = required_margin * used_leverage
            units = notional_value / entry_price
            risk_budget = units * stop_distance

        # 5. Check Liquidation Safety Buffer
        liq_price = self.calculate_liquidation_price(
            entry_price,
            direction,
            used_leverage,
            notional_usdt=notional_value,
        )
        # Ensure Stop Loss is triggered BEFORE liquidation price with a safe gap
        buffer = entry_price * 0.005
        if direction == 1 and stop_loss <= liq_price + buffer:
            return PositionProposal(
                symbol=symbol, direction=direction, entry_price=entry_price,
                stop_loss=stop_loss, take_profit=take_profit, units=0.0,
                notional_value=0.0, required_margin=0.0, leverage=used_leverage,
                risk_amount=0.0, est_liquidation_price=liq_price, approved=False,
                rejection_reason=f"Cảnh báo rủi ro: Stop Loss ({stop_loss:.2f}) nằm dưới giá thanh lý ({liq_price:.2f})."
            )
        elif direction == -1 and stop_loss >= liq_price - buffer:
            return PositionProposal(
                symbol=symbol, direction=direction, entry_price=entry_price,
                stop_loss=stop_loss, take_profit=take_profit, units=0.0,
                notional_value=0.0, required_margin=0.0, leverage=used_leverage,
                risk_amount=0.0, est_liquidation_price=liq_price, approved=False,
                rejection_reason=f"Cảnh báo rủi ro: Stop Loss ({stop_loss:.2f}) nằm trên giá thanh lý ({liq_price:.2f})."
            )

        return PositionProposal(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            units=round(units, 4),
            notional_value=round(notional_value, 2),
            required_margin=round(required_margin, 2),
            leverage=used_leverage,
            risk_amount=round(risk_budget, 2),
            est_liquidation_price=round(liq_price, 2),
            approved=True
        )
