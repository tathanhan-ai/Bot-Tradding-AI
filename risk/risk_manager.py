"""
Binance Futures Risk Management Engine
Controls position sizing, leverage limits, liquidation distance, and daily drawdown circuit breakers.
"""
from dataclasses import dataclass
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

    def reset_daily_stats(self, new_balance: float):
        """Call at 00:00 UTC each day"""
        self.daily_start_balance = new_balance
        self.current_balance = new_balance
        self.circuit_breaker_active = False

    def update_balance(self, new_balance: float):
        self.current_balance = new_balance
        # Check Daily Drawdown Circuit Breaker
        daily_loss = (self.daily_start_balance - self.current_balance) / self.daily_start_balance
        if daily_loss >= self.config.daily_max_loss_pct:
            self.circuit_breaker_active = True

    def calculate_liquidation_price(
        self,
        entry_price: float,
        direction: int,
        leverage: int,
        mmr: float = 0.005 # Maintenance margin rate ~ 0.5% on Binance Tier 1
    ) -> float:
        """
        Estimate Isolated liquidation price on Binance Futures
        """
        if direction == 1:  # LONG
            # Liq = Entry * (1 - 1/leverage + mmr)
            return max(0.0, entry_price * (1.0 - (1.0 / leverage) + mmr))
        else:  # SHORT
            # Liq = Entry * (1 + 1/leverage - mmr)
            return entry_price * (1.0 + (1.0 / leverage) - mmr)

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
        used_leverage = min(used_leverage, self.config.max_leverage)

        # 1. Check Circuit Breaker
        if self.circuit_breaker_active:
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
        liq_price = self.calculate_liquidation_price(entry_price, direction, used_leverage)
        # Ensure Stop Loss is triggered BEFORE liquidation price with a safe gap
        if direction == 1 and stop_loss <= liq_price:
            return PositionProposal(
                symbol=symbol, direction=direction, entry_price=entry_price,
                stop_loss=stop_loss, take_profit=take_profit, units=0.0,
                notional_value=0.0, required_margin=0.0, leverage=used_leverage,
                risk_amount=0.0, est_liquidation_price=liq_price, approved=False,
                rejection_reason=f"Cảnh báo rủi ro: Stop Loss ({stop_loss:.2f}) nằm dưới giá thanh lý ({liq_price:.2f})."
            )
        elif direction == -1 and stop_loss >= liq_price:
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
