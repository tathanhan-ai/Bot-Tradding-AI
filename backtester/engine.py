"""
Event-Driven Backtesting Engine for Binance Futures
Accurately models fees, slippage, intrabar SL/TP hits, and trailing stops.
"""
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

from config.settings import (
    RiskConfig,
    BinanceFees,
    DEFAULT_RISK,
    DEFAULT_FEES,
)
from risk.risk_manager import FuturesRiskManager, PositionProposal
from strategy.base import BaseStrategy
from backtester.metrics import TradeRecord, calculate_metrics, print_performance_report


class BacktestEngine:
    def __init__(
        self,
        strategy: BaseStrategy,
        risk_config: RiskConfig = DEFAULT_RISK,
        fees_config: BinanceFees = DEFAULT_FEES,
    ):
        self.strategy = strategy
        self.risk_config = risk_config
        self.fees_config = fees_config
        self.risk_manager = FuturesRiskManager(risk_config)

    def run(self, data_map: Dict[str, pd.DataFrame], symbol: str = "BTCUSDT") -> Dict:
        """
        Run backtest on multi-timeframe candle data.
        """
        # 1. Generate Signals
        print(f"[Backtest] Đang tạo tín hiệu giao dịch với chiến lược: {self.strategy.name}...")
        df_entry = self.strategy.generate_signals(data_map)

        # 2. Iterate bar by bar
        trades: List[TradeRecord] = []
        equity_records = []
        current_balance = self.risk_config.initial_balance

        in_position = False
        pos_direction = 0     # 1: Long, -1: Short
        pos_entry_price = 0.0
        pos_entry_time = None
        pos_units = 0.0
        pos_sl = 0.0
        pos_tp = 0.0
        pos_notional = 0.0
        initial_risk_dist = 0.0
        holding_candles = 0
        peak_price = 0.0
        trade_id = 0

        candles = df_entry.itertuples()

        for row in candles:
            timestamp = row.Index
            open_p = row.open
            high_p = row.high
            low_p = row.low
            close_p = row.close
            atr = getattr(row, "atr", (high_p - low_p))
            signal = getattr(row, "signal", 0)
            sig_sl = getattr(row, "stop_loss", np.nan)
            sig_tp = getattr(row, "take_profit", np.nan)

            # Record current equity
            if in_position:
                unrealized_pnl = (close_p - pos_entry_price) * pos_units * pos_direction
                equity_records.append({"timestamp": timestamp, "equity": current_balance + unrealized_pnl})
            else:
                equity_records.append({"timestamp": timestamp, "equity": current_balance})

            # Check existing position management
            if in_position:
                holding_candles += 1
                exit_price = None
                exit_reason = None

                if pos_direction == 1:  # LONG POSITION
                    # Update peak price for Trailing Stop
                    if high_p > peak_price:
                        peak_price = high_p
                        # If reached 1R profit -> Move SL to Breakeven
                        if (peak_price - pos_entry_price) >= initial_risk_dist:
                            new_be_sl = max(pos_sl, pos_entry_price)
                            # Dynamic Trailing Stop behind peak
                            trailing_sl = peak_price - (1.0 * atr)
                            pos_sl = max(new_be_sl, trailing_sl)

                    # Check TP hit (High >= Take Profit)
                    if high_p >= pos_tp:
                        exit_price = pos_tp
                        exit_reason = "TAKE_PROFIT"
                    # Check SL hit (Low <= Stop Loss)
                    elif low_p <= pos_sl:
                        # Slippage at SL: execute at min(open, pos_sl)
                        exit_price = min(open_p, pos_sl) if open_p < pos_sl else pos_sl
                        exit_reason = "STOP_LOSS" if pos_sl < pos_entry_price else "TRAILING_STOP"
                    # Check Signal reversal
                    elif signal == -1:
                        exit_price = close_p
                        exit_reason = "REVERSAL"

                elif pos_direction == -1:  # SHORT POSITION
                    # Update trough price for Trailing Stop
                    if low_p < peak_price:
                        peak_price = low_p
                        # If reached 1R profit -> Move SL to Breakeven
                        if (pos_entry_price - peak_price) >= initial_risk_dist:
                            new_be_sl = min(pos_sl, pos_entry_price)
                            # Dynamic Trailing Stop above trough
                            trailing_sl = peak_price + (1.0 * atr)
                            pos_sl = min(new_be_sl, trailing_sl)

                    # Check TP hit (Low <= Take Profit)
                    if low_p <= pos_tp:
                        exit_price = pos_tp
                        exit_reason = "TAKE_PROFIT"
                    # Check SL hit (High >= Stop Loss)
                    elif high_p >= pos_sl:
                        exit_price = max(open_p, pos_sl) if open_p > pos_sl else pos_sl
                        exit_reason = "STOP_LOSS" if pos_sl > pos_entry_price else "TRAILING_STOP"
                    # Check Signal reversal
                    elif signal == 1:
                        exit_price = close_p
                        exit_reason = "REVERSAL"

                # Execute Exit if triggered
                if exit_price is not None:
                    gross_pnl = (exit_price - pos_entry_price) * pos_units * pos_direction
                    # Binance Futures Taker Fees on entry and exit
                    entry_fee = pos_notional * (self.fees_config.taker_fee + self.fees_config.slippage)
                    exit_fee = (pos_units * exit_price) * (self.fees_config.taker_fee + self.fees_config.slippage)
                    total_fees = entry_fee + exit_fee
                    net_pnl = gross_pnl - total_fees

                    current_balance += net_pnl
                    self.risk_manager.update_balance(current_balance)

                    trade_id += 1
                    trades.append(
                        TradeRecord(
                            trade_id=trade_id,
                            symbol=symbol,
                            direction="LONG" if pos_direction == 1 else "SHORT",
                            entry_time=pos_entry_time,
                            entry_price=pos_entry_price,
                            exit_time=timestamp,
                            exit_price=exit_price,
                            units=pos_units,
                            notional_value=pos_notional,
                            gross_pnl=round(gross_pnl, 2),
                            fees=round(total_fees, 2),
                            net_pnl=round(net_pnl, 2),
                            return_pct=round((net_pnl / pos_notional) * 100 * self.risk_config.default_leverage, 2),
                            exit_reason=exit_reason,
                            holding_candles=holding_candles,
                        )
                    )

                    in_position = False
                    pos_direction = 0

            # Evaluate New Entry Signal if not in position
            if not in_position and signal in (1, -1) and not np.isnan(sig_sl) and not np.isnan(sig_tp):
                entry_p = close_p
                # Evaluate order proposal through risk manager
                proposal: PositionProposal = self.risk_manager.evaluate_order(
                    symbol=symbol,
                    direction=signal,
                    entry_price=entry_p,
                    stop_loss=sig_sl,
                    take_profit=sig_tp,
                    leverage=self.risk_config.default_leverage
                )

                if proposal.approved and proposal.units > 0:
                    in_position = True
                    pos_direction = signal
                    pos_entry_price = entry_p
                    pos_entry_time = timestamp
                    pos_units = proposal.units
                    pos_notional = proposal.notional_value
                    pos_sl = sig_sl
                    pos_tp = sig_tp
                    initial_risk_dist = abs(entry_p - sig_sl)
                    peak_price = entry_p
                    holding_candles = 0

        equity_df = pd.DataFrame(equity_records).set_index("timestamp")["equity"]
        metrics = calculate_metrics(trades, self.risk_config.initial_balance, equity_df)
        print_performance_report(metrics, symbol, self.strategy.name, trades)

        return {
            "metrics": metrics,
            "trades": trades,
            "equity_curve": equity_df,
            "df_entry": df_entry
        }
