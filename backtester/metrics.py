"""
Performance Metrics Calculator & Reporter
Calculates quantitative metrics: Win Rate, Sharpe, Max Drawdown, Profit Factor, etc.
"""
from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from tabulate import tabulate


@dataclass
class TradeRecord:
    trade_id: int
    symbol: str
    direction: str          # 'LONG' or 'SHORT'
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp
    exit_price: float
    units: float
    notional_value: float
    gross_pnl: float
    fees: float
    net_pnl: float
    return_pct: float
    exit_reason: str        # 'TP', 'SL', 'TRAILING_SL', 'SIGNAL_REVERSAL'
    holding_candles: int


def calculate_metrics(trades: List[TradeRecord], initial_balance: float, equity_curve: pd.Series) -> Dict[str, Any]:
    if not trades:
        return {"total_trades": 0, "net_profit": 0.0}

    pnls = [t.net_pnl for t in trades]
    returns = [t.return_pct for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    total_trades = len(trades)
    win_count = len(wins)
    loss_count = len(losses)
    win_rate = (win_count / total_trades) * 100.0 if total_trades > 0 else 0.0

    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (gross_profit if gross_profit > 0 else 0.0)

    total_fees = sum(t.fees for t in trades)
    net_profit = sum(pnls)
    final_balance = initial_balance + net_profit
    total_return_pct = (net_profit / initial_balance) * 100.0

    # Max Drawdown from equity curve
    if not equity_curve.empty:
        rolling_max = equity_curve.cummax()
        drawdowns = (equity_curve - rolling_max) / rolling_max
        max_drawdown_pct = abs(drawdowns.min()) * 100.0
        max_drawdown_usd = (rolling_max - equity_curve).max()
    else:
        max_drawdown_pct = 0.0
        max_drawdown_usd = 0.0

    # Sharpe Ratio (Assuming zero risk-free rate, daily resampling or per-trade)
    pnl_array = np.array(pnls)
    if len(pnl_array) > 1 and np.std(pnl_array) > 0:
        # Annualized approx (assuming 4 trades/day = ~1460 trades/year)
        sharpe_ratio = (np.mean(pnl_array) / np.std(pnl_array)) * np.sqrt(365 * 4)
    else:
        sharpe_ratio = 0.0

    # Sortino Ratio (Downside deviation)
    downside_losses = [p for p in pnls if p < 0]
    if len(downside_losses) > 1 and np.std(downside_losses) > 0:
        sortino_ratio = (np.mean(pnl_array) / np.std(downside_losses)) * np.sqrt(365 * 4)
    else:
        sortino_ratio = 0.0

    avg_win = np.mean(wins) if wins else 0.0
    avg_loss = abs(np.mean(losses)) if losses else 0.0
    payoff_ratio = (avg_win / avg_loss) if avg_loss > 0 else 0.0

    return {
        "initial_balance": initial_balance,
        "final_balance": final_balance,
        "net_profit": net_profit,
        "total_return_pct": total_return_pct,
        "total_trades": total_trades,
        "win_count": win_count,
        "loss_count": loss_count,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "total_fees": total_fees,
        "max_drawdown_pct": max_drawdown_pct,
        "max_drawdown_usd": max_drawdown_usd,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "avg_trade_pnl": np.mean(pnls),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "payoff_ratio": payoff_ratio,
    }


from typing import List, Dict, Any, Optional

def print_performance_report(metrics: Dict[str, Any], symbol: str, strategy_name: str, trades: Optional[List[TradeRecord]] = None):
    if metrics["total_trades"] == 0:
        print("[Backtest] Không có lệnh nào được tạo trong giai đoạn này.")
        return

    table_data = [
        ["Chiến lược (Strategy)", strategy_name],
        ["Cặp giao dịch (Symbol)", symbol],
        ["Vốn ban đầu (Initial Balance)", f"${metrics['initial_balance']:,.2f}"],
        ["Vốn kết thúc (Final Balance)", f"${metrics['final_balance']:,.2f}"],
        ["Lợi nhuận ròng (Net Profit)", f"${metrics['net_profit']:+,.2f} ({metrics['total_return_pct']:+.2f}%)"],
        ["Tổng số lệnh (Total Trades)", metrics["total_trades"]],
        ["Lệnh thắng / thua (Wins / Losses)", f"{metrics['win_count']} / {metrics['loss_count']}"],
        ["Tỷ lệ thắng (Win Rate)", f"{metrics['win_rate']:.2f}%"],
        ["Tỷ số Lời/Lỗ (Profit Factor)", f"{metrics['profit_factor']:.2f}"],
        ["Tỷ số Thắng/Thua (Payoff Ratio)", f"{metrics['payoff_ratio']:.2f}"],
        ["Tổng phí sàn & Trượt giá (Fees)", f"${metrics['total_fees']:,.2f}"],
        ["Drawdown lớn nhất (Max Drawdown)", f"-{metrics['max_drawdown_pct']:.2f}% (${metrics['max_drawdown_usd']:,.2f})"],
        ["Sharpe Ratio (Annualized)", f"{metrics['sharpe_ratio']:.2f}"],
        ["Sortino Ratio (Annualized)", f"{metrics['sortino_ratio']:.2f}"],
        ["Lợi nhuận TB/lệnh (Avg PnL/Trade)", f"${metrics['avg_trade_pnl']:+,.2f}"],
    ]

    print("\n" + "=" * 65)
    print(f"       BÁO CÁO KẾT QUẢ BACKTEST BINANCE FUTURES")
    print("=" * 65)
    print(tabulate(table_data, tablefmt="fancy_grid"))
    print("=" * 65)

    if trades:
        trade_rows = []
        for t in trades[-10:]:  # Show last 10 trades
            trade_rows.append([
                t.trade_id,
                t.direction,
                t.entry_time.strftime("%m-%d %H:%M"),
                f"${t.entry_price:,.1f}",
                t.exit_time.strftime("%m-%d %H:%M"),
                f"${t.exit_price:,.1f}",
                t.exit_reason,
                f"${t.net_pnl:+,.2f}",
                f"{t.return_pct:+,.1f}%"
            ])
        headers = ["ID", "Vị thế", "Vào lệnh", "Giá vào", "Thoát lệnh", "Giá ra", "Lý do", "PnL ($)", "Tỷ suất"]
        print("\n📜 NHẬT KÝ 10 GIAO DỊCH GẦN NHẤT:")
        print(tabulate(trade_rows, headers=headers, tablefmt="simple"))
        print("\n")
