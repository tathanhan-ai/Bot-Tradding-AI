"""
Configuration settings for Binance USD(S)-M Futures Bot
"""
from dataclasses import dataclass
from pathlib import Path
import os

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("BOT_DATA_CACHE", "D:/Codex/cache/binance-futures-bot" if os.name == "nt" else str(BASE_DIR / "data" / "cache")))
DATA_DIR.mkdir(parents=True, exist_ok=True)

@dataclass
class RiskConfig:
    initial_balance: float = 10000.0   # USD
    risk_per_trade_pct: float = 0.015  # 1.5% balance risk per trade
    max_account_risk_pct: float = 0.05 # Max 5% total open risk across all positions
    default_leverage: int = 3          # Default leverage 3x (safe zone)
    max_leverage: int = 5              # Hard cap leverage
    daily_max_loss_pct: float = 0.04   # Circuit breaker: stop bot if daily loss >= 4%

@dataclass
class BinanceFees:
    maker_fee: float = 0.0002          # 0.02% (standard VIP 0 Binance Futures)
    taker_fee: float = 0.0005          # 0.05%
    slippage: float = 0.0002           # Estimated slippage 0.02%

@dataclass
class StrategyConfig:
    symbol: str = "BTCUSDT"
    trend_timeframe: str = "1h"        # Higher timeframe for macro trend direction
    entry_timeframe: str = "15m"       # Lower timeframe for entry execution
    ema_trend_period: int = 200        # EMA 200 on trend timeframe
    donchian_period: int = 20          # Donchian channel breakout period
    atr_period: int = 14               # ATR period for volatility
    atr_sl_multiplier: float = 1.5     # Stop loss = 1.5 * ATR
    risk_reward_ratio: float = 2.0     # Take Profit = 2.0 * Stop distance (1:2 R:R)
    trailing_stop_activation: float = 1.0 # Move SL to Breakeven when profit >= 1.0R
    trailing_stop_distance: float = 1.0   # Trail behind peak by 1.0 * ATR
    rsi_period: int = 14
    rsi_long_max: float = 70.0         # Avoid buying if RSI > 70
    rsi_short_min: float = 30.0        # Avoid selling if RSI < 30
    volume_factor: float = 1.1         # Volume must exceed 1.1x of 20-period SMA volume
    adx_period: int = 14               # ADX trend strength period
    adx_min: float = 20.0              # Minimum ADX to avoid ranging chop
    use_adx_filter: bool = True        # Enable/disable ADX filter

@dataclass
class SystemConfig:
    binance_fapi_url: str = "https://fapi.binance.com"
    binance_ws_url: str = "wss://fstream.binance.com/ws"
    log_level: str = "INFO"

# Default instances
DEFAULT_RISK = RiskConfig()
DEFAULT_FEES = BinanceFees()
DEFAULT_STRATEGY = StrategyConfig()
DEFAULT_SYSTEM = SystemConfig()
