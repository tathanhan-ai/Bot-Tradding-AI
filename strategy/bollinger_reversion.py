"""
Bollinger Bands + RSI Mean Reversion Strategy (Range & Pullback Hunter)
Specially designed for ranging / choppy / consolidation market regimes where breakouts fail.
- Long: Price touches Lower Bollinger Band + RSI < 35 + Rejection candle
- Short: Price touches Upper Bollinger Band + RSI > 65 + Rejection candle
- Exit: Middle Band (SMA 20) or fixed R:R Take-Profit / ATR Stop-Loss
"""
from typing import Dict
import numpy as np
import pandas as pd

from config.settings import StrategyConfig, DEFAULT_STRATEGY
from strategy.base import BaseStrategy
from strategy.indicators import (
    calculate_ema,
    calculate_atr,
    calculate_rsi,
    calculate_bollinger_bands,
)


class BollingerReversionStrategy(BaseStrategy):
    def __init__(self, config: StrategyConfig = DEFAULT_STRATEGY, bb_period: int = 20, bb_std: float = 2.0):
        super().__init__(name="Bollinger_RSI_MeanReversion")
        self.config = config
        self.bb_period = bb_period
        self.bb_std = bb_std

    def generate_signals(self, data_map: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        tf_entry = self.config.entry_timeframe
        if tf_entry not in data_map:
            raise KeyError(f"data_map must contain '{tf_entry}' timeframe.")

        df = data_map[tf_entry].copy()

        # Indicators
        df["bb_mid"], df["bb_upper"], df["bb_lower"] = calculate_bollinger_bands(
            df["close"], self.bb_period, self.bb_std
        )
        df["rsi"] = calculate_rsi(df["close"], self.config.rsi_period)
        df["atr"] = calculate_atr(df["high"], df["low"], df["close"], self.config.atr_period)
        df["ema_trend"] = calculate_ema(df["close"], 100)

        # Long conditions: Oversold at Lower Band
        long_trigger = (df["low"] <= df["bb_lower"]) & (df["rsi"] < 38) & (df["close"] > df["open"])
        # Short conditions: Overbought at Upper Band
        short_trigger = (df["high"] >= df["bb_upper"]) & (df["rsi"] > 62) & (df["close"] < df["open"])

        df["signal"] = 0
        df.loc[long_trigger, "signal"] = 1
        df.loc[short_trigger, "signal"] = -1

        df["stop_loss"] = np.nan
        df["take_profit"] = np.nan

        # SL and TP calculation
        # Long: SL below low by 1.2 ATR, TP at BB Middle or 1.5R
        long_sl = df["close"] - (1.2 * df["atr"])
        long_tp = df["close"] + (1.8 * df["atr"])
        df.loc[long_trigger, "stop_loss"] = long_sl
        df.loc[long_trigger, "take_profit"] = long_tp

        # Short: SL above high by 1.2 ATR, TP at 1.5R
        short_sl = df["close"] + (1.2 * df["atr"])
        short_tp = df["close"] - (1.8 * df["atr"])
        df.loc[short_trigger, "stop_loss"] = short_sl
        df.loc[short_trigger, "take_profit"] = short_tp

        df.dropna(subset=["bb_upper", "bb_lower", "rsi", "atr"], inplace=True)
        return df
