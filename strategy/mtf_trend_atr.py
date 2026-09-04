"""
Multi-Timeframe Trend Following & Volatility Breakout (MTF-Trend-ATR)
- Macro Trend (1h / 4h): 200 EMA Filter (Long only if price > EMA200, Short only if price < EMA200)
- Execution (15m): Donchian Breakout + Volume Surge + RSI Filter
- Risk: Dynamic ATR Stop-Loss and Take-Profit
"""
from typing import Dict
import numpy as np
import pandas as pd

from config.settings import StrategyConfig, DEFAULT_STRATEGY
from strategy.base import BaseStrategy
from strategy.indicators import (
    calculate_ema,
    calculate_sma,
    calculate_atr,
    calculate_rsi,
    calculate_donchian_channel,
    calculate_adx,
)


class MTFTrendATRStrategy(BaseStrategy):
    def __init__(self, config: StrategyConfig = DEFAULT_STRATEGY):
        super().__init__(name="MTF_Trend_ATR_Futures")
        self.config = config

    def generate_signals(self, data_map: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        tf_trend = self.config.trend_timeframe
        tf_entry = self.config.entry_timeframe

        if tf_trend not in data_map or tf_entry not in data_map:
            raise KeyError(f"data_map must contain both '{tf_trend}' and '{tf_entry}' timeframes.")

        df_trend = data_map[tf_trend].copy()
        df_entry = data_map[tf_entry].copy()

        # 1. Macro Trend Calculation on Higher Timeframe
        df_trend["ema_trend"] = calculate_ema(df_trend["close"], self.config.ema_trend_period)
        # Shift higher timeframe by 1 to ensure zero lookahead bias before aligning to entry timeframe
        df_trend["macro_bullish"] = (df_trend["close"] > df_trend["ema_trend"]).astype(int)
        df_trend_shifted = df_trend[["ema_trend", "macro_bullish"]].shift(1)

        # 2. Align Higher Timeframe to Lower Timeframe (Forward Fill)
        # We reindex to entry timeframe timestamps and forward fill
        aligned_trend = df_trend_shifted.reindex(df_entry.index, method="ffill")
        df_entry["ema_trend_macro"] = aligned_trend["ema_trend"]
        df_entry["macro_bullish"] = aligned_trend["macro_bullish"]

        # 3. Indicators on Entry Timeframe
        df_entry["atr"] = calculate_atr(df_entry["high"], df_entry["low"], df_entry["close"], self.config.atr_period)
        df_entry["rsi"] = calculate_rsi(df_entry["close"], self.config.rsi_period)
        df_entry["donchian_high"], df_entry["donchian_low"], _ = calculate_donchian_channel(
            df_entry["high"], df_entry["low"], self.config.donchian_period
        )
        df_entry["volume_sma"] = calculate_sma(df_entry["volume"], 20)
        df_entry["adx"], df_entry["plus_di"], df_entry["minus_di"] = calculate_adx(
            df_entry["high"], df_entry["low"], df_entry["close"], self.config.adx_period
        )

        # 4. Entry Signals
        # Breakout condition: current close breaks previous N-candle high/low
        long_breakout = df_entry["close"] > df_entry["donchian_high"]
        short_breakout = df_entry["close"] < df_entry["donchian_low"]

        # Volume condition: volume exceeds average
        volume_ok = df_entry["volume"] > (df_entry["volume_sma"] * self.config.volume_factor)

        # ADX trend strength filter
        if self.config.use_adx_filter:
            trend_strength_ok = df_entry["adx"] >= self.config.adx_min
        else:
            trend_strength_ok = True

        # RSI conditions
        rsi_long_ok = (df_entry["rsi"] > 50) & (df_entry["rsi"] < self.config.rsi_long_max)
        rsi_short_ok = (df_entry["rsi"] < 50) & (df_entry["rsi"] > self.config.rsi_short_min)

        # Macro filters
        macro_long = df_entry["macro_bullish"] == 1
        macro_short = df_entry["macro_bullish"] == 0

        # Raw Signals
        long_cond = long_breakout & volume_ok & rsi_long_ok & macro_long & trend_strength_ok
        short_cond = short_breakout & volume_ok & rsi_short_ok & macro_short & trend_strength_ok

        df_entry["signal"] = 0
        df_entry.loc[long_cond, "signal"] = 1
        df_entry.loc[short_cond, "signal"] = -1

        # Calculate initial Stop Loss and Take Profit levels
        df_entry["stop_loss"] = np.nan
        df_entry["take_profit"] = np.nan

        # Long SL/TP
        long_sl = df_entry["close"] - (self.config.atr_sl_multiplier * df_entry["atr"])
        long_tp = df_entry["close"] + (self.config.atr_sl_multiplier * self.config.risk_reward_ratio * df_entry["atr"])
        df_entry.loc[long_cond, "stop_loss"] = long_sl
        df_entry.loc[long_cond, "take_profit"] = long_tp

        # Short SL/TP
        short_sl = df_entry["close"] + (self.config.atr_sl_multiplier * df_entry["atr"])
        short_tp = df_entry["close"] - (self.config.atr_sl_multiplier * self.config.risk_reward_ratio * df_entry["atr"])
        df_entry.loc[short_cond, "stop_loss"] = short_sl
        df_entry.loc[short_cond, "take_profit"] = short_tp

        # Clean NaN values from warm-up period
        df_entry.dropna(subset=["atr", "donchian_high", "donchian_low", "ema_trend_macro"], inplace=True)
        return df_entry
