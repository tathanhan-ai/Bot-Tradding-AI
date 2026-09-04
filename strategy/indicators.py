"""
Technical Indicators Module for Quantitative Trading
Pure vectorized implementations using pandas and numpy for high performance.
"""
import numpy as np
import pandas as pd


def calculate_ema(series: pd.Series, period: int = 200) -> pd.Series:
    """Exponential Moving Average (EMA)"""
    return series.ewm(span=period, adjust=False).mean()


def calculate_sma(series: pd.Series, period: int = 20) -> pd.Series:
    """Simple Moving Average (SMA)"""
    return series.rolling(window=period).mean()


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range (ATR) with Wilder's Smoothing"""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Wilder's smoothing (alpha = 1 / period)
    atr = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    return atr


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (RSI)"""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def calculate_donchian_channel(high: pd.Series, low: pd.Series, period: int = 20) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Donchian Channel: Upper Band, Lower Band, Middle Band
    We shift by 1 to avoid lookahead bias when testing breakout of previous N candles.
    """
    upper = high.shift(1).rolling(window=period).max()
    lower = low.shift(1).rolling(window=period).min()
    middle = (upper + lower) / 2.0
    return upper, lower, middle


def calculate_bollinger_bands(close: pd.Series, period: int = 20, num_std: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands: Middle, Upper, Lower"""
    middle = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    return upper, middle, lower


def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Average Directional Index (ADX) with +DI and -DI
    ADX > 20/25 indicates a strong trend. ADX < 20 indicates choppy range (avoid breakout).
    """
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    plus_dm = (high - prev_high).clip(lower=0)
    minus_dm = (prev_low - low).clip(lower=0)

    # When +DM < -DM, +DM is 0; when -DM < +DM, -DM is 0
    plus_dm[plus_dm < minus_dm] = 0.0
    minus_dm[minus_dm < plus_dm] = 0.0

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Wilder's Smoothing
    alpha = 1.0 / period
    atr_val = tr.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    plus_di = 100.0 * (plus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean() / atr_val.replace(0, np.nan))
    minus_di = 100.0 * (minus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean() / atr_val.replace(0, np.nan))

    dx = 100.0 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx = dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    return adx.fillna(0.0), plus_di.fillna(0.0), minus_di.fillna(0.0)
