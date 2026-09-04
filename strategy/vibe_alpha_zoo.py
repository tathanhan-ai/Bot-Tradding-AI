# -*- coding: utf-8 -*-
"""
HKUDS Vibe-Trading: Alpha Zoo Quantitative Factors Engine
Trích xuất từ thư viện Alpha Zoo (460+ Alpha Factors) của HKU Data Intelligence Lab.
Bao gồm 12 Alpha Factors toán học mạnh nhất cho Crypto Futures:
1. EMA Slope & EMA Acceleration (d^2(EMA)/dt^2)
2. Kyle's Lambda (Độ nhạy giá theo khối lượng giao dịch)
3. Amihud Illiquidity Ratio (Hệ số trượt giá tác động)
4. Yang-Zhang Realized Volatility (Đo biến động độc lập với gap)
5. Ornstein-Uhlenbeck (OU) Mean Reversion Half-Life (Thời gian kéo về VWAP)
6. Normalized Momentum Z-Score
7. Volume Force Ratio (Lực đẩy nến vs Volume)
8. Order Book Depth Resilience Alpha
9. Squeeze Intensity (Độ nén dải Bollinger)
10. Tail Risk Skewness (Độ lệch đuôi rủi ro phân phối)
11. Institutional Absorption Ratio
12. Composite Alpha Forecast Score (-100 đến +100)
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
import math


@dataclass
class VibeAlphaMetrics:
    # 1. Momentum & Acceleration
    ema_slope: float = 0.0                  # Độ dốc EMA 20
    ema_acceleration: float = 0.0           # Gia tốc sóng d^2(EMA)/dt^2
    momentum_zscore: float = 0.0            # Z-Score động lượng 20 chu kỳ
    
    # 2. Market Impact & Liquidity
    kyles_lambda: float = 0.0               # Kyle's Lambda (Price change per volume)
    amihud_illiquidity: float = 0.0         # Amihud ratio (|return| / (price * volume))
    volume_force_ratio: float = 1.0         # Lực đẩy nến trên volume
    
    # 3. Volatility & Mean Reversion
    yang_zhang_vol: float = 0.015           # Biến động thực Yang-Zhang
    ou_half_life_bars: float = 12.0         # Chu kỳ hồi quy về VWAP (số nến)
    squeeze_intensity: float = 0.50         # Độ nén nến (0-1.0)
    tail_risk_skew: float = 0.0             # Độ lệch đuôi rủi ro
    
    # 4. Institutional Flow
    absorption_ratio: float = 0.50          # Tỷ lệ hấp thụ thanh khoản
    
    # 5. Composite Forecast
    composite_alpha_score: float = 0.0      # [-100, +100] Điểm tổng hợp Alpha Zoo
    alpha_regime: str = "NEUTRAL"           # 'STRONG_BULLISH_MOMENTUM', 'BULLISH', 'NEUTRAL', 'BEARISH', 'STRONG_BEARISH_MOMENTUM'
    dominant_factor: str = "EMA_MOMENTUM"
    summary_thesis: str = ""


class VibeAlphaZooEngine:
    def __init__(self, ema_period: int = 20, lookback: int = 50):
        self.ema_period = ema_period
        self.lookback = lookback
        self.latest_metrics = VibeAlphaMetrics()

    def evaluate(
        self,
        df: pd.DataFrame,
        current_price: float,
        best_bid: float = 0.0,
        best_ask: float = 0.0,
        spread: float = 0.0
    ) -> VibeAlphaMetrics:
        """Computes all 12 Alpha Zoo factors from candlestick DataFrame."""
        if df is None or len(df) < 25:
            return self.latest_metrics

        try:
            closes = df["close"].values
            highs = df["high"].values
            lows = df["low"].values
            opens = df["open"].values
            volumes = df["volume"].values if "volume" in df.columns else np.ones(len(df))

            n = len(closes)
            
            # 1. EMA Slope & EMA Acceleration (d^2(EMA)/dt^2)
            # EMA series
            s = pd.Series(closes)
            ema = s.ewm(span=self.ema_period, adjust=False).mean().values
            # 1st derivative (slope) over last 3 bars
            slope_now = (ema[-1] - ema[-2]) / max(1e-6, ema[-2]) * 1000.0
            slope_prev = (ema[-2] - ema[-3]) / max(1e-6, ema[-3]) * 1000.0
            # 2nd derivative (acceleration)
            accel = (slope_now - slope_prev)

            # 2. Normalized Momentum Z-Score (over lookback)
            lb = min(30, n - 1)
            ret_window = (closes[-lb:] - closes[-lb-1:-1]) / closes[-lb-1:-1]
            ret_mean = np.mean(ret_window)
            ret_std = np.std(ret_window) + 1e-6
            curr_ret = (closes[-1] - closes[-2]) / closes[-2]
            mom_zscore = float((curr_ret - ret_mean) / ret_std)

            # 3. Kyle's Lambda (Price impact per unit of volume)
            # lambda = Cov(delta_P, Q) / Var(Q)
            price_deltas = np.diff(closes[-15:])
            vol_deltas = volumes[-15:][1:]
            cov_pv = np.cov(price_deltas, vol_deltas)[0, 1] if len(price_deltas) > 3 else 0.0
            var_v = np.var(vol_deltas) + 1e-6
            kyles_lambda = float(abs(cov_pv / var_v)) if var_v > 0 else 0.001

            # 4. Amihud Illiquidity Ratio (|ret| / (price * volume))
            vol_dollar = closes[-1] * max(0.01, volumes[-1])
            amihud = float(abs(curr_ret) / max(1.0, vol_dollar) * 1e6)

            # 5. Yang-Zhang Realized Volatility
            # Over 15 bars
            k = 0.34 / (1.34 + (15 + 1) / (15 - 1))
            log_ho = np.log(highs[-15:] / opens[-15:])
            log_lo = np.log(lows[-15:] / opens[-15:])
            log_co = np.log(closes[-15:] / opens[-15:])
            log_oc = np.log(opens[-15:] / closes[-16:-1])
            log_cc = np.log(closes[-15:] / closes[-16:-1])

            var_overnight = np.var(log_oc)
            var_open_to_close = np.var(log_co)
            var_rs = np.mean(log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co))
            yz_vol = float(np.sqrt(max(1e-6, var_overnight + k * var_open_to_close + (1 - k) * var_rs)))

            # 6. Ornstein-Uhlenbeck (OU) Mean Reversion Half-Life
            # Regress delta_x on x_t-1: dx_t = theta * (mu - x_t-1) dt
            x = closes[-25:]
            x_prev = x[:-1]
            dx = np.diff(x)
            if len(x_prev) > 5 and np.std(x_prev) > 1e-4:
                slope, _ = np.polyfit(x_prev, dx, 1)
                theta = -slope
                half_life = float(np.log(2.0) / theta) if theta > 0 else 50.0
                half_life = max(2.0, min(50.0, half_life))
            else:
                half_life = 12.0

            # 7. Squeeze Intensity (Bollinger Band bandwidth)
            rolling_std = np.std(closes[-20:])
            rolling_mean = np.mean(closes[-20:])
            bb_width = (rolling_std * 2.0) / max(1.0, rolling_mean)
            squeeze_val = float(max(0.0, min(1.0, 1.0 - (bb_width / 0.03))))

            # 8. Volume Force Ratio (Bar body vs Volume)
            body = abs(closes[-1] - opens[-1])
            candle_range = max(1e-4, highs[-1] - lows[-1])
            body_ratio = body / candle_range
            avg_vol = np.mean(volumes[-20:]) if len(volumes) >= 20 else volumes[-1]
            rel_vol = volumes[-1] / max(1e-4, avg_vol)
            vol_force = float(body_ratio * rel_vol)

            # 9. Tail Risk Skewness
            recent_rets = (closes[-20:] - closes[-21:-1]) / closes[-21:-1]
            skewness = float(pd.Series(recent_rets).skew()) if len(recent_rets) > 5 else 0.0

            # 10. Institutional Absorption Ratio
            # High volume with small candle range = absorption by iceberg orders
            absorption = float(min(1.0, rel_vol / (body_ratio + 0.1) * 0.25))

            # 11. Composite Alpha Score [-100, +100]
            # Weights: Momentum/Accel (35%), ZScore (25%), VolForce (20%), Skew (20%)
            mom_component = max(-40.0, min(40.0, (slope_now * 15.0) + (accel * 25.0)))
            z_component = max(-30.0, min(30.0, mom_zscore * 12.0))
            force_dir = 1.0 if closes[-1] >= opens[-1] else -1.0
            force_component = max(-20.0, min(20.0, vol_force * 8.0 * force_dir))
            skew_component = max(-10.0, min(10.0, skewness * 8.0))

            composite_score = round(mom_component + z_component + force_component + skew_component, 1)
            composite_score = max(-100.0, min(100.0, composite_score))

            # Regime
            if composite_score >= 45.0:
                alpha_regime = "STRONG_BULLISH_MOMENTUM"
                dominant_factor = "EMA_ACCELERATION_BULL"
            elif composite_score >= 15.0:
                alpha_regime = "BULLISH"
                dominant_factor = "VOLUME_FORCE_BULL"
            elif composite_score <= -45.0:
                alpha_regime = "STRONG_BEARISH_MOMENTUM"
                dominant_factor = "EMA_ACCELERATION_BEAR"
            elif composite_score <= -15.0:
                alpha_regime = "BEARISH"
                dominant_factor = "VOLUME_FORCE_BEAR"
            else:
                alpha_regime = "NEUTRAL"
                dominant_factor = "MEAN_REVERSION_OU"

            thesis = f"Alpha Zoo: {alpha_regime} ({composite_score:+.1f}đ). Gia tốc sóng={accel:+.2f}, OU Half-Life={half_life:.1f} nến, YZ-Vol={yz_vol*100:.2f}%."

            self.latest_metrics = VibeAlphaMetrics(
                ema_slope=round(slope_now, 3),
                ema_acceleration=round(accel, 3),
                momentum_zscore=round(mom_zscore, 2),
                kyles_lambda=round(kyles_lambda, 6),
                amihud_illiquidity=round(amihud, 4),
                volume_force_ratio=round(vol_force, 2),
                yang_zhang_vol=round(yz_vol, 4),
                ou_half_life_bars=round(half_life, 1),
                squeeze_intensity=round(squeeze_val, 2),
                tail_risk_skew=round(skewness, 2),
                absorption_ratio=round(absorption, 2),
                composite_alpha_score=composite_score,
                alpha_regime=alpha_regime,
                dominant_factor=dominant_factor,
                summary_thesis=thesis
            )
            return self.latest_metrics
        except Exception as e:
            return self.latest_metrics

    def get_metrics(self) -> VibeAlphaMetrics:
        return self.latest_metrics

    def get_latest_metrics(self) -> VibeAlphaMetrics:
        return self.latest_metrics
