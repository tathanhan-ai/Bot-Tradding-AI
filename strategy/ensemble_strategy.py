"""
AI Multi-Strategy Ensemble Hub (Hệ thống Ma trận Đa chiến lược do AI điều phối)
Coordinates 4 specialized quant strategies simultaneously:
1. Multi-Timeframe Trend Following (Theo sóng EMA/MACD)
2. Mean Reversion / Bollinger Bands Squeeze (Bắt đảo chiều RSI & dải BB)
3. Liquidity Sweep & Volume Surge (Săn rút chân râu nến & bẫy thanh khoản cá mập)
4. Futures Grid Trading (Khai thác vùng tích lũy Sideway)

The Antigravity AI Brain dynamically weights each strategy based on current market regime.
"""
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


@dataclass
class StrategyVote:
    name: str                   # Tên chiến lược
    direction: int              # 1 (Long), -1 (Short), 0 (Neutral)
    score: float                # -100 to +100
    weight: float               # 0.0 to 1.0 (Tổng các weight = 1.0)
    rationale: str              # Lý giải chi tiết


@dataclass
class EnsembleResult:
    consensus_score: float      # -100 to +100
    consensus_direction: int    # 1 (Long), -1 (Short), 0 (Neutral / Grid)
    consensus_verdict: str      # 'STRONG_LONG', 'STRONG_SHORT', 'SIDEWAY_GRID', 'STAND_ASIDE'
    confidence: int             # 0 - 100%
    votes: List[StrategyVote]
    active_mode: str            # 'TREND_DOMINANT', 'MEAN_REVERSION_GRID', 'LIQUIDITY_HUNT'
    rationale: str


class TrendFollowingSubEngine:
    def evaluate(
        self,
        df_15m: Optional[pd.DataFrame] = None,
        df_1h: Optional[pd.DataFrame] = None,
        current_price: float = 0.0,
        df_1w: Optional[pd.DataFrame] = None,
        df_1M: Optional[pd.DataFrame] = None,
        base_tf: str = "15m",
        htf_tf: str = "1h",
        **kwargs
    ) -> StrategyVote:
        df_base = df_15m if df_15m is not None else kwargs.get("df_base")
        df_htf = df_1h if df_1h is not None else kwargs.get("df_htf")
        if df_base is None or len(df_base) < 30:
            return StrategyVote(f"Xu Hướng ({base_tf.upper()})", 0, 0.0, 0.3, "Chưa đủ dữ liệu nến.")

        close = df_base["close"]
        ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
        ema50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])
        macro_ema100 = float(df_htf["close"].ewm(span=100, adjust=False).mean().iloc[-1]) if (df_htf is not None and len(df_htf) >= 50) else ema50

        # Secular / Macro Weekly & Monthly EMA calculation
        weekly_ema20 = None
        if df_1w is not None and len(df_1w) >= 15:
            weekly_ema20 = float(df_1w["close"].ewm(span=20, adjust=False).mean().iloc[-1])

        monthly_ema10 = None
        if df_1M is not None and len(df_1M) >= 8:
            monthly_ema10 = float(df_1M["close"].ewm(span=10, adjust=False).mean().iloc[-1])

        # MACD (12, 26, 9) on base timeframe
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist = float((macd_line - signal_line).iloc[-1])

        # Directional alignment
        bullish_alignment = (current_price > ema20 > ema50) and (current_price > macro_ema100)
        bearish_alignment = (current_price < ema20 < ema50) and (current_price < macro_ema100)

        # Early momentum detection
        above_ema20 = current_price > ema20
        below_ema20 = current_price < ema20

        # Macro Alignment Evaluation
        macro_bull = True if (weekly_ema20 is None or current_price >= weekly_ema20) else False
        macro_bear = True if (weekly_ema20 is None or current_price <= weekly_ema20) else False
        if monthly_ema10 is not None:
            macro_bull = macro_bull and (current_price >= monthly_ema10)
            macro_bear = macro_bear and (current_price <= monthly_ema10)

        macro_suffix = ""
        if weekly_ema20 is not None:
            macro_suffix = f" | W1-EMA20: ${weekly_ema20:,.0f}"

        if bullish_alignment:
            if macro_bull:
                # Strong Secular Bull Tailwind
                score = round(min(100.0, 85.0 + (15.0 if macd_hist > 0 else 0.0)), 1)
                return StrategyVote(
                    f"Xu Hướng ({base_tf.upper()})", 1, score, 0.40,
                    f"🚀 Đại Chu Kỳ Tăng: Giá (${current_price:,.1f}) trên cụm EMA20/50/{htf_tf.upper()} & thuận sóng Tuần{macro_suffix}, MACD histogram ({macd_hist:+.1f}) bùng nổ đà tăng."
                )
            else:
                # Counter-trend Relief Rally under Macro Resistance
                score = round(min(65.0, 50.0 + (10.0 if macd_hist > 0 else 0.0)), 1)
                return StrategyVote(
                    f"Xu Hướng ({base_tf.upper()})", 1, score, 0.30,
                    f"⚠️ Sóng Hồi Ngắn Hạn: Giá trên EMA20/50 ({base_tf.upper()}) nhưng dưới cản vĩ mô Tuần{macro_suffix} -> Ưu tiên Take Profit nhanh."
                )
        elif bearish_alignment:
            if macro_bear:
                # Secular Bear Market Dominance
                score = round(max(-100.0, -85.0 - (15.0 if macd_hist < 0 else 0.0)), 1)
                return StrategyVote(
                    f"Xu Hướng ({base_tf.upper()})", -1, score, 0.40,
                    f"🔻 Xu Hướng Giảm Sâu: Giá thủng hoàn toàn EMA20/50/{htf_tf.upper()} & thuận sóng giảm vĩ mô{macro_suffix}, MACD ({macd_hist:+.1f}) chịu áp lực xả mạnh."
                )
            else:
                score = round(max(-65.0, -50.0 - (10.0 if macd_hist < 0 else 0.0)), 1)
                return StrategyVote(
                    f"Xu Hướng ({base_tf.upper()})", -1, score, 0.30,
                    f"⚠️ Nhịp Rũ Ngắn Hạn: Giá dưới EMA20/50 ({base_tf.upper()}) trong xu hướng vĩ mô tăng -> Cẩn trọng bẫy gấu (Bear Trap)."
                )
        elif above_ema20 and macd_hist > 0:
            score = round(min(65.0, 35.0 + (macd_hist * 0.5)), 1)
            return StrategyVote(
                f"Xu Hướng ({base_tf.upper()})", 1, score, 0.25,
                f"Chớm Tăng Hồi: Giá vượt EMA20 (${ema20:,.1f}), MACD phân kỳ dương ({macd_hist:+.1f}){macro_suffix} -> Đang hình thành sóng tăng."
            )
        elif below_ema20 and macd_hist < 0:
            score = round(max(-65.0, -35.0 + (macd_hist * 0.5)), 1)
            return StrategyVote(
                f"Xu Hướng ({base_tf.upper()})", -1, score, 0.25,
                f"Chớm Giảm Điều Chỉnh: Giá thủng EMA20 (${ema20:,.1f}), MACD phân kỳ âm ({macd_hist:+.1f}){macro_suffix} -> Đang chịu áp lực giảm."
            )
        else:
            neutral_score = round(max(-15.0, min(15.0, (current_price - ema20) / max(ema20, 1.0) * 1000.0)), 1)
            dir_val = 1 if neutral_score > 5 else (-1 if neutral_score < -5 else 0)
            return StrategyVote(
                f"Xu Hướng ({base_tf.upper()})", dir_val, neutral_score, 0.15,
                f"Tích Lũy Đi Ngang: EMA20 (${ema20:,.1f}) & EMA50 (${ema50:,.1f}) co cụm, MACD phẳng ({macd_hist:+.1f}){macro_suffix} -> Thị trường nén biên."
            )


class MeanReversionSubEngine:
    def evaluate(self, df_15m: Optional[pd.DataFrame] = None, current_price: float = 0.0, base_tf: str = "15m", **kwargs) -> StrategyVote:
        df_base = df_15m if df_15m is not None else kwargs.get("df_base")
        if df_base is None or len(df_base) < 30:
            return StrategyVote(f"Bắt Đảo Chiều ({base_tf.upper()})", 0, 0.0, 0.25, "Đang nạp nến...")

        close = df_base["close"]
        bb_mid = float(close.rolling(20).mean().iloc[-1])
        bb_std = float(close.rolling(20).std().iloc[-1])
        bb_upper = bb_mid + 2.0 * bb_std
        bb_lower = bb_mid - 2.0 * bb_std
        bb_band_width = max(1e-6, bb_upper - bb_lower)

        delta = close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean().iloc[-1]
        loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean().iloc[-1]
        rsi = float(100 - (100 / (1 + (gain / (loss if loss > 0 else 1e-6)))))

        eff_price = float(current_price) if current_price > 0 else float(close.iloc[-1])
        # Continuous Bollinger %B: 0.0 = Lower band, 0.5 = Middle band, 1.0 = Upper band
        pct_b = (eff_price - bb_lower) / bb_band_width

        # Real-time continuous score (-100 to +100):
        # When %B < 0.5 and RSI < 50, price is undervalued -> positive score (pulls towards Long)
        # When %B > 0.5 and RSI > 50, price is overextended -> negative score (pulls towards Short)
        base_score = (0.5 - pct_b) * 120.0 + (50.0 - rsi) * 0.8
        score = round(max(-100.0, min(100.0, base_score)), 1)

        # Extreme thresholds overrides
        if eff_price <= bb_lower or (rsi <= 30.0 and eff_price <= bb_mid):
            score = max(score, 75.0)
            direction = 1
            rationale = f"Giá chạm cận dưới Bollinger Bands (%B={pct_b*100:.1f}%), RSI={rsi:.1f} quá bán cực đại $\rightarrow$ Kỳ vọng hồi phục về trục giữa."
        elif eff_price >= bb_upper or (rsi >= 70.0 and eff_price >= bb_mid):
            score = min(score, -75.0)
            direction = -1
            rationale = f"Giá chạm dải trên Bollinger Bands (%B={pct_b*100:.1f}%), RSI={rsi:.1f} quá mua cực đại $\rightarrow$ Kỳ vọng điều chỉnh về trục giữa."
        else:
            direction = 1 if score >= 15.0 else (-1 if score <= -15.0 else 0)
            bias_text = "Hồi phục Long" if score > 5.0 else ("Điều chỉnh Short" if score < -5.0 else "Cân bằng BB")
            rationale = f"Bollinger %B={pct_b*100:.1f}%, RSI={rsi:.1f} -> {bias_text} (Score: {score:+.1f})."

        return StrategyVote(f"Bắt Đảo Chiều ({base_tf.upper()})", direction, score, 0.25, rationale)


class LiquiditySweepSubEngine:
    def evaluate(self, df_15m: Optional[pd.DataFrame] = None, current_price: float = 0.0, base_tf: str = "15m", **kwargs) -> StrategyVote:
        df_base = df_15m if df_15m is not None else kwargs.get("df_base")
        if df_base is None or len(df_base) < 20:
            return StrategyVote(f"Săn Rút Chân ({base_tf.upper()})", 0, 0.0, 0.2, "Đang nạp nến...")

        last_candle = df_base.iloc[-1]
        c_open = float(last_candle["open"])
        c_high = float(last_candle["high"])
        c_low = float(last_candle["low"])
        c_close = float(last_candle["close"])
        eff_price = float(current_price) if current_price > 0 else c_close

        # Dynamic high/low including real-time tick
        eff_high = max(c_high, eff_price)
        eff_low = min(c_low, eff_price)
        candle_range = max(1.0, eff_high - eff_low)

        lower_wick = min(c_open, eff_price) - eff_low
        upper_wick = eff_high - max(c_open, eff_price)
        vol_ratio = float((last_candle["volume"] / df_base["volume"].iloc[-10:].mean())) if len(df_base) >= 10 else 1.0

        # Check recent 5-candle high/low liquidity sweep
        recent_high = float(np.max(df_base["high"].iloc[-6:-1])) if len(df_base) >= 6 else c_high
        recent_low = float(np.min(df_base["low"].iloc[-6:-1])) if len(df_base) >= 6 else c_low
        is_sweep_high = (eff_high >= recent_high and eff_price < recent_high)
        is_sweep_low = (eff_low <= recent_low and eff_price > recent_low)

        lower_ratio = lower_wick / candle_range
        upper_ratio = upper_wick / candle_range

        # Bullish Pinbar / Sweep low
        if (lower_ratio >= 0.55 and vol_ratio >= 1.4) or (is_sweep_low and lower_ratio >= 0.4):
            return StrategyVote(f"Săn Rút Chân ({base_tf.upper()})", 1, 90.0, 0.25, f"Xuất hiện nến rút chân bẫy gấu (Râu {lower_ratio*100:.0f}%, Vol x{vol_ratio:.1f}), cá mập quét thanh khoản đáy rồi mua thốc lên.")
        # Bearish Shooting Star / Sweep high
        elif (upper_ratio >= 0.55 and vol_ratio >= 1.4) or (is_sweep_high and upper_ratio >= 0.4):
            return StrategyVote(f"Săn Rút Chân ({base_tf.upper()})", -1, -90.0, 0.25, f"Xuất hiện nến rút râu trên (Râu {upper_ratio*100:.0f}%, Vol x{vol_ratio:.1f}), cá mập xả hàng từ chối giá cao.")
        else:
            # Continuous absorption gradient based on real-time wick asymmetry
            wick_bias = (lower_wick - upper_wick) / candle_range
            score = round(max(-60.0, min(60.0, wick_bias * 80.0)), 1)
            direction = 1 if score >= 15.0 else (-1 if score <= -15.0 else 0)
            bias_str = "Hấp thụ Mua (Rút râu dưới)" if score > 5 else ("Áp lực Xả (Rút râu trên)" if score < -5 else "Cân bằng")
            return StrategyVote(
                f"Săn Rút Chân ({base_tf.upper()})",
                direction,
                score,
                0.15,
                f"Râu dưới {lower_ratio*100:.0f}%, Râu trên {upper_ratio*100:.0f}% -> {bias_str} (Score: {score:+.1f})."
            )


from strategy.multi_candle_patterns import MultiTimeframeCandleStrategyEngine, MTFCandleConfluenceResult


class MultiCandlePatternSubEngine:
    def __init__(self):
        self.engine = MultiTimeframeCandleStrategyEngine()

    def evaluate(self, data_map: Dict[str, pd.DataFrame], current_price: float, active_timeframe: str = "15m") -> Tuple[StrategyVote, MTFCandleConfluenceResult]:
        res = self.engine.evaluate(data_map, current_price, active_timeframe=active_timeframe)
        dir_val = 1 if res.confluence_score >= 25.0 else (-1 if res.confluence_score <= -25.0 else 0)
        vote = StrategyVote(
            name=f"Mẫu Hình Nến Đa Khung ({active_timeframe.upper()} Focus)",
            direction=dir_val,
            score=res.confluence_score,
            weight=0.25,
            rationale=res.summary_rationale
        )
        return vote, res


class EnsembleCoordinator:
    def __init__(self):
        self.trend_engine = TrendFollowingSubEngine()
        self.reversion_engine = MeanReversionSubEngine()
        self.sweep_engine = LiquiditySweepSubEngine()
        self.candle_engine = MultiCandlePatternSubEngine()
        self.last_candle_confluence: Optional[MTFCandleConfluenceResult] = None

    def evaluate_ensemble(
        self,
        data_map: Dict[str, pd.DataFrame],
        current_price: float,
        market_regime: str,
        active_timeframe: str = "15m"
    ) -> EnsembleResult:
        htf_map = {
            "1m": "15m", "3m": "15m", "5m": "1h", "15m": "1h",
            "30m": "1h", "1h": "4h", "4h": "1d", "1d": "1w", "1w": "1M", "1M": "1M"
        }
        htf_tf = htf_map.get(active_timeframe, "1h")

        df_base = data_map.get(active_timeframe)
        if df_base is None or df_base.empty:
            df_base = data_map.get("15m")
            active_timeframe = "15m"

        df_htf = data_map.get(htf_tf)
        if df_htf is None or df_htf.empty:
            df_htf = data_map.get("1h")
            if df_htf is None or df_htf.empty:
                df_htf = df_base

        df_1w = data_map.get("1w")
        df_1M = data_map.get("1M")

        v_trend = self.trend_engine.evaluate(df_base, df_htf, current_price, df_1w=df_1w, df_1M=df_1M, base_tf=active_timeframe, htf_tf=htf_tf)
        v_reversion = self.reversion_engine.evaluate(df_base, current_price, base_tf=active_timeframe)
        v_sweep = self.sweep_engine.evaluate(df_base, current_price, base_tf=active_timeframe)
        v_candle, candle_res = self.candle_engine.evaluate(data_map, current_price, active_timeframe=active_timeframe)
        self.last_candle_confluence = candle_res

        # Dynamic Weight Rebalancing across all 4 quant strategies (Total weights = 1.0)
        if market_regime in ("TRENDING_BULL", "TRENDING_BEAR"):
            v_trend.weight = 0.40
            v_candle.weight = 0.30
            v_sweep.weight = 0.20
            v_reversion.weight = 0.10
            mode = "TREND_DOMINANT 🚀 (Sóng Xu Hướng & Nến Tiếp Diễn 70%)"
        elif market_regime == "RANGING_SIDEWAY":
            v_reversion.weight = 0.35
            v_candle.weight = 0.30
            v_sweep.weight = 0.25
            v_trend.weight = 0.10
            mode = "MEAN_REVERSION_GRID ⚖️ (Bắt Đỉnh Đáy & Mẫu Hình Đảo Chiều)"
        elif market_regime == "VOLATILE_PANIC":
            v_sweep.weight = 0.40
            v_candle.weight = 0.30
            v_trend.weight = 0.15
            v_reversion.weight = 0.15
            mode = "LIQUIDITY_HUNT 🐋 (Săn Quét Rút Chân Cá Mập 70%)"
        else:
            v_trend.weight = 0.25
            v_candle.weight = 0.25
            v_reversion.weight = 0.25
            v_sweep.weight = 0.25
            mode = "BALANCED_ENSEMBLE"

        # Adaptive Timeframe Fine-Tuning
        if active_timeframe in ("1m", "3m"):
            v_sweep.weight += 0.05
            v_candle.weight += 0.05
            v_trend.weight = max(0.05, v_trend.weight - 0.10)
        elif active_timeframe in ("4h", "1d", "1w"):
            v_trend.weight += 0.10
            v_sweep.weight = max(0.05, v_sweep.weight - 0.05)
            v_reversion.weight = max(0.05, v_reversion.weight - 0.05)

        # Calculate Weighted Consensus Score (-100 to +100)
        consensus_score = (
            (v_trend.score * v_trend.weight) +
            (v_candle.score * v_candle.weight) +
            (v_reversion.score * v_reversion.weight) +
            (v_sweep.score * v_sweep.weight)
        )
        consensus_score = round(max(-100.0, min(100.0, consensus_score)), 1)

        if consensus_score >= 45.0:
            direction = 1
            verdict = "STRONG_LONG"
            confidence = int(min(98, 60 + (consensus_score * 0.4)))
            rationale = f"Đồng thuận Đa Chiến Lược & Đa Khung Nến: {consensus_score:+.1f} điểm LONG. {candle_res.confluence_verdict}."
        elif consensus_score <= -45.0:
            direction = -1
            verdict = "STRONG_SHORT"
            confidence = int(min(98, 60 + (abs(consensus_score) * 0.4)))
            rationale = f"Đồng thuận Đa Chiến Lược & Đa Khung Nến: {consensus_score:+.1f} điểm SHORT. {candle_res.confluence_verdict}."
        else:
            direction = 0
            verdict = "SIDEWAY_GRID"
            confidence = 88
            rationale = f"Điểm số đa chiến lược ở mức trung tính ({consensus_score:+.1f} điểm). AI điều phối kích hoạt Bot Lưới cân đối 10 tầng để thu gom lợi nhuận sideway."

        return EnsembleResult(
            consensus_score=consensus_score,
            consensus_direction=direction,
            consensus_verdict=verdict,
            confidence=confidence,
            votes=[v_trend, v_reversion, v_sweep, v_candle],
            active_mode=mode,
            rationale=rationale
        )
