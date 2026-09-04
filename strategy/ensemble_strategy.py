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
        df_15m: pd.DataFrame,
        df_1h: pd.DataFrame,
        current_price: float,
        df_1w: Optional[pd.DataFrame] = None,
        df_1M: Optional[pd.DataFrame] = None
    ) -> StrategyVote:
        if df_15m is None or len(df_15m) < 30:
            return StrategyVote("Xu Hướng (Trend Following)", 0, 0.0, 0.3, "Chưa đủ dữ liệu nến.")

        close = df_15m["close"]
        ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
        ema50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])
        macro_ema100 = float(df_1h["close"].ewm(span=100, adjust=False).mean().iloc[-1]) if (df_1h is not None and len(df_1h) >= 50) else ema50

        # Secular / Macro Weekly & Monthly EMA calculation
        weekly_ema20 = None
        if df_1w is not None and len(df_1w) >= 15:
            weekly_ema20 = float(df_1w["close"].ewm(span=20, adjust=False).mean().iloc[-1])

        monthly_ema10 = None
        if df_1M is not None and len(df_1M) >= 8:
            monthly_ema10 = float(df_1M["close"].ewm(span=10, adjust=False).mean().iloc[-1])

        # MACD (12, 26, 9) on 15m
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
                    "Xu Hướng (Trend Following)", 1, score, 0.40,
                    f"🚀 Đại Chu Kỳ Tăng: Giá (${current_price:,.1f}) trên cụm EMA20/50/100 & thuận sóng Tuần{macro_suffix}, MACD histogram ({macd_hist:+.1f}) bùng nổ đà tăng."
                )
            else:
                # Counter-trend Relief Rally under Macro Resistance
                score = round(min(65.0, 50.0 + (10.0 if macd_hist > 0 else 0.0)), 1)
                return StrategyVote(
                    "Xu Hướng (Trend Following)", 1, score, 0.28,
                    f"⚠️ Sóng Hồi Tăng (Gặp Cản Tuần): Giá vượt EMA20/50 15m nhưng dưới EMA20 Tuần{macro_suffix}. Khuyến nghị chốt lời từng phần, phòng thủ cản vĩ mô."
                )
        elif bearish_alignment:
            if macro_bear:
                # Strong Secular Bear Headwind
                score = round(max(-100.0, -85.0 - (15.0 if macd_hist < 0 else 0.0)), 1)
                return StrategyVote(
                    "Xu Hướng (Trend Following)", -1, score, 0.40,
                    f"🩸 Đại Chu Kỳ Giảm: Giá (${current_price:,.1f}) dưới cụm EMA20/50/100 & thuận sóng xả Tuần{macro_suffix}, MACD histogram ({macd_hist:+.1f}) xác nhận đà rơi."
                )
            else:
                # Counter-trend Dip in Secular Bull
                score = round(max(-65.0, -50.0 - (10.0 if macd_hist < 0 else 0.0)), 1)
                return StrategyVote(
                    "Xu Hướng (Trend Following)", -1, score, 0.28,
                    f"📉 Nhịp Rũ Ngắn Hạn (Nền Tăng Tuần): Giá giảm thủng EMA 15m nhưng vẫn giữ trên EMA20 Tuần{macro_suffix}. Canh điểm kết thúc nhịp chỉnh để Buy the Dip."
                )
        elif above_ema20 and macd_hist > 0:
            score = round(min(65.0, 35.0 + (macd_hist * 0.5)), 1)
            return StrategyVote(
                "Xu Hướng (Trend Following)", 1, score, 0.25,
                f"Chớm Tăng Hồi: Giá vượt EMA20 (${ema20:,.1f}), MACD phân kỳ dương ({macd_hist:+.1f}){macro_suffix} -> Đang hình thành sóng tăng."
            )
        elif below_ema20 and macd_hist < 0:
            score = round(max(-65.0, -35.0 + (macd_hist * 0.5)), 1)
            return StrategyVote(
                "Xu Hướng (Trend Following)", -1, score, 0.25,
                f"Chớm Giảm Điều Chỉnh: Giá thủng EMA20 (${ema20:,.1f}), MACD phân kỳ âm ({macd_hist:+.1f}){macro_suffix} -> Đang chịu áp lực giảm."
            )
        else:
            neutral_score = round(max(-15.0, min(15.0, (current_price - ema20) / max(ema20, 1.0) * 1000.0)), 1)
            dir_val = 1 if neutral_score > 5 else (-1 if neutral_score < -5 else 0)
            return StrategyVote(
                "Xu Hướng (Trend Following)", dir_val, neutral_score, 0.15,
                f"Tích Lũy Đi Ngang: EMA20 (${ema20:,.1f}) & EMA50 (${ema50:,.1f}) co cụm, MACD phẳng ({macd_hist:+.1f}){macro_suffix} -> Thị trường nén biên."
            )


class MeanReversionSubEngine:
    def evaluate(self, df_15m: pd.DataFrame, current_price: float) -> StrategyVote:
        if df_15m is None or len(df_15m) < 30:
            return StrategyVote("Bắt Đảo Chiều (Mean Reversion)", 0, 0.0, 0.25, "Đang nạp nến...")

        close = df_15m["close"]
        bb_mid = close.rolling(20).mean().iloc[-1]
        bb_std = close.rolling(20).std().iloc[-1]
        bb_upper = bb_mid + 2.0 * bb_std
        bb_lower = bb_mid - 2.0 * bb_std

        delta = close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean().iloc[-1]
        loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean().iloc[-1]
        rsi = 100 - (100 / (1 + (gain / (loss if loss > 0 else 1e-6))))

        # Oversold Snapback (Long opportunity)
        if current_price <= bb_lower or rsi <= 30.0:
            return StrategyVote("Bắt Đảo Chiều (Mean Reversion)", 1, 80.0, 0.35, f"Giá chạm cận dưới Bollinger Bands, RSI={rsi:.1f} quá bán cực đại $\rightarrow$ Kỳ vọng hồi phục về trục giữa.")
        # Overbought Snapback (Short opportunity)
        elif current_price >= bb_upper or rsi >= 70.0:
            return StrategyVote("Bắt Đảo Chiều (Mean Reversion)", -1, -80.0, 0.35, f"Giá chạm dải trên Bollinger Bands, RSI={rsi:.1f} quá mua cực đại $\rightarrow$ Kỳ vọng điều chỉnh về trục giữa.")
        else:
            return StrategyVote("Bắt Đảo Chiều (Mean Reversion)", 0, 0.0, 0.2, f"RSI ở mức trung tính {rsi:.1f}, dao động trong lòng dải Bollinger Bands.")


class LiquiditySweepSubEngine:
    def evaluate(self, df_15m: pd.DataFrame, current_price: float) -> StrategyVote:
        if df_15m is None or len(df_15m) < 20:
            return StrategyVote("Săn Rút Chân Cá Mập (Liquidity Sweep)", 0, 0.0, 0.2, "Đang nạp nến...")

        last_candle = df_15m.iloc[-1]
        c_open = last_candle["open"]
        c_high = last_candle["high"]
        c_low = last_candle["low"]
        c_close = last_candle["close"]
        candle_range = max(1.0, c_high - c_low)

        lower_wick = min(c_open, c_close) - c_low
        upper_wick = c_high - max(c_open, c_close)
        vol_ratio = (last_candle["volume"] / df_15m["volume"].iloc[-10:].mean()) if len(df_15m) >= 10 else 1.0

        # Bullish Pinbar / Sweep low
        if (lower_wick / candle_range) >= 0.55 and vol_ratio >= 1.4:
            return StrategyVote("Săn Rút Chân Cá Mập (Liquidity Sweep)", 1, 90.0, 0.25, f"Xuất hiện nến rút chân bẫy gấu (Volume x{vol_ratio:.1f}), cá mập quét thanh khoản đáy rồi mua thốc lên.")
        # Bearish Shooting Star / Sweep high
        elif (upper_wick / candle_range) >= 0.55 and vol_ratio >= 1.4:
            return StrategyVote("Săn Rút Chân Cá Mập (Liquidity Sweep)", -1, -90.0, 0.25, f"Xuất hiện nến rút râu trên (Volume x{vol_ratio:.1f}), cá mập xả hàng từ chối giá cao.")
        else:
            return StrategyVote("Săn Rút Chân Cá Mập (Liquidity Sweep)", 0, 0.0, 0.15, "Thân nến tiêu chuẩn, không có dấu hiệu quét thanh khoản bất thường.")


from strategy.multi_candle_patterns import MultiTimeframeCandleStrategyEngine, MTFCandleConfluenceResult


class MultiCandlePatternSubEngine:
    def __init__(self):
        self.engine = MultiTimeframeCandleStrategyEngine()

    def evaluate(self, data_map: Dict[str, pd.DataFrame], current_price: float) -> Tuple[StrategyVote, MTFCandleConfluenceResult]:
        res = self.engine.evaluate(data_map, current_price)
        dir_val = 1 if res.confluence_score >= 25.0 else (-1 if res.confluence_score <= -25.0 else 0)
        vote = StrategyVote(
            name="Mẫu Hình Nến Đa Khung (MTF Patterns)",
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
        market_regime: str
    ) -> EnsembleResult:
        df_15m = data_map.get("15m")
        df_1h = data_map.get("1h")
        df_1w = data_map.get("1w")
        df_1M = data_map.get("1M")

        v_trend = self.trend_engine.evaluate(df_15m, df_1h, current_price, df_1w=df_1w, df_1M=df_1M)
        v_reversion = self.reversion_engine.evaluate(df_15m, current_price)
        v_sweep = self.sweep_engine.evaluate(df_15m, current_price)
        v_candle, candle_res = self.candle_engine.evaluate(data_map, current_price)
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
