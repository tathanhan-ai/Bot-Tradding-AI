"""
Multi-Candle Pattern Recognition & Multi-Timeframe Strategy Engine
Modeled after Institutional Price Action & Japanese Candlestick Quantification.

Key Mechanisms:
1. Multi-Candle Pattern Recognizer:
   - Pin Bar / Hammer (Bullish Rejection) & Shooting Star (Bearish Rejection)
   - Bullish & Bearish Engulfing (Nhấn chìm xu hướng)
   - Morning Star & Evening Star (Cụm 3 nến đảo chiều đáy/đỉnh)
   - Three White Soldiers & Three Black Crows (3 nến dòng tiền áp đảo)
   - Inside Bar & Fakey Breakout (Nến nén tích lũy & phá vỡ giả)
   - Tweezer Tops & Bottoms (Đỉnh/Đáy nhíp từ chối ngưỡng cản)

2. 4 Timeframe-Specific Strategic Sub-Engines:
   - 1m Micro Scalp: Bứt phá xung lực ngắn hạn (Micro Momentum Breakout)
   - 5m Key Reversal: Săn đảo chiều tại vùng cản SMC hoặc VWAP (Pinbar/Engulfing)
   - 15m Trend Continuation: Nến tiếp diễn đà tăng/giảm sau cú pullback về EMA
   - 1h Institutional Macro Filter: Cấu trúc xu hướng lớn vĩ mô (Cấm đánh ngược HTF)

3. Cascade Multi-Timeframe Confluence Scorer:
   - Tổng hợp đa khung thời gian (1m, 5m, 15m, 1h) để triệt tiêu nhiễu của 1 cây nến đơn lẻ.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np


@dataclass
class CandlePatternResult:
    timeframe: str
    pattern_name: str           # e.g. "BULLISH_ENGULFING", "HAMMER_PINBAR", "NONE"
    pattern_display: str        # e.g. "Nhấn Chìm Tăng 🟢", "Pin Bar Rút Chân 🟢", "Bình Thường"
    bias: str                   # "BULLISH", "BEARISH", "NEUTRAL"
    strength: float             # 0.0 to 100.0
    candle_count: int           # 1, 2, or 3 candles involved
    description: str
    last_candle_open: float = 0.0
    last_candle_high: float = 0.0
    last_candle_low: float = 0.0
    last_candle_close: float = 0.0
    last_candle_volume: float = 0.0


@dataclass
class MTFCandleConfluenceResult:
    confluence_score: float     # -100.0 (Extreme Bearish) to +100.0 (Extreme Bullish)
    confluence_verdict: str     # 'STRONG_BULLISH_CASCADE', 'BULLISH_BIAS', 'NEUTRAL_MIXED', 'BEARISH_BIAS', 'STRONG_BEARISH_CASCADE'
    aligned_timeframes_count: int  # 0 to 4
    total_timeframes: int       # 4 (1m, 5m, 15m, 1h)
    timeframe_patterns: Dict[str, CandlePatternResult]
    recommended_action: str     # 'BUY_SCALP', 'BUY_SWING', 'SELL_SCALP', 'SELL_SWING', 'STAND_ASIDE'
    macro_alignment: bool       # True if lower timeframes agree with 1h HTF
    summary_rationale: str
    tactical_timeframe: str = "15m"
    radar_score: float = 0.0


class CandlePatternDetector:
    """
    Algorithmic quantification of multi-candle patterns using mathematical thresholds:
    - Shadow-to-body ratios
    - Relative volume surges
    - High/Low engulfment logic
    """

    @staticmethod
    def detect_patterns(df: Optional[pd.DataFrame], timeframe: str = "15m") -> CandlePatternResult:
        if df is None or len(df) < 5:
            return CandlePatternResult(
                timeframe=timeframe,
                pattern_name="INSUFFICIENT_DATA",
                pattern_display="Đang nạp nến...",
                bias="NEUTRAL",
                strength=0.0,
                candle_count=0,
                description="Chưa đủ dữ liệu nến để nhận diện mẫu hình."
            )

        # Work with the last 3 closed or forming candles
        c0 = df.iloc[-1]  # Latest candle
        c1 = df.iloc[-2]  # Previous candle
        c2 = df.iloc[-3]  # 2 candles ago

        o0, h0, l0, cl0, v0 = float(c0["open"]), float(c0["high"]), float(c0["low"]), float(c0["close"]), float(c0["volume"])
        o1, h1, l1, cl1, v1 = float(c1["open"]), float(c1["high"]), float(c1["low"]), float(c1["close"]), float(c1["volume"])
        o2, h2, l2, cl2, v2 = float(c2["open"]), float(c2["high"]), float(c2["low"]), float(c2["close"]), float(c2["volume"])

        range0 = max(0.1, h0 - l0)
        body0 = abs(cl0 - o0)
        upper_wick0 = h0 - max(o0, cl0)
        lower_wick0 = min(o0, cl0) - l0

        range1 = max(0.1, h1 - l1)
        body1 = abs(cl1 - o1)

        # Baseline volume average (last 10 candles)
        avg_vol = float(df["volume"].iloc[-10:].mean()) if len(df) >= 10 else v0
        vol_ratio0 = (v0 / avg_vol) if avg_vol > 0 else 1.0

        # -------------------------------------------------------------
        # 1. MORNING STAR (3-Candle Bullish Reversal)
        # -------------------------------------------------------------
        if (cl2 < o2 and (abs(cl2 - o2) / max(0.1, h2 - l2)) > 0.4) and \
           (body1 / range1 < 0.35 and l1 <= l2) and \
           (cl0 > o0 and cl0 >= (o2 + cl2) / 2.0 and v0 > avg_vol * 0.9):
            return CandlePatternResult(
                timeframe=timeframe,
                pattern_name="MORNING_STAR",
                pattern_display="Sao Mai Đảo Chiều Đáy 🌅",
                bias="BULLISH",
                strength=90.0,
                candle_count=3,
                description="Mẫu hình 3 nến Sao Mai (Morning Star) xác nhận cạn kiệt lực bán và đảo chiều tăng mạnh.",
                last_candle_open=o0, last_candle_high=h0, last_candle_low=l0, last_candle_close=cl0, last_candle_volume=v0
            )

        # -------------------------------------------------------------
        # 2. EVENING STAR (3-Candle Bearish Reversal)
        # -------------------------------------------------------------
        if (cl2 > o2 and (abs(cl2 - o2) / max(0.1, h2 - l2)) > 0.4) and \
           (body1 / range1 < 0.35 and h1 >= h2) and \
           (cl0 < o0 and cl0 <= (o2 + cl2) / 2.0 and v0 > avg_vol * 0.9):
            return CandlePatternResult(
                timeframe=timeframe,
                pattern_name="EVENING_STAR",
                pattern_display="Sao Hôm Đảo Chiều Đỉnh 🌇",
                bias="BEARISH",
                strength=90.0,
                candle_count=3,
                description="Mẫu hình 3 nến Sao Hôm (Evening Star) xác nhận đỉnh kiệt sức và dòng tiền bán tháo đảo chiều.",
                last_candle_open=o0, last_candle_high=h0, last_candle_low=l0, last_candle_close=cl0, last_candle_volume=v0
            )

        # -------------------------------------------------------------
        # 3. THREE WHITE SOLDIERS (3 Consecutive Bullish Marubozu)
        # -------------------------------------------------------------
        if (cl2 > o2 and cl1 > o1 and cl0 > o0) and \
           (cl0 > cl1 > cl2) and (h0 > h1 > h2) and \
           (lower_wick0 / range0 < 0.25 and upper_wick0 / range0 < 0.25):
            return CandlePatternResult(
                timeframe=timeframe,
                pattern_name="THREE_WHITE_SOLDIERS",
                pattern_display="3 Chàng Lính Ngự Lâm ⚔️🟢",
                bias="BULLISH",
                strength=88.0,
                candle_count=3,
                description="3 nến xanh đặc thân tăng liên tục bứt phá các đỉnh, phe mua áp đảo hoàn toàn.",
                last_candle_open=o0, last_candle_high=h0, last_candle_low=l0, last_candle_close=cl0, last_candle_volume=v0
            )

        # -------------------------------------------------------------
        # 4. THREE BLACK CROWS (3 Consecutive Bearish Marubozu)
        # -------------------------------------------------------------
        if (cl2 < o2 and cl1 < o1 and cl0 < o0) and \
           (cl0 < cl1 < cl2) and (l0 < l1 < l2) and \
           (lower_wick0 / range0 < 0.25 and upper_wick0 / range0 < 0.25):
            return CandlePatternResult(
                timeframe=timeframe,
                pattern_name="THREE_BLACK_CROWS",
                pattern_display="3 Con Quạ Đen 🦅🔴",
                bias="BEARISH",
                strength=88.0,
                candle_count=3,
                description="3 nến đỏ đặc thân giảm dốc liên tục phá vỡ các đáy, phe bán áp đảo toàn diện.",
                last_candle_open=o0, last_candle_high=h0, last_candle_low=l0, last_candle_close=cl0, last_candle_volume=v0
            )

        # -------------------------------------------------------------
        # 5. BULLISH ENGULFING (2-Candle Engulfing)
        # -------------------------------------------------------------
        if (cl1 < o1) and (cl0 > o0) and (cl0 >= o1) and (o0 <= cl1) and (body0 > body1 * 1.05):
            str_bonus = min(20.0, vol_ratio0 * 10.0)
            return CandlePatternResult(
                timeframe=timeframe,
                pattern_name="ENGULFING_BULLISH",
                pattern_display="Nhấn Chìm Tăng Trưởng (Engulfing) 🟢",
                bias="BULLISH",
                strength=min(95.0, 75.0 + str_bonus),
                candle_count=2,
                description="Nến xanh sau bao trùm hoàn toàn nến đỏ trước kèm khối lượng cao, phủ định phe bán.",
                last_candle_open=o0, last_candle_high=h0, last_candle_low=l0, last_candle_close=cl0, last_candle_volume=v0
            )

        # -------------------------------------------------------------
        # 6. BEARISH ENGULFING (2-Candle Engulfing)
        # -------------------------------------------------------------
        if (cl1 > o1) and (cl0 < o0) and (cl0 <= o1) and (o0 >= cl1) and (body0 > body1 * 1.05):
            str_bonus = min(20.0, vol_ratio0 * 10.0)
            return CandlePatternResult(
                timeframe=timeframe,
                pattern_name="ENGULFING_BEARISH",
                pattern_display="Nhấn Chìm Giảm Giá (Engulfing) 🔴",
                bias="BEARISH",
                strength=min(95.0, 75.0 + str_bonus),
                candle_count=2,
                description="Nến đỏ sau nuốt trọn nến xanh trước với volume lớn, phe bán bẻ gãy đà tăng.",
                last_candle_open=o0, last_candle_high=h0, last_candle_low=l0, last_candle_close=cl0, last_candle_volume=v0
            )

        # -------------------------------------------------------------
        # 7. BULLISH PIN BAR / HAMMER (1-Candle Long Tail Rejection)
        # -------------------------------------------------------------
        if (lower_wick0 >= 2.0 * body0) and (upper_wick0 <= 0.25 * range0) and (lower_wick0 >= 0.55 * range0):
            return CandlePatternResult(
                timeframe=timeframe,
                pattern_name="HAMMER_PINBAR",
                pattern_display="Pin Bar Búa Rút Chân 🔨🟢",
                bias="BULLISH",
                strength=82.0,
                candle_count=1,
                description=f"Râu dưới dài gấp {lower_wick0/max(1.0, body0):.1f}x thân nến. Phe mua quét sạch thanh khoản và đẩy giá ngược trở lại cực mạnh.",
                last_candle_open=o0, last_candle_high=h0, last_candle_low=l0, last_candle_close=cl0, last_candle_volume=v0
            )

        # -------------------------------------------------------------
        # 8. BEARISH PIN BAR / SHOOTING STAR (1-Candle Rejection)
        # -------------------------------------------------------------
        if (upper_wick0 >= 2.0 * body0) and (lower_wick0 <= 0.25 * range0) and (upper_wick0 >= 0.55 * range0):
            return CandlePatternResult(
                timeframe=timeframe,
                pattern_name="SHOOTING_STAR",
                pattern_display="Sao Băng Xả Hàng (Shooting Star) 🌠🔴",
                bias="BEARISH",
                strength=82.0,
                candle_count=1,
                description=f"Râu trên dài gấp {upper_wick0/max(1.0, body0):.1f}x thân nến. Phe bán xả hàng cực mạnh đập gãy nhịp tăng của phe mua.",
                last_candle_open=o0, last_candle_high=h0, last_candle_low=l0, last_candle_close=cl0, last_candle_volume=v0
            )

        # -------------------------------------------------------------
        # 9. INSIDE BAR (Volatility Contraction)
        # -------------------------------------------------------------
        if (h0 < h1) and (l0 > l1):
            return CandlePatternResult(
                timeframe=timeframe,
                pattern_name="INSIDE_BAR",
                pattern_display="Nến Nén Biên Độ (Inside Bar) 📦",
                bias="NEUTRAL",
                strength=60.0,
                candle_count=2,
                description="Nến hiện tại nằm trọn trong nến mẹ (Mother Bar). Thị trường đang nén lò xo trước cú bùng nổ.",
                last_candle_open=o0, last_candle_high=h0, last_candle_low=l0, last_candle_close=cl0, last_candle_volume=v0
            )

        # -------------------------------------------------------------
        # 10. TWEEZER BOTTOM (Đáy nhíp đôi)
        # -------------------------------------------------------------
        if abs(l0 - l1) <= (range0 * 0.05) and lower_wick0 > body0 and (cl0 > o0):
            return CandlePatternResult(
                timeframe=timeframe,
                pattern_name="TWEEZER_BOTTOM",
                pattern_display="Đáy Nhíp Từ Chối Cản 🧲🟢",
                bias="BULLISH",
                strength=78.0,
                candle_count=2,
                description="2 nến liên tiếp giữ vững cùng một mốc đáy giá thấp nhất, xác nhận vùng hỗ trợ vững chắc.",
                last_candle_open=o0, last_candle_high=h0, last_candle_low=l0, last_candle_close=cl0, last_candle_volume=v0
            )

        # -------------------------------------------------------------
        # 11. TWEEZER TOP (Đỉnh nhíp đôi)
        # -------------------------------------------------------------
        if abs(h0 - h1) <= (range0 * 0.05) and upper_wick0 > body0 and (cl0 < o0):
            return CandlePatternResult(
                timeframe=timeframe,
                pattern_name="TWEEZER_TOP",
                pattern_display="Đỉnh Nhíp Từ Chối Cản 🧲🔴",
                bias="BEARISH",
                strength=78.0,
                candle_count=2,
                description="2 nến liên tiếp dội ngược tại cùng một đỉnh kháng cự, xác nhận tường xả cản trên.",
                last_candle_open=o0, last_candle_high=h0, last_candle_low=l0, last_candle_close=cl0, last_candle_volume=v0
            )

        # Default trend candle
        bias = "BULLISH" if cl0 > o0 else ("BEARISH" if cl0 < o0 else "NEUTRAL")
        return CandlePatternResult(
            timeframe=timeframe,
            pattern_name="NORMAL_MOMENTUM",
            pattern_display=f"Nến Xu Hướng ({'Tăng' if bias == 'BULLISH' else ('Giảm' if bias == 'BEARISH' else 'Doji')})",
            bias=bias,
            strength=40.0,
            candle_count=1,
            description="Không phát hiện mẫu hình đảo chiều đặc biệt, vận động theo nến xu hướng tự nhiên.",
            last_candle_open=o0, last_candle_high=h0, last_candle_low=l0, last_candle_close=cl0, last_candle_volume=v0
        )


class MultiTimeframeCandleStrategyEngine:
    """
    Cascading Multi-Timeframe Strategy Coordinator:
    Evaluates 1m, 5m, 15m, 1h, 1w (Weekly), and 1M (Monthly) simultaneously.
    Combines timeframe weights to produce an institutional confluence signal.
    """
    TIMEFRAME_WEIGHTS = {
        "1m": 0.08,
        "3m": 0.08,
        "5m": 0.12,
        "15m": 0.18,
        "30m": 0.10,
        "1h": 0.16,
        "4h": 0.10,
        "1d": 0.08,
        "1w": 0.05,
        "1M": 0.05,
    }

    def __init__(self):
        self.detector = CandlePatternDetector()

    def evaluate(self, data_map: Dict[str, pd.DataFrame], current_price: float, active_timeframe: str = "15m") -> MTFCandleConfluenceResult:
        base_weights = dict(self.TIMEFRAME_WEIGHTS)
        if active_timeframe in base_weights:
            base_weights[active_timeframe] += 0.15

        valid_tfs = [tf for tf, df in data_map.items() if df is not None and not df.empty and tf in base_weights]
        total_w = sum(base_weights[tf] for tf in valid_tfs) or 1.0

        patterns: Dict[str, CandlePatternResult] = {}
        for tf in ("1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d", "1w", "1M"):
            df_tf = data_map.get(tf)
            if df_tf is not None and not df_tf.empty:
                patterns[tf] = self.detector.detect_patterns(df_tf, timeframe=tf)

        # 1. Macro HTF Direction (1h, 4h, 1d, 1w, 1M)
        p_1h = patterns.get("1h")
        p_4h = patterns.get("4h")
        p_1d = patterns.get("1d")
        p_1w = patterns.get("1w")
        p_1M = patterns.get("1M")

        htf_biases = [p.bias for p in [p_1h, p_4h, p_1d, p_1w, p_1M] if p and p.bias != "NEUTRAL"]
        if htf_biases.count("BULLISH") > htf_biases.count("BEARISH"):
            macro_bias = "BULLISH"
        elif htf_biases.count("BEARISH") > htf_biases.count("BULLISH"):
            macro_bias = "BEARISH"
        else:
            macro_bias = p_1h.bias if p_1h else "NEUTRAL"

        # 2. Weighted Score Calculation
        weighted_score = 0.0
        bullish_count = 0
        bearish_count = 0

        for tf, pat in patterns.items():
            w = (base_weights.get(tf, 0.10) / total_w)
            direction_mult = 1.0 if pat.bias == "BULLISH" else (-1.0 if pat.bias == "BEARISH" else 0.0)
            weighted_score += direction_mult * pat.strength * w

            if pat.bias == "BULLISH":
                bullish_count += 1
            elif pat.bias == "BEARISH":
                bearish_count += 1

        # Check macro alignment (Are smaller timeframes aligned with HTF?)
        p_15m = patterns.get("15m")
        p_5m = patterns.get("5m")
        p_1m = patterns.get("1m")

        macro_alignment = True
        if macro_bias == "BULLISH" and ((p_15m and p_15m.bias == "BEARISH") or (p_5m and p_5m.bias == "BEARISH")):
            macro_alignment = False
        elif macro_bias == "BEARISH" and ((p_15m and p_15m.bias == "BULLISH") or (p_5m and p_5m.bias == "BULLISH")):
            macro_alignment = False

        # Determine tactical timeframe from the clearest/strongest pattern.
        # Ha nguong 65 -> 55 de khung tac chien kich hoat duoc (Tweezer 78, Engulfing 75-95
        # van qua; Inside Bar 60 van qua khi thuan huong; chi nen thuong 40 bi loai).
        target_bias = "BULLISH" if weighted_score > 0 else ("BEARISH" if weighted_score < 0 else None)
        tactical_tf = active_timeframe or "15m"
        best_str = 0.0
        candidate_tfs = ["15m", "1h", "5m", "1m"]
        if target_bias:
            for tf_cand in candidate_tfs:
                p_cand = patterns.get(tf_cand)
                if p_cand and p_cand.bias == target_bias and p_cand.strength > best_str and p_cand.strength >= 55.0:
                    best_str = p_cand.strength
                    tactical_tf = tf_cand

        # Verdict and Multi-Timeframe Tactical Action
        if weighted_score >= 45.0 and macro_bias == "BULLISH":
            verdict = "STRONG_BULLISH_CASCADE"
            action = "BUY_SWING" if tactical_tf in ("15m", "1h", "4h") else "BUY_SCALP"
            aligned_count = bullish_count
        elif weighted_score >= 18.0:
            verdict = "BULLISH_BIAS"
            action = "BUY_SWING" if tactical_tf in ("15m", "1h", "4h") else "BUY_SCALP"
            aligned_count = bullish_count
        elif weighted_score <= -45.0 and macro_bias == "BEARISH":
            verdict = "STRONG_BEARISH_CASCADE"
            action = "SELL_SWING" if tactical_tf in ("15m", "1h", "4h") else "SELL_SCALP"
            aligned_count = bearish_count
        elif weighted_score <= -18.0:
            verdict = "BEARISH_BIAS"
            action = "SELL_SWING" if tactical_tf in ("15m", "1h", "4h") else "SELL_SCALP"
            aligned_count = bearish_count
        else:
            verdict = "NEUTRAL_MIXED"
            action = "STAND_ASIDE"
            aligned_count = max(bullish_count, bearish_count)

        # Rationale builder
        p1m_name = p_1m.pattern_display if p_1m else "--"
        p5m_name = p_5m.pattern_display if p_5m else "--"
        p15m_name = p_15m.pattern_display if p_15m else "--"
        p1h_name = p_1h.pattern_display if p_1h else "--"
        p1w_name = p_1w.pattern_display if p_1w else "--"
        p1M_name = p_1M.pattern_display if p_1M else "--"

        rationale = (
            f"[1m]: {p1m_name} | [5m]: {p5m_name} | [15m]: {p15m_name} | [1h]: {p1h_name} | [1W]: {p1w_name} | [1M]: {p1M_name} "
            f"-> Đồng thuận {aligned_count}/6 khung ({'+' if weighted_score >= 0 else ''}{weighted_score:.1f} điểm, Khung tác chiến: {tactical_tf})."
        )

        return MTFCandleConfluenceResult(
            confluence_score=round(weighted_score, 1),
            confluence_verdict=verdict,
            aligned_timeframes_count=aligned_count,
            total_timeframes=6,
            timeframe_patterns=patterns,
            recommended_action=action,
            macro_alignment=macro_alignment,
            summary_rationale=rationale,
            tactical_timeframe=tactical_tf,
            radar_score=round(weighted_score, 1)
        )
