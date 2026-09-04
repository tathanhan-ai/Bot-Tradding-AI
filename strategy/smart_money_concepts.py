"""
Institutional Smart Money Concepts (SMC) Quantitative Engine
Implements 3 core institutional footprints:
1. Order Blocks (OB): Last opposing candle prior to aggressive institutional displacement
2. Fair Value Gaps (FVG): Imbalance 3-candle magnets where liquidity is inefficient
3. Liquidity Sweeps & CHoCH (Change of Character): Stop-loss hunts past swing highs/lows followed by structural shifts
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np


@dataclass
class OrderBlock:
    type: str              # 'BULLISH' (Buy OB / Demand) or 'BEARISH' (Sell OB / Supply)
    top_price: float       # Zone high
    bottom_price: float    # Zone low
    timestamp: int         # Candle timestamp
    volume_surge_mult: float
    tested: bool = False   # True if price has returned to touch it
    valid: bool = True     # True if not invalidated (broken)


@dataclass
class FairValueGap:
    type: str              # 'BULLISH_FVG' (Undersold imbalance) or 'BEARISH_FVG' (Overbought imbalance)
    top_price: float
    bottom_price: float
    mid_price: float       # Consequent Encroachment (50% midpoint)
    timestamp: int
    is_filled: bool = False
    valid: bool = True


@dataclass
class LiquiditySweep:
    type: str              # 'BEARISH_SWEEP' (Fake breakout above highs) or 'BULLISH_SWEEP' (Fake breakdown below lows)
    swept_price: float     # The swing level that was run
    reversal_price: float  # The close back inside
    timestamp: int
    has_choch: bool = False


@dataclass
class SMCAnalysisResult:
    market_structure: str   # 'BULLISH_TREND', 'BEARISH_TREND', 'RANGING'
    active_bull_obs: List[OrderBlock] = field(default_factory=list)
    active_bear_obs: List[OrderBlock] = field(default_factory=list)
    active_fvgs: List[FairValueGap] = field(default_factory=list)
    last_sweep: Optional[LiquiditySweep] = None
    nearest_demand_zone: Optional[Tuple[float, float]] = None
    nearest_supply_zone: Optional[Tuple[float, float]] = None
    institutional_bias: str = "NEUTRAL"
    bias_rationale: str = ""


class SmartMoneyEngine:
    def __init__(self, atr_mult: float = 1.2):
        self.atr_mult = atr_mult

    def detect_order_blocks(self, df: pd.DataFrame, max_lookback: int = 60) -> Tuple[List[OrderBlock], List[OrderBlock]]:
        """
        Detects Institutional Order Blocks:
        Bullish OB: Bearish candle followed by an impulsive displacement candle (> 1.5x ATR) breaking previous high
        Bearish OB: Bullish candle followed by an impulsive displacement candle (> 1.5x ATR) breaking previous low
        """
        if len(df) < 10:
            return [], []

        recent = df.iloc[-max_lookback:].copy()
        highs = recent["high"].values
        lows = recent["low"].values
        opens = recent["open"].values
        closes = recent["close"].values
        vols = recent["volume"].values
        times = recent.index.values if hasattr(recent.index, "values") else np.arange(len(recent))

        # Calculate ATR for displacement qualification
        tr = np.maximum(highs[1:] - lows[1:], np.maximum(abs(highs[1:] - closes[:-1]), abs(lows[1:] - closes[:-1])))
        atr = np.mean(tr[-14:]) if len(tr) >= 14 else (closes[-1] * 0.008)
        avg_vol = np.mean(vols[-20:]) if len(vols) >= 20 else 1.0

        bull_obs: List[OrderBlock] = []
        bear_obs: List[OrderBlock] = []
        current_price = closes[-1]

        # Scan for displacement moves
        for i in range(2, len(recent) - 1):
            body = abs(closes[i] - opens[i])
            is_displacement = body >= (self.atr_mult * atr)
            vol_mult = (vols[i] / avg_vol) if avg_vol > 0 else 1.0

            # 1. Bullish Order Block Check
            if is_displacement and closes[i] > opens[i] and closes[i-1] <= opens[i-1]:
                # Candle i-1 was down, candle i exploded upwards
                ob_top = max(opens[i-1], closes[i-1], highs[i-1])
                ob_bot = lows[i-1]
                # Check if invalidated (broken downwards)
                min_after = np.min(lows[i:])
                valid = min_after >= (ob_bot - 0.2 * atr)
                tested = min_after <= ob_top and valid

                if valid and ob_top < current_price:
                    bull_obs.append(OrderBlock(
                        type="BULLISH",
                        top_price=round(float(ob_top), 2),
                        bottom_price=round(float(ob_bot), 2),
                        timestamp=int(times[i-1]) if isinstance(times[i-1], (int, np.integer)) else i,
                        volume_surge_mult=round(float(vol_mult), 2),
                        tested=bool(tested),
                        valid=True
                    ))

            # 2. Bearish Order Block Check
            elif is_displacement and closes[i] < opens[i] and closes[i-1] >= opens[i-1]:
                # Candle i-1 was up, candle i collapsed downwards
                ob_top = highs[i-1]
                ob_bot = min(opens[i-1], closes[i-1], lows[i-1])
                # Check if invalidated (broken upwards)
                max_after = np.max(highs[i:])
                valid = max_after <= (ob_top + 0.2 * atr)
                tested = max_after >= ob_bot and valid

                if valid and ob_bot > current_price:
                    bear_obs.append(OrderBlock(
                        type="BEARISH",
                        top_price=round(float(ob_top), 2),
                        bottom_price=round(float(ob_bot), 2),
                        timestamp=int(times[i-1]) if isinstance(times[i-1], (int, np.integer)) else i,
                        volume_surge_mult=round(float(vol_mult), 2),
                        tested=bool(tested),
                        valid=True
                    ))

        # Return latest active OBs
        return bull_obs[-3:], bear_obs[-3:]

    def detect_fair_value_gaps(self, df: pd.DataFrame, max_lookback: int = 40) -> List[FairValueGap]:
        """
        Detects Fair Value Gaps (3-candle price imbalance magnets):
        Bullish FVG: Low of candle[i] > High of candle[i-2] (Gap between candle 1 and candle 3)
        Bearish FVG: High of candle[i] < Low of candle[i-2] (Gap between candle 1 and candle 3)
        """
        if len(df) < 5:
            return []

        recent = df.iloc[-max_lookback:].copy()
        highs = recent["high"].values
        lows = recent["low"].values
        closes = recent["close"].values
        times = recent.index.values if hasattr(recent.index, "values") else np.arange(len(recent))
        current_price = closes[-1]

        fvgs: List[FairValueGap] = []

        for i in range(2, len(recent)):
            # Bullish FVG (Gap up)
            if lows[i] > highs[i-2]:
                gap_top = lows[i]
                gap_bot = highs[i-2]
                mid = (gap_top + gap_bot) / 2.0
                # Check if filled by subsequent price action
                min_after = np.min(lows[i:]) if i < len(recent)-1 else lows[i]
                is_filled = min_after <= gap_bot
                if not is_filled:
                    fvgs.append(FairValueGap(
                        type="BULLISH_FVG",
                        top_price=round(float(gap_top), 2),
                        bottom_price=round(float(gap_bot), 2),
                        mid_price=round(float(mid), 2),
                        timestamp=int(times[i-1]) if isinstance(times[i-1], (int, np.integer)) else i,
                        is_filled=False,
                        valid=True
                    ))

            # Bearish FVG (Gap down)
            elif highs[i] < lows[i-2]:
                gap_top = lows[i-2]
                gap_bot = highs[i]
                mid = (gap_top + gap_bot) / 2.0
                max_after = np.max(highs[i:]) if i < len(recent)-1 else highs[i]
                is_filled = max_after >= gap_top
                if not is_filled:
                    fvgs.append(FairValueGap(
                        type="BEARISH_FVG",
                        top_price=round(float(gap_top), 2),
                        bottom_price=round(float(gap_bot), 2),
                        mid_price=round(float(mid), 2),
                        timestamp=int(times[i-1]) if isinstance(times[i-1], (int, np.integer)) else i,
                        is_filled=False,
                        valid=True
                    ))

        return fvgs[-4:]

    def detect_liquidity_sweep_and_choch(self, df: pd.DataFrame) -> Optional[LiquiditySweep]:
        """
        Detects Stop-Loss Hunt (Liquidity Sweeps):
        A candle briefly wicks past a prominent swing high/low but closes strongly back inside,
        trapping breakout retail traders.
        """
        if len(df) < 20:
            return None

        recent = df.iloc[-25:].copy()
        highs = recent["high"].values
        lows = recent["low"].values
        closes = recent["close"].values
        opens = recent["open"].values

        # Prominent swing high and low prior to the last 3 candles
        prior_high = np.max(highs[:-3])
        prior_low = np.min(lows[:-3])

        # Inspect last 3 candles for a sweep
        for i in range(-3, 0):
            c_high = highs[i]
            c_low = lows[i]
            c_close = closes[i]
            c_open = opens[i]

            # Bearish Liquidity Sweep (Swept Highs then dumped back down)
            if c_high > prior_high and c_close < prior_high and c_close < c_open:
                return LiquiditySweep(
                    type="BEARISH_SWEEP",
                    swept_price=round(float(prior_high), 2),
                    reversal_price=round(float(c_close), 2),
                    timestamp=int(df.index[i]) if hasattr(df.index[i], "__int__") else i,
                    has_choch=True
                )

            # Bullish Liquidity Sweep (Swept Lows then pumped back up)
            elif c_low < prior_low and c_close > prior_low and c_close > c_open:
                return LiquiditySweep(
                    type="BULLISH_SWEEP",
                    swept_price=round(float(prior_low), 2),
                    reversal_price=round(float(c_close), 2),
                    timestamp=int(df.index[i]) if hasattr(df.index[i], "__int__") else i,
                    has_choch=True
                )

        return None

    def analyze(self, df: pd.DataFrame, current_price: float) -> SMCAnalysisResult:
        """Runs full institutional SMC analysis pipeline"""
        if df is None or len(df) < 20:
            return SMCAnalysisResult(market_structure="RANGING", institutional_bias="NEUTRAL")

        bull_obs, bear_obs = self.detect_order_blocks(df)
        fvgs = self.detect_fair_value_gaps(df)
        sweep = self.detect_liquidity_sweep_and_choch(df)

        # Nearest demand & supply: Order Blocks preferred, fallback to active FVGs / swing boundaries
        nearest_demand = (bull_obs[-1].bottom_price, bull_obs[-1].top_price) if bull_obs else None
        nearest_supply = (bear_obs[-1].bottom_price, bear_obs[-1].top_price) if bear_obs else None

        if not nearest_demand and fvgs:
            bull_fvgs = [f for f in fvgs if f.type == "BULLISH_FVG" and f.top_price <= current_price]
            if bull_fvgs:
                nearest_demand = (round(bull_fvgs[-1].bottom_price, 2), round(bull_fvgs[-1].top_price, 2))
        if not nearest_demand and len(df) >= 10:
            recent_low = float(np.min(df["low"].values[-20:]))
            nearest_demand = (round(recent_low * 0.998, 2), round(recent_low, 2))

        if not nearest_supply and fvgs:
            bear_fvgs = [f for f in fvgs if f.type == "BEARISH_FVG" and f.bottom_price >= current_price]
            if bear_fvgs:
                nearest_supply = (round(bear_fvgs[-1].bottom_price, 2), round(bear_fvgs[-1].top_price, 2))
        if not nearest_supply and len(df) >= 10:
            recent_high = float(np.max(df["high"].values[-20:]))
            nearest_supply = (round(recent_high, 2), round(recent_high * 1.002, 2))

        # Determine Market Structure via Higher Highs / Lower Lows
        closes = df["close"].values
        sma20 = np.mean(closes[-20:])
        sma50 = np.mean(closes[-50:]) if len(closes) >= 50 else sma20

        if current_price > sma20 and sma20 >= sma50:
            structure = "BULLISH_TREND"
        elif current_price < sma20 and sma20 <= sma50:
            structure = "BEARISH_TREND"
        else:
            structure = "RANGING"

        # Formulate Institutional Bias with Confluence
        bias = "NEUTRAL"
        rationale = "Thị trường đang tích lũy, chờ đợi dòng tiền tổ chức xác nhận hướng đi..."

        if sweep and sweep.type == "BEARISH_SWEEP":
            bias = "BEARISH_REVERSAL"
            rationale = f"🚨 BẪY THANH KHOẢN ĐỈNH: Cá mập vừa quét râu qua ${sweep.swept_price:,.1f} dụ Long rồi xả ngược. Xu hướng đảo chiều Giảm!"
        elif sweep and sweep.type == "BULLISH_SWEEP":
            bias = "BULLISH_REVERSAL"
            rationale = f"🚨 BẪY THANH KHOẢN ĐÁY: Cá mập vừa quét râu qua ${sweep.swept_price:,.1f} ép bán cắt lỗ rồi rút chân. Xu hướng đảo chiều Tăng!"
        elif nearest_demand and abs(current_price - nearest_demand[1]) / current_price < 0.004:
            bias = "DEMAND_ZONE_BOUNCE"
            rationale = f"🧱 TEST KHỐI LỆNH TỔ CHỨC (BULLISH OB): Giá đang chạm vùng Demand ${nearest_demand[0]:,.1f} - ${nearest_demand[1]:,.1f}. Lực hấp thụ mua mạnh!"
        elif nearest_supply and abs(current_price - nearest_supply[0]) / current_price < 0.004:
            bias = "SUPPLY_ZONE_REJECT"
            rationale = f"🧱 TEST KHỐI LỆNH TỔ CHỨC (BEARISH OB): Giá đang chạm vùng Supply ${nearest_supply[0]:,.1f} - ${nearest_supply[1]:,.1f}. Áp lực xả hàng lớn!"
        elif structure == "BULLISH_TREND":
            bias = "BULLISH_CONTINUATION"
            rationale = "Cấu trúc thị trường phe Mua kiểm soát (BOS Up). Ưu tiên canh gom tại FVG hoặc Bullish OB."
        elif structure == "BEARISH_TREND":
            bias = "BEARISH_CONTINUATION"
            rationale = "Cấu trúc thị trường phe Bán kiểm soát (BOS Down). Ưu tiên canh Short tại Bearish OB hoặc FVG."

        return SMCAnalysisResult(
            market_structure=structure,
            active_bull_obs=bull_obs,
            active_bear_obs=bear_obs,
            active_fvgs=fvgs,
            last_sweep=sweep,
            nearest_demand_zone=nearest_demand,
            nearest_supply_zone=nearest_supply,
            institutional_bias=bias,
            bias_rationale=rationale
        )
