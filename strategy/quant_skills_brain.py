"""
Quant Skills Intelligence Core (Động Cơ Định Lượng Độc Lập Chuẩn Hóa)
Được chuẩn hóa và chưng cất từ bộ kỹ năng giao dịch định lượng hàng đầu (Claude Trading Skills).
Hoạt động độc lập 100% bằng NumPy và Pandas thuần túy, không phụ thuộc thư viện ngoài:
1. Hurst Exponent (R/S Rescaled Range) -> Phân định chính xác Mean-Reversion (<0.45) vs Random Walk vs Trending (>0.55).
2. Garman-Klass và Parkinson Volatility -> Đo lường độ biến động OHLC hiệu suất thống kê x8 so với nến đóng cửa.
3. Order Book Imbalance (OBI) -> Dự báo áp lực dịch chuyển giá tức thời từ vi mô sổ lệnh.
4. Chandelier ATR Trailing Exit và 4-Stage Systematic Take-Profit -> Tự động dời Breakeven và thả trôi sóng lớn.
"""
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class HurstRegimeResult:
    hurst_exponent: float
    regime_type: str        # 'MEAN_REVERTING', 'RANDOM_WALK', 'TRENDING'
    confidence: float       # 0.0 - 1.0
    description: str


@dataclass
class HighEfficiencyVolatilityResult:
    garman_klass_annualized: float
    garman_klass_bar: float
    parkinson_bar: float
    close_to_close_bar: float
    volatility_ratio: float  # Current vol / historical baseline (x1.0 normal, >x1.8 breakout)
    volatility_regime: str   # 'LOW_COMPRESSION', 'NORMAL', 'HIGH_VOLATILITY', 'EXTREME_EXPANSION'


@dataclass
class ScaledExitTranche:
    tranche_id: int
    size_pct: float
    target_price: float
    trigger_condition: str
    action_after: str


class QuantSkillsBrain:
    """
    Động cơ phân tích định lượng chuẩn hóa, thiết kế độc lập và di động 100%.
    """
    def __init__(self):
        pass

    @staticmethod
    def calculate_hurst_exponent(series: pd.Series, max_lag: int = 30) -> HurstRegimeResult:
        """
        Tính toán chỉ số Hurst Exponent (H) qua phương pháp Rescaled Range (R/S analysis):
        - H < 0.45: Mean-Reverting (Hồi quy trung bình cao, thích hợp đánh Grid / Biên tích lũy)
        - 0.45 <= H <= 0.55: Random Walk (Thị trường ngẫu nhiên, không có lợi thế thống kê rõ ràng)
        - H > 0.55: Persistent Trending (Xu hướng bền bỉ, thích hợp bám sóng Breakout)
        """
        if series is None or len(series) < 30:
            return HurstRegimeResult(
                hurst_exponent=0.50,
                regime_type='RANDOM_WALK',
                confidence=0.50,
                description='Chưa đủ dữ liệu chuỗi thời gian để tính Hurst (cần >= 30 nến).'
            )

        clean_series = series.dropna().astype(float)
        n = len(clean_series)
        eff_max_lag = min(max_lag, n // 2)
        lags = range(4, eff_max_lag)
        rs_values = []
        valid_lags = []

        for lag in lags:
            num_chunks = n // lag
            if num_chunks < 2:
                continue
            chunk_rs = []
            for i in range(num_chunks):
                chunk = clean_series.iloc[i * lag : (i + 1) * lag].values
                mean_val = np.mean(chunk)
                cum_dev = np.cumsum(chunk - mean_val)
                r = np.max(cum_dev) - np.min(cum_dev)
                s = np.std(chunk, ddof=1)
                if s > 1e-8:
                    chunk_rs.append(r / s)
            if chunk_rs:
                rs_values.append(np.mean(chunk_rs))
                valid_lags.append(lag)

        if len(valid_lags) < 4:
            return HurstRegimeResult(
                hurst_exponent=0.50,
                regime_type='RANDOM_WALK',
                confidence=0.50,
                description='Không đủ điểm phân đoạn thống kê R/S.'
            )

        log_lags = np.log(valid_lags)
        log_rs = np.log(rs_values)
        coeffs = np.polyfit(log_lags, log_rs, 1)
        h = float(coeffs[0])
        h_clamped = round(max(0.05, min(0.95, h)), 2)

        if h_clamped < 0.45:
            regime = 'MEAN_REVERTING'
            conf = min(0.95, 0.50 + (0.45 - h_clamped) * 1.5)
            desc = f'H={h_clamped:.2f} < 0.45 (Hồi Quy Trung Bình): Thị trường nén tích lũy, giá liên tục quay lại trục giữa.'
        elif h_clamped > 0.55:
            regime = 'TRENDING'
            conf = min(0.95, 0.50 + (h_clamped - 0.55) * 1.5)
            desc = f'H={h_clamped:.2f} > 0.55 (Xu Hướng Bền Bỉ): Thị trường có quán tính dẫn hướng mạnh, ưu tiên bám trend.'
        else:
            regime = 'RANDOM_WALK'
            conf = 0.50
            desc = f'H={h_clamped:.2f} ≈ 0.50 (Dao Động Ngẫu Nhiên): Chuyển động nhiễu không định hướng, nên giảm quy mô vị thế.'

        return HurstRegimeResult(
            hurst_exponent=h_clamped,
            regime_type=regime,
            confidence=round(conf, 2),
            description=desc
        )

    @staticmethod
    def bar_seconds(df: Optional[pd.DataFrame] = None, timeframe: Optional[str] = None) -> float:
        if timeframe is not None:
            units = {"m": 60, "h": 3600, "d": 86400, "w": 604800, "M": 2592000}
            seconds = float(timeframe[:-1]) * units[timeframe[-1]]
        elif df is not None and isinstance(df.index, pd.DatetimeIndex) and len(df) > 1:
            seconds = float(df.index.to_series().diff().dt.total_seconds().dropna().median())
        else:
            seconds = 900.0  # Preserve the original 15m default for unindexed callers.
        if not np.isfinite(seconds) or seconds <= 0:
            raise ValueError("candle duration must be positive")
        return seconds

    @staticmethod
    def calculate_garman_klass_volatility(
        df: pd.DataFrame, window: int = 14, timeframe: Optional[str] = None
    ) -> HighEfficiencyVolatilityResult:
        """
        Đo lường độ biến động Garman-Klass (OHLC) và Parkinson (High-Low).
        Đạt hiệu quả thống kê gấp 8 lần so với phương sai close-to-close thông thường.
        """
        if df is None or len(df) < window + 2:
            return HighEfficiencyVolatilityResult(
                garman_klass_annualized=0.0,
                garman_klass_bar=0.0,
                parkinson_bar=0.0,
                close_to_close_bar=0.0,
                volatility_ratio=1.0,
                volatility_regime='NORMAL'
            )

        o = df['open'].astype(float)
        h = df['high'].astype(float)
        l = df['low'].astype(float)
        c = df['close'].astype(float)

        log_hl = np.log(np.maximum(h / np.maximum(l, 1e-8), 1.0))
        log_co = np.log(np.maximum(c, 1e-8) / np.maximum(o, 1e-8))
        gk_series = 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)
        gk_bar = float(np.sqrt(np.maximum(0.0, gk_series.iloc[-window:].mean())))

        park_bar = float(np.sqrt(np.maximum(0.0, (log_hl ** 2).iloc[-window:].mean() / (4 * np.log(2)))))

        returns = np.log(c / c.shift(1)).dropna()
        c2c_bar = float(returns.iloc[-window:].std(ddof=1)) if len(returns) >= window else gk_bar

        annual_factor = np.sqrt(365 * 86400 / QuantSkillsBrain.bar_seconds(df, timeframe))
        gk_annualized = round(gk_bar * annual_factor * 100.0, 1)

        baseline_gk = float(np.sqrt(np.maximum(0.0, gk_series.iloc[-min(len(df), 50):].mean())))
        vol_ratio = round(gk_bar / max(baseline_gk, 1e-6), 2)

        if vol_ratio < 0.65:
            regime = 'LOW_COMPRESSION'
        elif vol_ratio > 1.80:
            regime = 'EXTREME_EXPANSION'
        elif vol_ratio > 1.25:
            regime = 'HIGH_VOLATILITY'
        else:
            regime = 'NORMAL'

        return HighEfficiencyVolatilityResult(
            garman_klass_annualized=gk_annualized,
            garman_klass_bar=round(gk_bar, 5),
            parkinson_bar=round(park_bar, 5),
            close_to_close_bar=round(c2c_bar, 5),
            volatility_ratio=vol_ratio,
            volatility_regime=regime
        )

    @staticmethod
    def calculate_order_book_imbalance(
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
        depth_levels: int = 5
    ) -> Dict[str, Any]:
        """
        Tính toán chỉ số Order Book Imbalance (OBI) từ vi mô sổ lệnh:
        OBI = (Bid_Vol - Ask_Vol) / (Bid_Vol + Ask_Vol) nằm trong [-1.0, +1.0]
        """
        if not bids or not asks:
            return {'obi': 0.0, 'status': 'BALANCED ⚪', 'bid_depth': 0.0, 'ask_depth': 0.0, 'ratio': 1.0}

        bid_vol = sum(size for _, size in bids[:depth_levels])
        ask_vol = sum(size for _, size in asks[:depth_levels])
        total_vol = bid_vol + ask_vol

        if total_vol <= 0:
            return {'obi': 0.0, 'status': 'BALANCED ⚪', 'bid_depth': 0.0, 'ask_depth': 0.0, 'ratio': 1.0}

        obi = (bid_vol - ask_vol) / total_vol
        obi = round(max(-1.0, min(1.0, obi)), 3)

        if obi >= 0.35:
            status = 'STRONG_BUY_PRESSURE 🟢'
        elif obi <= -0.35:
            status = 'STRONG_SELL_PRESSURE 🔴'
        else:
            status = 'BALANCED ⚪'

        return {
            'obi': obi,
            'status': status,
            'bid_depth': round(bid_vol, 2),
            'ask_depth': round(ask_vol, 2),
            'ratio': round(bid_vol / max(ask_vol, 1e-4), 2)
        }

    @staticmethod
    def calculate_chandelier_exit(
        df: pd.DataFrame,
        direction: int,
        entry_price: float,
        period: int = 22,
        atr_mult: float = 3.0
    ) -> float:
        """
        Chandelier Exit (Chuck LeBeau):
        - Long: Highest High của N nến gần nhất - (ATR * atr_mult)
        - Short: Lowest Low của N nến gần nhất + (ATR * atr_mult)
        """
        if df is None or len(df) < period + 1:
            return round(entry_price * (0.98 if direction == 1 else 1.02), 2)

        highs = df['high'].iloc[-period:]
        lows = df['low'].iloc[-period:]
        closes = df['close']

        tr = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - closes.shift(1)),
                abs(df['low'] - closes.shift(1))
            )
        )
        atr = float(tr.ewm(span=14, adjust=False).mean().iloc[-1])

        if direction == 1:
            highest_high = float(highs.max())
            stop = highest_high - (atr * atr_mult)
            return round(max(stop, entry_price * 0.95), 2)
        else:
            lowest_low = float(lows.min())
            stop = lowest_low + (atr * atr_mult)
            return round(min(stop, entry_price * 1.05), 2)

    @staticmethod
    def generate_four_stage_exit_plan(
        entry_price: float,
        stop_loss: float,
        direction: int,
        macro_resistance: Optional[float] = None
    ) -> List[ScaledExitTranche]:
        """
        Quy trình thoát lệnh 4 giai đoạn chuẩn mực từ exit-strategies trong claude-trading-skills:
        - Tranche 1 (25%): Chốt lời 1.8R -> Dời ngay SL về Hòa Vốn (Risk-Free Trade)
        - Tranche 2 (25%): Chốt lời 3.0R -> Kích hoạt Chandelier Trailing Stop
        - Tranche 3 (25%): Chốt lời tại Cản Cấu Trúc Macro (SMC Order Block / VWAP Band)
        - Tranche 4 (25%): Moonbag gồng sóng lớn theo nến Tuần/Tháng (1W/1M)
        """
        risk_dist = abs(entry_price - stop_loss)
        if risk_dist <= 0:
            risk_dist = entry_price * 0.01

        tp1 = round(entry_price + (1.8 * risk_dist * direction), 2)
        tp2 = round(entry_price + (3.0 * risk_dist * direction), 2)

        if macro_resistance and ((direction == 1 and macro_resistance > tp2) or (direction == -1 and macro_resistance < tp2)):
            tp3 = round(macro_resistance, 2)
        else:
            tp3 = round(entry_price + (4.5 * risk_dist * direction), 2)

        tp4 = round(entry_price + (7.0 * risk_dist * direction), 2)

        return [
            ScaledExitTranche(
                tranche_id=1,
                size_pct=0.25,
                target_price=tp1,
                trigger_condition=f'Đạt 1.8R (+${abs(tp1 - entry_price):,.1f})',
                action_after='Dời Stop Loss về Hòa Vốn (+0.04% Phí Maker) -> Đưa vị thế về Risk-Free 🛡️'
            ),
            ScaledExitTranche(
                tranche_id=2,
                size_pct=0.25,
                target_price=tp2,
                trigger_condition=f'Đạt 3.0R (+${abs(tp2 - entry_price):,.1f})',
                action_after='Khóa lãi 50% vị thế, kích hoạt Chandelier ATR Trailing Stop 🚀'
            ),
            ScaledExitTranche(
                tranche_id=3,
                size_pct=0.25,
                target_price=tp3,
                trigger_condition=f'Chạm Cản Cấu Trúc Macro (${tp3:,.1f})',
                action_after='Chốt lời chủ động trước vùng thanh khoản cá mập (Supply/Demand Zone)'
            ),
            ScaledExitTranche(
                tranche_id=4,
                size_pct=0.25,
                target_price=tp4,
                trigger_condition=f'Sóng Mở Rộng 1W/1M (${tp4:,.1f})',
                action_after='Moonbag 25% gồng hết biên độ Đại Chu Kỳ Vĩ Mô Tuần/Tháng'
            )
        ]
