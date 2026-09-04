# -*- coding: utf-8 -*-
"""
VisualHFT Market Microstructure Engine (Trích xuất từ VisualHFT C#/.NET Core)
Bao gồm 4 giải thuật vi mô tần số cao (HFT) cốt lõi:
1. VPIN (Volume-Synchronized Probability of Informed Trading - Easley, Lopez de Prado, O'Hara 2012)
   Đo lường xác suất xuất hiện dòng tiền độc hại (toxic flow / insider) theo volume buckets.
2. 20-Level Weighted LOB Imbalance (Bất đối xứng sổ lệnh 20 tầng có trọng số phân rã)
3. Market Resilience (Độ đàn hồi & Tốc độ tái tạo thanh khoản sau cú quét sổ lệnh)
4. OTT Ratio & Spoofing Detector (Tỷ lệ lệnh đặt/hủy so với khớp lệnh)
"""
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
import time
import math


@dataclass
class VolumeBucket:
    bucket_id: int
    buy_volume: float = 0.0
    sell_volume: float = 0.0
    start_time: float = 0.0
    end_time: float = 0.0
    is_completed: bool = False

    @property
    def total_volume(self) -> float:
        return self.buy_volume + self.sell_volume

    @property
    def absolute_imbalance(self) -> float:
        return abs(self.buy_volume - self.sell_volume)


@dataclass
class VisualHFTMetrics:
    # 1. VPIN
    vpin: float = 0.35                      # [0.0, 1.0]
    vpin_percentile: float = 40.0           # Phân vị lịch sử (0-100%)
    toxicity_regime: str = "CLEAN"          # 'CLEAN' (<0.45), 'MODERATE' (0.45-0.65), 'TOXIC_INFORMED' (>0.65)
    is_toxic_flow: bool = False             # Cảnh báo khẩn cấp khi VPIN > 0.65

    # 2. Weighted LOB Imbalance
    lob_imbalance_top1: float = 0.0         # Tầng 1 (Best Bid/Ask) [-1.0, 1.0]
    lob_imbalance_5: float = 0.0            # 5 tầng đầu [-1.0, 1.0]
    lob_imbalance_20: float = 0.0           # 20 tầng có trọng số khoảng cách [-1.0, 1.0]
    bid_depth_usdt: float = 0.0             # Tổng giá trị Bid 20 tầng (USDT)
    ask_depth_usdt: float = 0.0             # Tổng giá trị Ask 20 tầng (USDT)
    book_pressure: str = "NEUTRAL"          # 'STRONG_BUY_PRESSURE', 'BUY_PRESSURE', 'NEUTRAL', 'SELL_PRESSURE', 'STRONG_SELL_PRESSURE'

    # 3. Market Resilience
    market_resilience_pct: float = 85.0     # [0, 100]% Tỷ lệ hồi phục thanh khoản sau cú quét
    liquidity_drought_warning: bool = False # Cảnh báo khô cạn thanh khoản khi Resilience < 35%
    avg_sweep_recovery_sec: float = 1.2     # Thời gian trung bình hồi phục độ sâu

    # 4. OTT Ratio & Spoofing
    ott_ratio: float = 12.0                 # Order-to-Trade Ratio
    spoofing_detected: bool = False         # Tường thanh khoản ảo bị hủy bất thường
    spoofing_side: str = "NONE"             # 'BID_SPOOF', 'ASK_SPOOF', 'NONE'

    # 5. Composite Microstructure Score & Advice
    microstructure_score: float = 0.0       # [-100, +100] Tín hiệu tổng hợp vi mô
    execution_safety_status: str = "SAFE"   # 'SAFE', 'CAUTION_SLIPPAGE', 'CRITICAL_TOXIC_AVOID'
    hft_rationale: str = ""


class VisualHFTMicrostructureEngine:
    def __init__(
        self,
        bucket_size_btc: float = 10.0,
        num_buckets: int = 30,
        toxic_vpin_threshold: float = 0.65,
        min_resilience_threshold: float = 35.0
    ):
        self.bucket_size = bucket_size_btc
        self.num_buckets = num_buckets
        self.toxic_vpin_threshold = toxic_vpin_threshold
        self.min_resilience_threshold = min_resilience_threshold

        # Volume Buckets for VPIN
        self.completed_buckets: List[VolumeBucket] = []
        self.current_bucket = VolumeBucket(bucket_id=1, start_time=time.time())
        self.bucket_counter = 1

        # LOB Depth baseline history for Market Resilience
        self.depth_history: List[Tuple[float, float]] = []  # (timestamp, total_20_depth)
        self.baseline_depth_usdt: float = 500_000.0         # Moving baseline depth
        self.last_sweep_time: float = 0.0

        # OTT Tracking (Updates vs Trades)
        self.book_update_count: int = 0
        self.trade_count: int = 0
        self.last_ott_calc_time: float = time.time()
        self.current_ott: float = 8.0

        # Cached latest metrics
        self.latest_metrics: VisualHFTMetrics = VisualHFTMetrics()

    def update_trade(self, price: float, qty: float, is_buyer_maker: bool, timestamp: Optional[float] = None) -> VisualHFTMetrics:
        """
        Updates VPIN volume buckets from incoming trades.
        is_buyer_maker = True -> Maker was buyer -> Aggressor was SELLER (sell volume).
        is_buyer_maker = False -> Maker was seller -> Aggressor was BUYER (buy volume).
        """
        now = timestamp or time.time()
        self.trade_count += 1

        # If price indicates different asset size, adapt bucket size dynamically
        if price > 20000 and self.bucket_size < 1.0:
            self.bucket_size = 10.0  # BTC default 10 BTC

        remaining_qty = qty
        is_buy = not is_buyer_maker

        while remaining_qty > 0:
            current_fill = self.current_bucket.total_volume
            space_left = max(0.0, self.bucket_size - current_fill)

            if remaining_qty <= space_left:
                if is_buy:
                    self.current_bucket.buy_volume += remaining_qty
                else:
                    self.current_bucket.sell_volume += remaining_qty
                remaining_qty = 0.0
            else:
                # Fill current bucket to exactly bucket_size
                if is_buy:
                    self.current_bucket.buy_volume += space_left
                else:
                    self.current_bucket.sell_volume += space_left
                remaining_qty -= space_left

                # Finalize bucket
                self.current_bucket.end_time = now
                self.current_bucket.is_completed = True
                self.completed_buckets.append(self.current_bucket)

                # Maintain sliding window of N completed buckets
                if len(self.completed_buckets) > self.num_buckets:
                    self.completed_buckets.pop(0)

                # Create next bucket
                self.bucket_counter += 1
                self.current_bucket = VolumeBucket(bucket_id=self.bucket_counter, start_time=now)

        return self._recompute_metrics()

    def update_order_book(
        self,
        bids: List[List[float]],
        asks: List[List[float]],
        timestamp: Optional[float] = None
    ) -> VisualHFTMetrics:
        """
        Updates 20-Level Weighted LOB Imbalance, Market Resilience, and OTT Ratio.
        bids, asks: lists of [price, qty] sorted best to worst.
        """
        now = timestamp or time.time()
        self.book_update_count += 1

        if not bids or not asks:
            return self.latest_metrics

        # 1. Tầng 1 (Best Bid/Ask)
        best_bid_qty = bids[0][1] if len(bids) > 0 else 0.0
        best_ask_qty = asks[0][1] if len(asks) > 0 else 0.0
        tot_top1 = best_bid_qty + best_ask_qty
        top1_imbalance = (best_bid_qty - best_ask_qty) / tot_top1 if tot_top1 > 0 else 0.0

        # 2. 5 Tầng đầu
        top5_bids = sum(b[1] for b in bids[:5])
        top5_asks = sum(a[1] for a in asks[:5])
        tot_top5 = top5_bids + top5_asks
        top5_imbalance = (top5_bids - top5_asks) / tot_top5 if tot_top5 > 0 else 0.0

        # 3. 20 Tầng có trọng số khoảng cách (Distance Decay: w_i = 1 / sqrt(i))
        weighted_bid_sum = 0.0
        weighted_ask_sum = 0.0
        total_bid_usdt = 0.0
        total_ask_usdt = 0.0

        for i in range(min(20, max(len(bids), len(asks)))):
            weight = 1.0 / math.sqrt(i + 1)
            if i < len(bids):
                b_price, b_qty = bids[i][0], bids[i][1]
                weighted_bid_sum += b_qty * weight
                total_bid_usdt += b_price * b_qty
            if i < len(asks):
                a_price, a_qty = asks[i][0], asks[i][1]
                weighted_ask_sum += a_qty * weight
                total_ask_usdt += a_price * a_qty

        tot_weighted = weighted_bid_sum + weighted_ask_sum
        lob_20_imbalance = (weighted_bid_sum - weighted_ask_sum) / tot_weighted if tot_weighted > 0 else 0.0

        # 4. Market Resilience Calculation
        total_20_depth_usdt = total_bid_usdt + total_ask_usdt
        self.depth_history.append((now, total_20_depth_usdt))
        # Keep last 60 seconds
        self.depth_history = [d for d in self.depth_history if (now - d[0]) <= 60.0]

        if len(self.depth_history) >= 10:
            recent_depths = [d[1] for d in self.depth_history]
            self.baseline_depth_usdt = sum(recent_depths) / len(recent_depths)

        resilience_pct = 100.0
        if self.baseline_depth_usdt > 0:
            resilience_pct = min(100.0, max(5.0, (total_20_depth_usdt / self.baseline_depth_usdt) * 100.0))

        # Check for sharp sweep
        if resilience_pct < 50.0 and (now - self.last_sweep_time) > 10.0:
            self.last_sweep_time = now

        # 5. Order-to-Trade Ratio (OTT)
        if (now - self.last_ott_calc_time) >= 3.0:
            elapsed = max(1.0, now - self.last_ott_calc_time)
            updates_sec = self.book_update_count / elapsed
            trades_sec = max(0.2, self.trade_count / elapsed)
            self.current_ott = round(updates_sec / trades_sec, 1)
            self.book_update_count = 0
            self.trade_count = 0
            self.last_ott_calc_time = now

        # 6. Spoofing Detection
        is_spoof = False
        spoof_side = "NONE"
        if self.current_ott > 40.0:
            if lob_20_imbalance > 0.60:
                is_spoof = True
                spoof_side = "BID_SPOOF"
            elif lob_20_imbalance < -0.60:
                is_spoof = True
                spoof_side = "ASK_SPOOF"

        # Update cache
        self.latest_metrics.lob_imbalance_top1 = round(top1_imbalance, 3)
        self.latest_metrics.lob_imbalance_5 = round(top5_imbalance, 3)
        self.latest_metrics.lob_imbalance_20 = round(lob_20_imbalance, 3)
        self.latest_metrics.bid_depth_usdt = round(total_bid_usdt, 2)
        self.latest_metrics.ask_depth_usdt = round(total_ask_usdt, 2)
        self.latest_metrics.market_resilience_pct = round(resilience_pct, 1)
        self.latest_metrics.liquidity_drought_warning = (resilience_pct < self.min_resilience_threshold)
        self.latest_metrics.ott_ratio = self.current_ott
        self.latest_metrics.spoofing_detected = is_spoof
        self.latest_metrics.spoofing_side = spoof_side

        return self._recompute_metrics()

    def _recompute_metrics(self) -> VisualHFTMetrics:
        """Computes current VPIN across completed buckets and final composite score."""
        m = self.latest_metrics

        # Compute VPIN
        if len(self.completed_buckets) >= 5:
            total_abs_imbalance = sum(b.absolute_imbalance for b in self.completed_buckets)
            denom = len(self.completed_buckets) * self.bucket_size
            raw_vpin = total_abs_imbalance / denom if denom > 0 else 0.35
            m.vpin = round(min(0.99, max(0.01, raw_vpin)), 3)
        else:
            if self.current_bucket.total_volume > 0:
                m.vpin = round(min(0.99, max(0.15, self.current_bucket.absolute_imbalance / self.current_bucket.total_volume)), 3)
            else:
                m.vpin = 0.35

        # Toxicity Regime
        if m.vpin >= self.toxic_vpin_threshold:
            m.toxicity_regime = "TOXIC_INFORMED"
            m.is_toxic_flow = True
        elif m.vpin >= 0.45:
            m.toxicity_regime = "MODERATE"
            m.is_toxic_flow = False
        else:
            m.toxicity_regime = "CLEAN"
            m.is_toxic_flow = False

        # Book Pressure label
        imb = m.lob_imbalance_20
        if imb >= 0.40:
            m.book_pressure = "STRONG_BUY_PRESSURE"
        elif imb >= 0.15:
            m.book_pressure = "BUY_PRESSURE"
        elif imb <= -0.40:
            m.book_pressure = "STRONG_SELL_PRESSURE"
        elif imb <= -0.15:
            m.book_pressure = "SELL_PRESSURE"
        else:
            m.book_pressure = "NEUTRAL"

        # Composite Microstructure Score [-100, +100]
        base_score = (m.lob_imbalance_20 * 65.0) + (m.lob_imbalance_top1 * 35.0)
        
        if m.is_toxic_flow:
            m.execution_safety_status = "CRITICAL_TOXIC_AVOID"
            base_score *= 0.5
            m.hft_rationale = f"⚠️ TOXIC FLOW VPIN={m.vpin:.2f} (> {self.toxic_vpin_threshold}): Cá mập/Insiders đang quét thanh khoản mạnh! Đề xuất hoãn lệnh Taker."
        elif m.liquidity_drought_warning:
            m.execution_safety_status = "CAUTION_SLIPPAGE"
            m.hft_rationale = f"💧 KHÔ CẠN THANH KHOẢN (Resilience={m.market_resilience_pct:.0f}%): Độ sâu sổ lệnh bị lõm sau cú quét. Cẩn thận trượt giá sàn."
        else:
            m.execution_safety_status = "SAFE"
            m.hft_rationale = f"✅ THANH KHOẢN LÀNH MẠNH: VPIN={m.vpin:.2f} (Lành mạnh), Áp lực 20 tầng: {m.book_pressure} ({m.lob_imbalance_20:+.2f})."

        m.microstructure_score = round(max(-100.0, min(100.0, base_score)), 1)
        return m

    def get_metrics(self) -> VisualHFTMetrics:
        return self.latest_metrics
