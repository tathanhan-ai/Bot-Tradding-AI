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
from dataclasses import dataclass, field, replace
from threading import RLock
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
    vpin: float = 0.0                       # Interpret only when vpin_ready=True.
    vpin_ready: bool = False
    completed_bucket_count: int = 0
    vpin_percentile: float = 0.0           # No historical percentile estimated yet.
    toxicity_regime: str = "WARMUP"
    is_toxic_flow: bool = False             # Cảnh báo khẩn cấp khi VPIN > 0.65

    # 2. Weighted LOB Imbalance
    lob_imbalance_top1: float = 0.0         # Tầng 1 (Best Bid/Ask) [-1.0, 1.0]
    lob_imbalance_5: float = 0.0            # 5 tầng đầu [-1.0, 1.0]
    lob_imbalance_20: float = 0.0           # 20 tầng có trọng số khoảng cách [-1.0, 1.0]
    bid_depth_usdt: float = 0.0             # Tổng giá trị Bid 20 tầng (USDT)
    ask_depth_usdt: float = 0.0             # Tổng giá trị Ask 20 tầng (USDT)
    depth_ready: bool = False
    book_pressure: str = "WARMUP"

    # 3. Market Resilience
    market_resilience_pct: float = 0.0      # Baseline is measured from valid L2 only.
    liquidity_drought_warning: bool = False # Cảnh báo khô cạn thanh khoản khi Resilience < 35%
    avg_sweep_recovery_sec: float = 0.0     # No sweep recovery observed yet.

    # 4. OTT Ratio & Spoofing
    ott_ratio: float = 0.0                  # Order-to-Trade Ratio
    spoofing_detected: bool = False         # Tường thanh khoản ảo bị hủy bất thường
    spoofing_side: str = "NONE"             # 'BID_SPOOF', 'ASK_SPOOF', 'NONE'

    # 5. Composite Microstructure Score & Advice
    microstructure_score: float = 0.0       # [-100, +100] Tín hiệu tổng hợp vi mô
    execution_safety_status: str = "WARMUP"
    hft_rationale: str = "Chờ ít nhất 5 volume buckets hoàn tất và L2 20 tầng hợp lệ."


class VisualHFTMicrostructureEngine:
    def __init__(
        self,
        bucket_size_btc: float = 10.0,
        num_buckets: int = 30,
        toxic_vpin_threshold: float = 0.65,
        min_resilience_threshold: float = 35.0
    ):
        if not math.isfinite(bucket_size_btc) or bucket_size_btc <= 0 or not isinstance(num_buckets, int) or num_buckets < 5:
            raise ValueError("VPIN requires a positive finite bucket size and at least 5 buckets")
        self._lock = RLock()
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
        self.baseline_depth_usdt: float = 0.0               # Measured moving baseline
        self.last_sweep_time: float = 0.0

        # OTT Tracking (Updates vs Trades)
        self.book_update_count: int = 0
        self.trade_count: int = 0
        self.last_ott_calc_time: float = time.time()
        self.current_ott: float = 0.0

        # Cached latest metrics
        self.latest_metrics: VisualHFTMetrics = VisualHFTMetrics()

    def update_trade(self, price: float, qty: float, is_buyer_maker: bool, timestamp: Optional[float] = None) -> VisualHFTMetrics:
        with self._lock:
            return replace(self._update_trade(price, qty, is_buyer_maker, timestamp))

    def _update_trade(self, price: float, qty: float, is_buyer_maker: bool, timestamp: Optional[float]) -> VisualHFTMetrics:
        """
        Updates VPIN volume buckets from incoming trades.
        is_buyer_maker = True -> Maker was buyer -> Aggressor was SELLER (sell volume).
        is_buyer_maker = False -> Maker was seller -> Aggressor was BUYER (buy volume).
        """
        if (not isinstance(is_buyer_maker, bool) or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in (price, qty))
                or not math.isfinite(price * qty) or not math.isfinite(qty / self.bucket_size)):
            return self.latest_metrics
        now = time.time() if timestamp is None else timestamp
        if not isinstance(now, (int, float)) or not math.isfinite(now) or now <= 0:
            return self.latest_metrics
        now = now / 1000 if now > 1e11 else now
        self.trade_count += 1

        remaining_qty = qty
        is_buy = not is_buyer_maker

        while remaining_qty > 0:
            # A large aggTrade may contain many full one-sided buckets. Keep only the measured window.
            if self.current_bucket.total_volume == 0 and remaining_qty >= self.bucket_size:
                full_count, remainder = divmod(remaining_qty, self.bucket_size)
                full_count = int(full_count)
                if remainder >= self.bucket_size * (1 - 1e-12):
                    full_count += 1
                    remainder = 0.0
                start_id = self.bucket_counter + max(0, full_count - self.num_buckets)
                for bucket_id in range(start_id, self.bucket_counter + full_count):
                    self.completed_buckets.append(VolumeBucket(bucket_id, self.bucket_size if is_buy else 0,
                        0 if is_buy else self.bucket_size, now, now, True))
                self.completed_buckets = self.completed_buckets[-self.num_buckets:]
                self.bucket_counter += full_count
                self.current_bucket = VolumeBucket(bucket_id=self.bucket_counter, start_time=now)
                remaining_qty = remainder
                continue
            current_fill = self.current_bucket.total_volume
            space_left = max(0.0, self.bucket_size - current_fill)
            consumed = min(remaining_qty, space_left)
            if is_buy:
                self.current_bucket.buy_volume += consumed
            else:
                self.current_bucket.sell_volume += consumed
            remaining_qty -= consumed
            if self.current_bucket.total_volume >= self.bucket_size * (1 - 1e-12):
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
        with self._lock:
            return replace(self._update_order_book(bids, asks, timestamp))

    def _update_order_book(self, bids, asks, timestamp=None) -> VisualHFTMetrics:
        """
        Updates 20-Level Weighted LOB Imbalance, Market Resilience, and OTT Ratio.
        bids, asks: lists of [price, qty] sorted best to worst.
        """
        now = time.time() if timestamp is None else timestamp
        try:
            bids = [(float(p), float(q)) for p, q in bids[:20]]
            asks = [(float(p), float(q)) for p, q in asks[:20]]
            valid = (isinstance(now, (int, float)) and math.isfinite(now) and now > 0 and len(bids) == len(asks) == 20
                and all(math.isfinite(v) and v > 0 for level in bids + asks for v in level)
                and all(math.isfinite(p * q) for p, q in bids + asks)
                and math.isfinite(sum(q for _, q in bids + asks)) and math.isfinite(sum(p * q for p, q in bids + asks))
                and all(bids[i][0] > bids[i + 1][0] and asks[i][0] < asks[i + 1][0] for i in range(19))
                and bids[0][0] < asks[0][0])
        except (TypeError, ValueError, OverflowError):
            valid = False
        self.latest_metrics.depth_ready = valid
        if not valid:
            return self._recompute_metrics()
        now = now / 1000 if now > 1e11 else now
        self.book_update_count += 1

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

        if self.depth_history:
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
            trades_sec = self.trade_count / elapsed
            self.current_ott = round(updates_sec / trades_sec, 1) if trades_sec > 0 else 0.0
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
        m.completed_bucket_count = len(self.completed_buckets)
        m.vpin_ready = m.completed_bucket_count >= 5
        if m.vpin_ready:
            total_abs_imbalance = sum(b.absolute_imbalance for b in self.completed_buckets)
            denom = len(self.completed_buckets) * self.bucket_size
            raw_vpin = total_abs_imbalance / denom
            m.vpin = round(min(1.0, max(0.0, raw_vpin)), 3)
        else:
            m.vpin = 0.0

        # Toxicity Regime
        if not m.vpin_ready:
            m.toxicity_regime = "WARMUP"
            m.is_toxic_flow = False
        elif m.vpin >= self.toxic_vpin_threshold:
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
        if not m.depth_ready:
            m.book_pressure = "WARMUP"
        elif imb >= 0.40:
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
        elif not m.vpin_ready or not m.depth_ready:
            m.execution_safety_status = "WARMUP"
            base_score = 0.0
            m.hft_rationale = f"Chờ dữ liệu đo: {m.completed_bucket_count}/5 VPIN buckets; L2 20 tầng {'đủ' if m.depth_ready else 'thiếu/không hợp lệ'}."
        elif m.liquidity_drought_warning:
            m.execution_safety_status = "CAUTION_SLIPPAGE"
            m.hft_rationale = f"💧 KHÔ CẠN THANH KHOẢN (Resilience={m.market_resilience_pct:.0f}%): Độ sâu sổ lệnh bị lõm sau cú quét. Cẩn thận trượt giá sàn."
        else:
            m.execution_safety_status = "SAFE"
            m.hft_rationale = f"✅ THANH KHOẢN LÀNH MẠNH: VPIN={m.vpin:.2f} (Lành mạnh), Áp lực 20 tầng: {m.book_pressure} ({m.lob_imbalance_20:+.2f})."

        m.microstructure_score = round(max(-100.0, min(100.0, base_score)), 1)
        return m

    def get_metrics(self) -> VisualHFTMetrics:
        with self._lock:
            return replace(self.latest_metrics)
