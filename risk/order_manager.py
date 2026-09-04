"""
Advanced Institutional Order Execution Engine for Binance Futures
Supports 7 specialized order types:
1. LIMIT - Giá giới hạn thông thường (Maker 0.02%)
2. POST_ONLY - Chỉ khớp lệnh Maker, tự động hủy nếu bị khớp Taker (bảo vệ 60% phí sàn)
3. MARKET - Lệnh thị trường khớp tức thì tại Best Bid/Ask (Taker 0.05%)
4. CONDITIONAL - Lệnh có điều kiện kích hoạt (Stop Market / Stop Limit đón bứt phá hoặc sập)
5. TRAILING_STOP - Lệnh đeo bám đỉnh/đáy động theo tỷ lệ Callback %
6. TWAP - Time-Weighted Average Price (chia nhỏ khối lượng theo thời gian giảm trượt giá)
7. SCALE_RATIO - Đặt lệnh theo tỷ lệ bậc thang (DCA thang giá 20% - 30% - 50% tối ưu vị thế)
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Tuple


@dataclass
class FuturesOrder:
    order_id: int
    symbol: str
    order_type: str        # 'LIMIT', 'POST_ONLY', 'MARKET', 'CONDITIONAL', 'TRAILING_STOP', 'TWAP', 'SCALE_RATIO'
    side: str              # 'BUY' (Long), 'SELL' (Short)
    direction: int         # 1 for Long, -1 for Short
    price: float           # Limit / Execution price
    margin: float          # Allocated USDT margin
    leverage: int
    units: float
    status: str            # 'PENDING', 'ACTIVE', 'FILLED', 'CANCELED', 'REJECTED'
    created_at: str
    timeframe: str = "15m"
    
    # Conditional / Stop fields
    trigger_price: float = 0.0
    trigger_condition: str = "ABOVE"  # 'ABOVE' or 'BELOW'
    
    # Trailing Stop fields
    callback_pct: float = 0.8
    peak_price: float = 0.0
    
    # TWAP fields
    twap_total_slices: int = 1
    twap_filled_slices: int = 0
    twap_interval_ticks: int = 6       # Số tick giữa các lần khớp (~3s)
    twap_tick_counter: int = 0
    
    # Scale Ratio fields
    scale_level: int = 1               # Tầng 1, 2, 3
    scale_ratio_pct: float = 100.0     # % tỷ lệ vốn
    
    # Meta
    stop_loss: float = 0.0
    take_profit: float = 0.0
    note: str = ""
    cancel_reason: str = ""
    cancelled_at: str = ""
    age_ticks: int = 0


class OrderQueueManager:
    def __init__(self):
        self.orders: List[FuturesOrder] = []
        self._next_id = 1

    def place_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        price: float,
        margin: float,
        leverage: int,
        trigger_price: float = 0.0,
        trigger_condition: str = "ABOVE",
        callback_pct: float = 0.8,
        twap_slices: int = 5,
        twap_interval_ticks: int = 6,
        stop_loss: float = 0.0,
        take_profit: float = 0.0,
        note: str = "",
        best_bid: float = 0.0,
        best_ask: float = 0.0,
        timeframe: str = "15m"
    ) -> Tuple[Optional[FuturesOrder], str]:
        """
        Validates and places any of the 7 order types.
        Returns (order, message).
        """
        direction = 1 if side.upper() == "BUY" else -1
        order_type_clean = order_type.upper()
        target_price = price if price > 0 else (best_ask if direction == 1 else best_bid)

        # 1. Validate POST_ONLY (Guaranteed Maker protection)
        if order_type_clean == "POST_ONLY":
            if direction == 1 and best_ask > 0 and price >= best_ask:
                return None, f"❌ BỊ TỪ CHỐI BỞI POST-ONLY: Giá mua ${price:,.2f} >= Best Ask ${best_ask:,.2f} (Lệnh sẽ bị khớp Taker 0.05%). Đã bảo vệ phí Maker cho bạn!"
            elif direction == -1 and best_bid > 0 and price <= best_bid:
                return None, f"❌ BỊ TỪ CHỐI BỞI POST-ONLY: Giá bán ${price:,.2f} <= Best Bid ${best_bid:,.2f} (Lệnh sẽ bị khớp Taker 0.05%). Đã bảo vệ phí Maker cho bạn!"

        # 2. Handle SCALE_RATIO (Automatic 3-stage ladder scaling 20% - 30% - 50%)
        if order_type_clean == "SCALE_RATIO":
            ratios = [0.20, 0.30, 0.50]
            step_offsets = [0.0, 0.005, 0.010]  # 0%, 0.5%, 1.0% deeper into support/resistance
            created_orders = []
            
            for lvl, (r, offset) in enumerate(zip(ratios, step_offsets), 1):
                lvl_price = price * (1.0 - offset) if direction == 1 else price * (1.0 + offset)
                lvl_margin = margin * r
                lvl_notional = lvl_margin * leverage
                lvl_units = round(lvl_notional / lvl_price, 4) if lvl_price > 0 else 0.0
                
                order = FuturesOrder(
                    order_id=self._next_id,
                    symbol=symbol,
                    order_type="SCALE_RATIO",
                    side=side.upper(),
                    direction=direction,
                    price=round(lvl_price, 2),
                    margin=round(lvl_margin, 2),
                    leverage=leverage,
                    units=lvl_units,
                    status="PENDING",
                    created_at=datetime.now().strftime("%H:%M:%S"),
                    timeframe=timeframe,
                    scale_level=lvl,
                    scale_ratio_pct=round(r * 100, 1),
                    stop_loss=round(stop_loss, 2),
                    take_profit=round(take_profit, 2),
                    note=f"Tầng {lvl}/3 ({round(r*100)}% vốn)"
                )
                self._next_id += 1
                self.orders.append(order)
                created_orders.append(order)
                
            return created_orders[0], f"Đã rải 3 tầng lệnh thang tỷ lệ 20% - 30% - 50% quanh ${price:,.2f} ({timeframe})"

        # Standard single order placement
        notional = margin * leverage
        units = round(notional / target_price, 4) if target_price > 0 else 0.0

        order = FuturesOrder(
            order_id=self._next_id,
            symbol=symbol,
            order_type=order_type_clean,
            side=side.upper(),
            direction=direction,
            price=round(target_price, 2),
            margin=round(margin, 2),
            leverage=leverage,
            units=units,
            status="PENDING",
            created_at=datetime.now().strftime("%H:%M:%S"),
            timeframe=timeframe,
            trigger_price=round(trigger_price, 2),
            trigger_condition=trigger_condition.upper(),
            callback_pct=callback_pct,
            peak_price=target_price,
            twap_total_slices=max(1, twap_slices) if order_type_clean == "TWAP" else 1,
            twap_filled_slices=0,
            twap_interval_ticks=twap_interval_ticks,
            stop_loss=round(stop_loss, 2),
            take_profit=round(take_profit, 2),
            note=note
        )
        self._next_id += 1
        self.orders.append(order)
        return order, f"Đã đặt lệnh {order_type_clean} #{order.order_id} thành công"

    def cancel_order(self, order_id: int) -> bool:
        for o in self.orders:
            if o.order_id == order_id and o.status in ("PENDING", "ACTIVE"):
                o.status = "CANCELED"
                return True
        return False

    def update_order(
        self,
        order_id: int,
        price: Optional[float] = None,
        units: Optional[float] = None,
        margin: Optional[float] = None,
        timeframe: Optional[str] = None,
        order_type: Optional[str] = None,
        side: Optional[str] = None,
        trigger_price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        leverage: Optional[int] = None,
        callback_pct: Optional[float] = None
    ) -> Tuple[bool, str]:
        """Modifies parameters of an active pending order."""
        for o in self.orders:
            if o.order_id == order_id and o.status in ("PENDING", "ACTIVE"):
                if price is not None and price > 0:
                    o.price = round(price, 2)
                if leverage is not None and leverage > 0:
                    o.leverage = int(leverage)
                if timeframe:
                    o.timeframe = timeframe.strip().lower()
                if order_type:
                    o.order_type = order_type.strip().upper()
                if side:
                    s_clean = side.strip().upper()
                    if s_clean in ("BUY", "SELL"):
                        o.side = s_clean
                        o.direction = 1 if s_clean == "BUY" else -1
                if trigger_price is not None and trigger_price > 0:
                    o.trigger_price = round(trigger_price, 2)
                if stop_loss is not None:
                    o.stop_loss = round(stop_loss, 2)
                if take_profit is not None:
                    o.take_profit = round(take_profit, 2)
                if callback_pct is not None and callback_pct > 0:
                    o.callback_pct = round(callback_pct, 2)

                # Sync units and margin
                if units is not None and units > 0:
                    o.units = round(units, 4)
                    if o.price > 0 and o.leverage > 0:
                        o.margin = round((o.units * o.price) / o.leverage, 2)
                elif margin is not None and margin > 0:
                    o.margin = round(margin, 2)
                    if o.price > 0 and o.leverage > 0:
                        o.units = round((o.margin * o.leverage) / o.price, 4)
                elif price is not None and price > 0:
                    if o.margin > 0 and o.leverage > 0:
                        o.units = round((o.margin * o.leverage) / o.price, 4)

                # Reset age ticks so order isn't immediately aged out
                o.age_ticks = 0
                o.note = f"Đã cập nhật thủ công lúc {datetime.now().strftime('%H:%M:%S')}"
                return True, f"Đã cập nhật lệnh chờ #{order_id} thành công!"
        return False, f"Không tìm thấy lệnh chờ #{order_id} hoặc lệnh không ở trạng thái hoạt động!"

    def force_execute_order(self, order_id: int, execution_price: float) -> Optional[FuturesOrder]:
        """Manually forces immediate market execution of a pending order"""
        for o in self.orders:
            if o.order_id == order_id and o.status in ("PENDING", "ACTIVE"):
                o.status = "FILLED"
                o.note = f"THỦ CÔNG: VÀO LỆNH NGAY ⚡ (Khớp Market ${execution_price:,.1f})"
                return o
        return None

    def evaluate_and_clean_unsuitable_orders(
        self,
        current_price: float,
        indicators: Optional[dict] = None,
        ai_verdict: Optional[Any] = None,
        ensemble_result: Optional[Any] = None,
        active_timeframe: str = "15m"
    ) -> List[FuturesOrder]:
        """
        Institutional AI Order Invalidation & Self-Cleaning Engine:
        Continuously audits all pending orders on every single price tick.
        Calibrated to match multi-timeframe candle cycles and market microstructure:
        1. Directional Shift: Order opposes Strong Ensemble consensus (|score| >= 40.0)
        2. Structural S/R Migration: Support moved up far above Buy Limit, or Resistance moved down far below Sell Limit
        3. Breakout Invalidation: Price breached the level with momentum against the order (chống bắt dao rơi / chặn đầu xe tải)
        4. Runaway Drift: Price rallied/dumped in the target direction far away (> 2.0% or 2.0x ATR) without retesting
        5. Timeframe-Aware Expiration: Never prematurely cancels in 30s. Only expires after 2-3 full candle cycles of the active timeframe!
        """
        cancelled = []
        indicators = indicators or {}
        atr = indicators.get("atr") or (current_price * 0.008)

        # Multi-timeframe candle lengths in sub-second ticks (~500ms / tick = 2 ticks per second)
        TF_TICKS_PER_CANDLE = {
            "1m": 120,      # 1 minute = 120 ticks
            "5m": 600,      # 5 minutes = 600 ticks
            "15m": 1800,    # 15 minutes = 1,800 ticks
            "1h": 7200,     # 60 minutes = 7,200 ticks
        }

        for o in self.orders:
            if o.status not in ("PENDING", "ACTIVE"):
                continue

            o.age_ticks = getattr(o, "age_ticks", 0) + 1
            dist_pct = abs(current_price - o.price) / current_price * 100.0 if current_price > 0 else 0.0
            o_tf = getattr(o, "timeframe", active_timeframe) or active_timeframe
            candle_ticks = TF_TICKS_PER_CANDLE.get(o_tf, 1800)

            cancel_reason = None

            # Rule 1: Ensemble Trend Invalidation (Đảo chiều xu hướng đồng thuận)
            if ensemble_result:
                score = ensemble_result.consensus_score
                if o.side == "BUY" and score <= -40.0:
                    cancel_reason = f"ĐỒNG THUẬN ĐẢO CHIỀU 🔄 (Ensemble chuyển sang Short {score:.1f}đ trong khung {o_tf}, hủy Buy #{o.order_id})"
                elif o.side == "SELL" and score >= 40.0:
                    cancel_reason = f"ĐỒNG THUẬN ĐẢO CHIỀU 🔄 (Ensemble chuyển sang Long +{score:.1f}đ trong khung {o_tf}, hủy Sell #{o.order_id})"

            # Rule 2: Structural Level Invalidation (Support/Resistance Shift & Breakout)
            if not cancel_reason and ai_verdict:
                sup = ai_verdict.support_price
                res = ai_verdict.resistance_price

                # Buy Limit invalidation
                if o.side == "BUY" and sup > 0:
                    if (sup - o.price) > (1.2 * atr):
                        cancel_reason = f"HỖ TRỢ DỊCH CHUYỂN 📈 (Hỗ trợ mới ${sup:,.1f} đã dâng cao hơn giá đặt ${o.price:,.1f}, hủy để tái định vị)"
                    elif current_price < (o.price - 1.5 * atr):
                        cancel_reason = f"THỦNG HỖ TRỢ 📉 (Giá đâm thủng cản ${o.price:,.1f}, hủy Long để chống bắt dao rơi)"

                # Sell Limit invalidation
                elif o.side == "SELL" and res > 0:
                    if (o.price - res) > (1.2 * atr):
                        cancel_reason = f"KHÁNG CỰ DỊCH CHUYỂN 📉 (Kháng cự mới ${res:,.1f} đã hạ thấp hơn giá đặt ${o.price:,.1f}, hủy để tái định vị)"
                    elif current_price > (o.price + 1.5 * atr):
                        cancel_reason = f"VƯỢT KHÁNG CỰ 📈 (Giá bứt phá qua cản ${o.price:,.1f}, hủy Short để chống chặn đầu xe tải)"

            # Rule 3: Intelligent Timeframe Expiration & Runaway Drift
            # NOTE: Never cancel CONDITIONAL or TRAILING_STOP by timer (they are ambush triggers)
            if not cancel_reason and o.order_type not in ("CONDITIONAL", "TRAILING_STOP"):
                if o.order_type == "POST_ONLY":
                    # Scalp Maker: must maintain tight spread edge. If drifted > 0.8% and aged > 60s (120 ticks), refresh
                    if dist_pct >= 0.8 and o.age_ticks >= 120:
                        cancel_reason = f"TRÔI SỔ LỆNH MAKER ⏱️ (Lệch {dist_pct:.2f}% > 60s, hủy để tái lập vị thế sát Best Bid/Ask)"

                elif o.order_type in ("LIMIT", "SCALE_RATIO"):
                    # Case 3A: Runaway Drift (Sóng đã đi xa mà không hồi về điểm đón)
                    runaway_patience_ticks = int(candle_ticks * 1.5)  # e.g. 22.5 mins on 15m
                    if dist_pct >= 2.0 and o.age_ticks >= runaway_patience_ticks:
                        cancel_reason = f"SÓNG ĐÃ RỜI ĐI 🚀 (Giá chạy xa +{dist_pct:.2f}% khỏi cản trong khung {o_tf}, hủy tránh khớp trễ)"
                    # Case 3B: Multi-Candle Timeout (Chờ đủ 3 chu kỳ nến mà không có sóng)
                    elif o.age_ticks >= (candle_ticks * 3):
                        candles_passed = round(o.age_ticks / candle_ticks, 1)
                        cancel_reason = f"HẾT HẠN CHU KỲ NẾN ⏱️ (Đã chờ {candles_passed} nến {o_tf} không khớp, hủy giải phóng vốn)"

            # Execute cancellation if invalidated
            if cancel_reason:
                o.status = "CANCELED"
                o.cancel_reason = cancel_reason
                o.cancelled_at = datetime.now().strftime("%H:%M:%S")
                o.note = cancel_reason
                cancelled.append(o)
                print(f"🧹 [AI TỰ ĐỘNG HỦY LỆNH] #{o.order_id} {o.side} {o.order_type} [{o_tf}] tại ${o.price:,.1f} | Lý do: {cancel_reason}", flush=True)

        return cancelled

    def clean_stale_orders(self, current_price: float, max_dist_pct: float = 1.5, max_ticks: int = 50):
        return self.evaluate_and_clean_unsuitable_orders(current_price)

    def match_orders(self, current_price: float, best_bid: float = 0.0, best_ask: float = 0.0) -> List[FuturesOrder]:
        """
        Evaluates all pending orders on live sub-second price ticks:
        1. LIMIT & POST_ONLY: Fills when market crosses limit price
        2. CONDITIONAL: Triggers when price crosses trigger threshold, turns into Market fill
        3. TRAILING_STOP: Updates peak/trough; triggers when pullback >= callback_pct
        4. TWAP: Fires 1 micro-slice every interval until total slices completed
        5. SCALE_RATIO: Fills individual ladder steps as price reaches them
        """
        filled_orders = []

        for o in self.orders:
            if o.status not in ("PENDING", "ACTIVE"):
                continue

            # -------------------------------------------------------------
            # 1. LIMIT, POST_ONLY & SCALE_RATIO Matching
            # -------------------------------------------------------------
            if o.order_type in ("LIMIT", "POST_ONLY", "SCALE_RATIO"):
                if o.side == "BUY" and current_price <= o.price:
                    o.status = "FILLED"
                    filled_orders.append(o)
                elif o.side == "SELL" and current_price >= o.price:
                    o.status = "FILLED"
                    filled_orders.append(o)

            # -------------------------------------------------------------
            # 2. CONDITIONAL (Lệnh Có Điều Kiện / Stop Trigger)
            # -------------------------------------------------------------
            elif o.order_type == "CONDITIONAL":
                triggered = False
                if o.trigger_condition == "ABOVE" and current_price >= o.trigger_price:
                    triggered = True
                elif o.trigger_condition == "BELOW" and current_price <= o.trigger_price:
                    triggered = True

                if triggered:
                    o.status = "FILLED"
                    o.price = current_price
                    o.note = f"Kích hoạt tại ${current_price:,.2f} (Ngưỡng {o.trigger_price:,.2f})"
                    filled_orders.append(o)

            # -------------------------------------------------------------
            # 3. TRAILING_STOP (Lệnh Đeo Bám Đỉnh/Đáy)
            # -------------------------------------------------------------
            elif o.order_type == "TRAILING_STOP":
                if o.peak_price == 0.0:
                    o.peak_price = current_price

                if o.side == "BUY":
                    # Buy Trailing (Bắt đáy đảo chiều lên): theo dõi đáy thấp nhất
                    if current_price < o.peak_price:
                        o.peak_price = current_price  # New trough
                    # Pullback up from trough by callback_pct
                    pullback_pct = ((current_price - o.peak_price) / o.peak_price) * 100.0
                    if pullback_pct >= o.callback_pct:
                        o.status = "FILLED"
                        o.price = current_price
                        o.note = f"Trailing Buy khớp sau nhịp hồi +{pullback_pct:.2f}% từ đáy ${o.peak_price:,.2f}"
                        filled_orders.append(o)

                else:
                    # Sell Trailing (Bắt đỉnh đảo chiều xuống): theo dõi đỉnh cao nhất
                    if current_price > o.peak_price:
                        o.peak_price = current_price  # New peak
                    # Pullback down from peak by callback_pct
                    pullback_pct = ((o.peak_price - current_price) / o.peak_price) * 100.0
                    if pullback_pct >= o.callback_pct:
                        o.status = "FILLED"
                        o.price = current_price
                        o.note = f"Trailing Sell khớp sau nhịp xả -{pullback_pct:.2f}% từ đỉnh ${o.peak_price:,.2f}"
                        filled_orders.append(o)

            # -------------------------------------------------------------
            # 4. TWAP (Khớp Từng Phần Theo Thời Gian)
            # -------------------------------------------------------------
            elif o.order_type == "TWAP":
                o.twap_tick_counter += 1
                if o.twap_tick_counter >= o.twap_interval_ticks:
                    o.twap_tick_counter = 0
                    o.twap_filled_slices += 1
                    
                    # Create micro-slice order execution
                    slice_margin = round(o.margin / o.twap_total_slices, 2)
                    slice_units = round(o.units / o.twap_total_slices, 4)
                    
                    slice_suborder = FuturesOrder(
                        order_id=o.order_id,
                        symbol=o.symbol,
                        order_type="TWAP_SLICE",
                        side=o.side,
                        direction=o.direction,
                        price=current_price,
                        margin=slice_margin,
                        leverage=o.leverage,
                        units=slice_units,
                        status="FILLED",
                        created_at=datetime.now().strftime("%H:%M:%S"),
                        note=f"TWAP Lát Cắt {o.twap_filled_slices}/{o.twap_total_slices}"
                    )
                    filled_orders.append(slice_suborder)
                    
                    if o.twap_filled_slices >= o.twap_total_slices:
                        o.status = "FILLED"
                        o.note = f"Hoàn tất {o.twap_total_slices}/{o.twap_total_slices} lát cắt TWAP"

        return filled_orders

    def get_pending_orders(self) -> List[dict]:
        active = [
            {
                "order_id": o.order_id,
                "symbol": o.symbol,
                "type": o.order_type,
                "side": o.side,
                "direction": o.direction,
                "price": o.price,
                "margin": o.margin,
                "leverage": o.leverage,
                "units": o.units,
                "status": o.status,
                "created_at": o.created_at,
                "trigger_price": o.trigger_price,
                "trigger_condition": o.trigger_condition,
                "callback_pct": o.callback_pct,
                "twap_progress": f"{o.twap_filled_slices}/{o.twap_total_slices}" if o.order_type == "TWAP" else "--",
                "scale_level": o.scale_level if o.order_type == "SCALE_RATIO" else "--",
                "scale_ratio": f"{o.scale_ratio_pct}%" if o.order_type == "SCALE_RATIO" else "--",
                "note": o.note,
                "cancel_reason": getattr(o, "cancel_reason", None),
                "age_ticks": getattr(o, "age_ticks", 0),
                "timeframe": getattr(o, "timeframe", "15m")
            }
            for o in self.orders if o.status in ("PENDING", "ACTIVE")
        ]
        recent_canceled = [
            {
                "order_id": o.order_id,
                "symbol": o.symbol,
                "type": o.order_type,
                "side": o.side,
                "direction": o.direction,
                "price": o.price,
                "margin": o.margin,
                "leverage": o.leverage,
                "units": o.units,
                "status": "CANCELED",
                "created_at": getattr(o, "cancelled_at", o.created_at),
                "trigger_price": o.trigger_price,
                "trigger_condition": o.trigger_condition,
                "callback_pct": o.callback_pct,
                "twap_progress": "--",
                "scale_level": "--",
                "scale_ratio": "--",
                "note": getattr(o, "cancel_reason", "AI Hủy: Mất vị thế phù hợp"),
                "cancel_reason": getattr(o, "cancel_reason", "AI Hủy: Mất vị thế phù hợp"),
                "age_ticks": getattr(o, "age_ticks", 0),
                "timeframe": getattr(o, "timeframe", "15m")
            }
            for o in self.orders if o.status == "CANCELED" and getattr(o, "cancel_reason", "")
        ][-3:]
        return active + recent_canceled

    @property
    def pending_orders(self) -> List[FuturesOrder]:
        return [o for o in self.orders if o.status in ("PENDING", "ACTIVE")]
