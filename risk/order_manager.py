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
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from decimal import Decimal
import copy
import json
import math
import time
from typing import Any, List, Dict, Optional, Tuple


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
    execution_group: str = ""
    client_order_id: str = ""
    exchange_order_id: str = ""
    exchange_status: str = ""
    group_type: str = ""
    position_side: str = "BOTH"
    parent_intent_id: str = ""
    child_index: int = 1
    due_at: float = 0.0
    submitted_at: float = 0.0
    applied_quantity: float = 0.0
    applied_quote: float = 0.0
    exchange_executed_quantity: float = 0.0
    exchange_cumulative_quote: float = 0.0
    exchange_avg_price: float = 0.0
    candidate_payload: dict = field(default_factory=dict)
    decision_trace: dict = field(default_factory=dict)


class OrderQueueManager:
    OPEN_STATUSES = {"PENDING", "ACTIVE", "SUBMIT_PENDING", "SUBMIT_UNKNOWN", "EXCHANGE_ACK", "CANCEL_REQUESTED"}

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
        timeframe: str = "15m",
        execution_group: str = "",
        client_order_id: str = "",
        quantity: Optional[float] = None,
        candidate_payload: Optional[dict] = None,
        decision_trace: Optional[dict] = None,
        group_type: str = "",
        position_side: str = "BOTH",
        due_at: Optional[float] = None,
        twap_interval_seconds: Optional[float] = None,
    ) -> Tuple[Optional[FuturesOrder], str]:
        """
        Validates and places any of the 7 order types.
        Returns (order, message).
        """
        for name, value in (("price", price), ("margin", margin), ("leverage", leverage), ("stop_loss", stop_loss), ("take_profit", take_profit),
                            ("trigger_price", trigger_price), ("callback_pct", callback_pct)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                return None, f"Invalid {name}"
        if side.upper() not in ("BUY", "SELL") or margin <= 0 or leverage < 1 or int(leverage) != leverage:
            return None, "Invalid side/margin/leverage"
        if position_side not in ("BOTH", "LONG", "SHORT"):
            return None, "Invalid position side"
        if (position_side == "LONG" and side.upper() != "BUY") or (position_side == "SHORT" and side.upper() != "SELL"):
            return None, "Position side conflicts with order side"
        direction = 1 if side.upper() == "BUY" else -1
        order_type_clean = order_type.upper()
        if order_type_clean not in ("LIMIT", "POST_ONLY", "MARKET", "CONDITIONAL", "TRAILING_STOP", "TWAP", "TWAP_SLICE", "SCALE_RATIO"):
            return None, "Unsupported order type"
        target_price = price if price > 0 else (best_ask if direction == 1 else best_bid)
        if not math.isfinite(target_price) or target_price <= 0:
            return None, "Invalid execution price"
        total_units = quantity if quantity is not None else margin * leverage / target_price
        if isinstance(total_units, bool) or not isinstance(total_units, (int, float)) or not math.isfinite(total_units) or total_units <= 0:
            return None, "Invalid quantity"
        existing = next((item for item in self.orders if client_order_id and
                         (item.client_order_id == client_order_id or item.parent_intent_id == client_order_id)), None)
        if existing:
            return existing, "Existing idempotent intent"
        try:
            json.dumps({"candidate": candidate_payload or {}, "trace": decision_trace or {}}, allow_nan=False)
        except (TypeError, ValueError):
            return None, "Candidate/trace must be finite JSON data"
        due_at = time.time() if due_at is None else due_at
        if not isinstance(due_at, (int, float)) or not math.isfinite(due_at) or due_at < 0:
            return None, "Invalid due_at"

        # 1. Validate POST_ONLY (Guaranteed Maker protection)
        if order_type_clean == "POST_ONLY":
            if direction == 1 and best_ask > 0 and price >= best_ask:
                return None, f"❌ BỊ TỪ CHỐI BỞI POST-ONLY: Giá mua ${price:,.2f} >= Best Ask ${best_ask:,.2f} (Lệnh sẽ bị khớp Taker 0.05%). Đã bảo vệ phí Maker cho bạn!"
            elif direction == -1 and best_bid > 0 and price <= best_bid:
                return None, f"❌ BỊ TỪ CHỐI BỞI POST-ONLY: Giá bán ${price:,.2f} <= Best Bid ${best_bid:,.2f} (Lệnh sẽ bị khớp Taker 0.05%). Đã bảo vệ phí Maker cho bạn!"

        is_twap = order_type_clean == "TWAP"
        is_scale = order_type_clean == "SCALE_RATIO"
        if is_twap and (not isinstance(twap_slices, int) or not 1 <= twap_slices <= 1000):
            return None, "Invalid TWAP slice count"
        count = twap_slices if is_twap else (3 if is_scale else 1)
        # Legacy callers supplied ~500ms ticks; convert once to seconds, never count incoming ticks.
        interval = twap_interval_seconds if twap_interval_seconds is not None else twap_interval_ticks * 0.5
        if is_twap and (not isinstance(interval, (int, float)) or not math.isfinite(interval) or interval <= 0):
            return None, "Invalid TWAP interval"
        ratios = [.2, .3, .5] if is_scale else [1 / count] * count
        units = [float(Decimal(str(total_units)) * Decimal(str(ratio))) for ratio in ratios[:-1]]
        units.append(total_units - math.fsum(units))
        while math.fsum(units) > total_units:
            units[-1] = math.nextafter(units[-1], 0.0)
        parent_id = client_order_id or f"intent-{self._next_id}"
        kind = "TWAP" if is_twap else ("SCALE" if is_scale else (group_type.upper() or ("OCO" if execution_group.lower().startswith(("oco", "dual")) else "")))
        group = execution_group or (parent_id if is_twap or is_scale else "")
        created = []
        scale_step = 0.005
        if isinstance(candidate_payload, dict) and "dca_ladder_step" in candidate_payload:
            try:
                scale_step = float(candidate_payload["dca_ladder_step"])
            except (ValueError, TypeError):
                scale_step = 0.005

        for index, child_units in enumerate(units, 1):
            child_price = target_price * (1 - direction * (index - 1) * scale_step) if is_scale else target_price
            price_shift = (child_price - target_price) if is_scale else 0.0
            child_sl = round(stop_loss + price_shift, 2) if (stop_loss and stop_loss > 0) else stop_loss
            child_tp = round(take_profit + price_shift, 2) if (take_profit and take_profit > 0) else take_profit
            child_client_id = f"{parent_id}-{index}" if count > 1 else parent_id
            if len(child_client_id) > 36:
                return None, "Client order ID exceeds Binance 36-character limit"
            order = FuturesOrder(
                order_id=self._next_id + index - 1, symbol=symbol, order_type="TWAP_SLICE" if is_twap else order_type_clean,
                side=side.upper(), direction=direction, price=child_price,
                margin=child_units * child_price / leverage, leverage=leverage, units=child_units, status="PENDING",
                created_at=datetime.now().isoformat(), timeframe=timeframe, trigger_price=trigger_price,
                trigger_condition=trigger_condition.upper(), callback_pct=callback_pct, peak_price=target_price,
                twap_total_slices=count if is_twap else 1, twap_interval_ticks=twap_interval_ticks,
                scale_level=index, scale_ratio_pct=ratios[index - 1] * 100, stop_loss=child_sl, take_profit=child_tp,
                note=note or (f"{order_type_clean} child {index}/{count}" if count > 1 else ""),
                execution_group=group, client_order_id=child_client_id, group_type=kind,
                position_side=position_side,
                parent_intent_id=parent_id, child_index=index, due_at=due_at + (index - 1) * interval if is_twap else due_at,
                candidate_payload=copy.deepcopy(candidate_payload or {}), decision_trace=copy.deepcopy(decision_trace or {}))
            created.append(order)
        self.orders.extend(created)
        self._next_id += count
        return created[0], f"Đã đặt {count} intent {order_type_clean}"

    def export_state(self) -> dict:
        result = {"version": 1, "next_id": self._next_id, "orders": [asdict(order) for order in self.orders]}
        json.dumps(result, allow_nan=False)
        return result

    def restore_state(self, payload: dict) -> None:
        """Restore atomically. Interrupted submissions must be queried, never blindly resent."""
        if not isinstance(payload, dict) or payload.get("version", 1) != 1 or not isinstance(payload.get("orders", []), list):
            raise ValueError("Invalid execution journal")
        json.dumps(payload, allow_nan=False)
        restored = []
        ids, clients = set(), set()
        field_names = {item.name for item in fields(FuturesOrder)}
        for raw in payload.get("orders", []):
            order = FuturesOrder(**{key: copy.deepcopy(value) for key, value in raw.items() if key in field_names})
            if not isinstance(order.order_id, int) or order.order_id <= 0 or order.order_id in ids:
                raise ValueError("Duplicate/invalid local order ID")
            if order.client_order_id and order.client_order_id in clients:
                raise ValueError("Duplicate client order ID")
            for value in (order.units, order.price, order.margin, order.due_at, order.applied_quantity,
                          order.applied_quote, order.exchange_executed_quantity, order.exchange_cumulative_quote):
                if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    raise ValueError("Invalid execution journal quantity/price")
            if order.side not in ("BUY", "SELL") or order.status not in self.OPEN_STATUSES | {"FILLED", "CANCELED", "REJECTED", "EXPIRED"}:
                raise ValueError("Invalid execution journal side/status")
            if order.applied_quantity > order.exchange_executed_quantity:
                raise ValueError("Applied quantity exceeds confirmed exchange quantity")
            if order.status == "SUBMIT_PENDING":
                order.status = "SUBMIT_UNKNOWN"
            ids.add(order.order_id)
            if order.client_order_id:
                clients.add(order.client_order_id)
            restored.append(order)
        next_id = payload.get("next_id", 1)
        if not isinstance(next_id, int) or next_id < 1:
            raise ValueError("Invalid next order ID")
        self.orders = restored
        self._next_id = max(next_id, max(ids, default=0) + 1)

    def due_orders(self, now: Optional[float] = None) -> List[FuturesOrder]:
        now = time.time() if now is None else now
        return [order for order in self.orders if order.status in ("PENDING", "ACTIVE") and order.due_at <= now
                and not order.exchange_order_id and not order.exchange_status]

    def mark_submit_pending(self, order_id: int) -> bool:
        order = next((item for item in self.orders if item.order_id == order_id), None)
        if not order or order.status not in ("PENDING", "ACTIVE") or order.exchange_order_id or order.exchange_status:
            return False
        order.status = "SUBMIT_PENDING"
        order.submitted_at = time.time()
        return True

    def mark_submit_unknown(self, order_id: int, reason: str = "Exchange outcome unknown; reconciliation required") -> bool:
        order = next((item for item in self.orders if item.order_id == order_id), None)
        if not order or order.status != "SUBMIT_PENDING":
            return False
        order.status = "SUBMIT_UNKNOWN"
        order.note = reason
        return True

    def record_exchange_update(self, order_id: int, response: dict) -> bool:
        """Save monotonic cumulative fill facts; applying them to a position is a separate durable step."""
        order = next((item for item in self.orders if item.order_id == order_id), None)
        if not order:
            return False
        try:
            qty = float(response.get("executedQty", 0) or 0)
            avg = float(response.get("avgPrice", 0) or 0)
            quote = float(response.get("cumQuote", 0) or 0) or qty * avg
            if any(not math.isfinite(value) or value < 0 for value in (qty, avg, quote)) or (qty > 0 and quote <= 0):
                return False
        except (TypeError, ValueError, OverflowError):
            return False
        if qty < order.exchange_executed_quantity:
            return False
        status = str(response.get("status", "NEW")).upper()
        if status not in {"NEW", "PARTIALLY_FILLED", "FILLED", "CANCELED", "EXPIRED", "REJECTED", "PENDING_TRIGGER"}:
            return False
        if order.status in ("FILLED", "CANCELED", "EXPIRED", "REJECTED") and status in ("NEW", "PARTIALLY_FILLED", "PENDING_TRIGGER"):
            return False
        if (qty > order.exchange_executed_quantity and quote <= order.exchange_cumulative_quote) or quote < order.applied_quote:
            return False
        if status == "FILLED" and qty <= 0:
            return False
        order.exchange_order_id = str(response.get("orderId") or order.exchange_order_id)
        order.exchange_status = status
        order.exchange_executed_quantity = qty
        order.exchange_cumulative_quote = quote
        order.exchange_avg_price = quote / qty if qty else 0.0
        if status in ("FILLED", "CANCELED", "EXPIRED", "REJECTED"):
            order.status = status
        elif order.status != "CANCEL_REQUESTED":
            order.status = "EXCHANGE_ACK"
        if qty > 0:
            self._cancel_oco_siblings(order)
        return True

    def pending_fill(self, order_id: int) -> Tuple[float, float]:
        order = next((item for item in self.orders if item.order_id == order_id), None)
        if not order or order.exchange_executed_quantity <= order.applied_quantity:
            return 0.0, 0.0
        quantity = order.exchange_executed_quantity - order.applied_quantity
        return quantity, (order.exchange_cumulative_quote - order.applied_quote) / quantity

    def mark_fill_applied(self, order_id: int) -> bool:
        order = next((item for item in self.orders if item.order_id == order_id), None)
        if not order:
            return False
        order.applied_quantity = order.exchange_executed_quantity
        order.applied_quote = order.exchange_cumulative_quote
        return True

    def _cancel_oco_siblings(self, filled: FuturesOrder) -> List[FuturesOrder]:
        cancelled = []
        if filled.execution_group and filled.group_type == "OCO":
            for sibling in self.orders:
                if (sibling.order_id != filled.order_id and sibling.execution_group == filled.execution_group
                        and sibling.group_type == "OCO" and sibling.side != filled.side and sibling.status in self.OPEN_STATUSES):
                    sibling.status = "CANCEL_REQUESTED"
                    sibling.cancel_reason = f"OCO sibling #{filled.order_id} filled"
                    cancelled.append(sibling)
        return cancelled

    def mark_exchange_ack(self, order_id: int, exchange_order_id: str, client_order_id: str = "") -> bool:
        """An ACK is an open exchange order, never a local fill."""
        for order in self.orders:
            if order.order_id == order_id and order.status in ("PENDING", "ACTIVE", "SUBMIT_PENDING", "SUBMIT_UNKNOWN"):
                order.status = "EXCHANGE_ACK"
                order.exchange_order_id = str(exchange_order_id)
                order.client_order_id = client_order_id or order.client_order_id
                order.exchange_status = "NEW"
                return True
        return False

    def mark_exchange_fill(self, order_id: int) -> List[FuturesOrder]:
        """Record a confirmed fill and request cancellation of OCO siblings."""
        filled = next((order for order in self.orders if order.order_id == order_id), None)
        if not filled or filled.status not in self.OPEN_STATUSES:
            return []
        filled.status = "FILLED"
        filled.exchange_status = "FILLED"
        return self._cancel_oco_siblings(filled)

    def mark_exchange_cancelled(self, order_id: int, reason: str = "exchange cancellation") -> bool:
        for order in self.orders:
            if order.order_id == order_id and order.status in self.OPEN_STATUSES:
                order.status = "CANCELED"
                order.exchange_status = "CANCELED"
                order.cancel_reason = reason
                order.cancelled_at = datetime.now().strftime("%H:%M:%S")
                return True
        return False

    def cancel_order(self, order_id: int) -> bool:
        for o in self.orders:
            if o.order_id == order_id and o.status in ("PENDING", "ACTIVE") and not o.exchange_order_id and not o.exchange_status:
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
        for value in (price, units, margin, trigger_price, stop_loss, take_profit, leverage, callback_pct):
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0):
                return False, "Invalid order update"
        if side is not None and side.upper() not in ("BUY", "SELL"):
            return False, "Invalid order side"
        if leverage is not None and (leverage < 1 or int(leverage) != leverage):
            return False, "Invalid order leverage"
        for o in self.orders:
            if o.order_id == order_id and o.status in ("PENDING", "ACTIVE") and not o.exchange_order_id and not o.exchange_status:
                if price is not None and price > 0:
                    o.price = price
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
                    o.units = units
                    if o.price > 0 and o.leverage > 0:
                        o.margin = (o.units * o.price) / o.leverage
                elif margin is not None and margin > 0:
                    o.margin = margin
                    if o.price > 0 and o.leverage > 0:
                        o.units = (o.margin * o.leverage) / o.price
                elif price is not None and price > 0:
                    o.margin = (o.units * o.price) / o.leverage

                # Reset age ticks so order isn't immediately aged out
                o.age_ticks = 0
                o.note = f"Đã cập nhật thủ công lúc {datetime.now().strftime('%H:%M:%S')}"
                return True, f"Đã cập nhật lệnh chờ #{order_id} thành công!"
        return False, f"Không tìm thấy lệnh chờ #{order_id} hoặc lệnh không ở trạng thái hoạt động!"

    def force_execute_order(self, order_id: int, execution_price: float) -> Optional[FuturesOrder]:
        """Manually forces immediate market execution of a pending order"""
        for o in self.orders:
            if o.order_id == order_id and o.status in ("PENDING", "ACTIVE") and not o.exchange_order_id and not o.exchange_status:
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
            if o.status not in ("PENDING", "ACTIVE") or o.exchange_order_id or o.exchange_status or o.due_at > time.time():
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

    def match_orders(self, current_price: float, best_bid: float = 0.0, best_ask: float = 0.0, now: Optional[float] = None) -> List[FuturesOrder]:
        """
        Evaluates all pending orders on live sub-second price ticks:
        1. LIMIT & POST_ONLY: Fills when market crosses limit price
        2. CONDITIONAL: Triggers when price crosses trigger threshold, turns into Market fill
        3. TRAILING_STOP: Updates peak/trough; triggers when pullback >= callback_pct
        4. TWAP: Match pre-created unique child intents only after their wall-clock due_at
        5. SCALE_RATIO: Fills individual ladder steps as price reaches them
        """
        filled_orders = []
        now = time.time() if now is None else now

        for o in self.orders:
            if o.status not in ("PENDING", "ACTIVE") or o.exchange_order_id or o.exchange_status or o.due_at > now:
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
            elif o.order_type in ("TWAP_SLICE", "MARKET"):
                o.status = "FILLED"
                o.price = (best_ask if o.side == "BUY" else best_bid) or current_price
                o.margin = o.units * o.price / o.leverage
                o.twap_filled_slices = 1
                filled_orders.append(o)

            if o.status == "FILLED":
                self._cancel_oco_siblings(o)

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
                "timeframe": getattr(o, "timeframe", "15m"),
                "execution_group": o.execution_group,
                "client_order_id": o.client_order_id,
                "exchange_order_id": o.exchange_order_id,
                "exchange_status": o.exchange_status,
                "group_type": o.group_type,
                "parent_intent_id": o.parent_intent_id,
                "child_index": o.child_index,
                "due_at": o.due_at,
                "applied_quantity": o.applied_quantity,
                "exchange_executed_quantity": o.exchange_executed_quantity,
            }
            for o in self.orders if o.status in self.OPEN_STATUSES or o.exchange_executed_quantity > o.applied_quantity
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
        return [o for o in self.orders if o.status in self.OPEN_STATUSES or o.exchange_executed_quantity > o.applied_quantity]
