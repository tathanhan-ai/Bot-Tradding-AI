"""Durable execution ledger. Exchange acknowledgements are never local fills."""
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import math
import time
import uuid

from trading.pipeline import CandidateOrder, StageOutcome
from strategy.memory_reasoning_engine import TradeContextProfile


class ExecutionLifecycle:
    def __init__(self, state):
        self.state = state
        self.protective = []
        self.exits = []
        self.traces = {}
        self.last_reconcile = 0.0
        self.processed_execution_ids = set()
        self.feedback_pending = False
        self.startup_reconciled = False
        self.entry_exit_barrier = ""

    @property
    def live(self):
        return self.state.active_exchange == "binance" and self.state.binance_api.is_live_enabled

    def restore(self):
        saved = self.state.storage.load_execution_runtime()
        if not saved:
            return
        s = self.state
        s.current_balance = saved["balance"]
        s.total_fees = saved["total_fees"]
        s.current_position = saved["position"]
        s.hedge_positions = saved.get("hedge_positions", {})
        s.order_manager.restore_state(saved["queue"])
        self.protective = saved.get("protective", [])
        self.exits = saved.get("exits", [])
        self.traces = saved.get("traces", {})
        self.processed_execution_ids = set(saved.get("processed_execution_ids", []))
        self.feedback_pending = bool(saved.get("feedback_pending", False))
        self.entry_exit_barrier = saved.get("entry_exit_barrier", "")
        s.trade_memory.memory_records = [TradeContextProfile(**record) for record in saved.get("memory_records", [])][-s.trade_memory.max_records:]
        s.trades = saved.get("trades", s.trades)
        s.execution_blocker = saved.get("blocker", "")
        # Restore the ledger balance first: a UTC day rollover must start from it,
        # not from the constructor's configured initial balance.
        s.risk_manager.update_balance(s.current_balance)
        if saved.get("risk_state") is not None and hasattr(s.risk_manager, "restore_state"):
            s.risk_manager.restore_state(saved["risk_state"])
            s.risk_manager.update_balance(s.current_balance)
        if saved.get("protection_state") is not None and hasattr(s.freqtrade_protections, "restore_state"):
            s.freqtrade_protections.restore_state(saved["protection_state"])

    def persist(self):
        s = self.state
        s.storage.save_execution_runtime({
            "version": 2, "balance": s.current_balance, "total_fees": s.total_fees,
            "position": s.current_position, "hedge_positions": getattr(s, "hedge_positions", {}), "queue": s.order_manager.export_state(),
            "protective": self.protective, "exits": self.exits,
            "traces": self.traces, "trades": s.trades[-200:],
            "processed_execution_ids": sorted(self.processed_execution_ids), "feedback_pending": self.feedback_pending,
            "entry_exit_barrier": self.entry_exit_barrier,
            "memory_records": [asdict(record) for record in s.trade_memory.memory_records],
            "risk_state": s.risk_manager.export_state() if hasattr(s.risk_manager, "export_state") else None,
            "protection_state": s.freqtrade_protections.export_state() if hasattr(s.freqtrade_protections, "export_state") else None,
            "blocker": getattr(s, "execution_blocker", ""),
        })

    def trace(self, parent, stage, verdict, reason, **details):
        item = {"stage": stage, "verdict": verdict, "reason": reason, "details": details, "timestamp": time.time()}
        self.traces.setdefault(parent, []).append(item)
        trace = self.state.last_decision_trace
        if trace and trace.order_id == parent:
            trace.add(stage, StageOutcome(verdict, reason, details=details))

    @staticmethod
    def is_hedge_grid(order):
        return order.group_type == "GRID" and order.position_side in ("LONG", "SHORT")

    def position_for(self, position_side=None):
        if position_side in ("LONG", "SHORT"):
            return getattr(self.state, "hedge_positions", {}).get(position_side)
        return self.state.current_position

    def all_positions(self):
        current = self.state.current_position
        if current:
            yield "BOTH", current
        yield from getattr(self.state, "hedge_positions", {}).items()

    @staticmethod
    def summarize_entry_reason(candidate, trace_entries) -> str:
        # Tóm tắt lý do vào lệnh cho bảng lệnh chờ/vị thế (tối đa ~200 ký tự).
        parts = []
        try:
            direction = "LONG" if getattr(candidate, "direction", 0) == 1 else ("SHORT" if getattr(candidate, "direction", 0) == -1 else "")
            kind = str(getattr(candidate, "order_type", "") or "")
            conf = getattr(candidate, "confidence", 0) or 0
            regime = str(getattr(candidate, "regime", "") or "")
            head = " ".join(p for p in (direction, kind) if p).strip()
            if head:
                parts.append(head)
            if regime:
                parts.append(f"chế độ {regime}")
            if conf:
                try:
                    parts.append(f"tin cậy {float(conf):.0f}%")
                except Exception:
                    pass
            # P6: ghi nhan trang thai song 1D/15m vao ly do (de nguoi dung thay
            # lenh co thuan chi thi song khong ma khong phai mo trace).
            try:
                meta = getattr(candidate, "metadata", {}) or {}
                wave = meta.get("wave_alignment", {}) or {}
                w1 = int(wave.get("wave_1d", 0) or 0)
                w15 = int(wave.get("wave_15m", 0) or 0)
                if w1 or w15:
                    arrow = lambda v: "tăng" if v == 1 else ("giảm" if v == -1 else "?")
                    tag = f"sóng 1D {arrow(w1)}/15m {arrow(w15)}"
                    tag += " lệch sóng-probation" if meta.get("wave_misaligned") else " đồng thuận"
                    parts.append(tag)
                for flag in ("hurst_downgraded", "octo_weak_conflict", "alpha_weak_conflict"):
                    if meta.get(flag):
                        label = {"hurst_downgraded": "Hurst hạ cấp Maker",
                                 "octo_weak_conflict": "OctoBot yếu-probation",
                                 "alpha_weak_conflict": "Alpha yếu-probation"}[flag]
                        parts.append(label)
                        break
            except Exception:
                pass
        except Exception:
            pass
        for entry in (trace_entries or [])[-6:]:
            try:
                stage = str(entry.get("stage", "") if isinstance(entry, dict) else getattr(entry, "stage", ""))
                verdict = str(entry.get("verdict", "") if isinstance(entry, dict) else getattr(entry, "verdict", ""))
                reason = str(entry.get("reason", "") if isinstance(entry, dict) else getattr(entry, "reason", ""))
                if verdict == "PASS" and reason and ("ouncil" in stage or "ctoBot" in stage or "pha Zoo" in stage or "arver" in stage or "isk" in stage):
                    short = reason.strip().split("\n")[0][:120]
                    if short and short not in parts:
                        parts.append(short)
            except Exception:
                continue
        text = " | ".join(p for p in parts if p).strip()
        return text[:500] if text else "Lệnh đã qua đủ cổng kiểm duyệt"

    def queue(self, candidate, research=None, execution_group=""):
        s = self.state
        if candidate.quantity <= 0 or not s.last_decision_trace or s.last_decision_trace.vetoed:
            raise ValueError("Only a sized, approved candidate can enter execution")
        if s.last_decision_trace.order_id != candidate.order_id:
            raise ValueError("Candidate/trace mismatch")
        self.traces[candidate.order_id] = [asdict(e) for e in s.last_decision_trace.entries]
        meta = candidate.metadata
        kind = candidate.order_type
        if kind not in ("MARKET", "LIMIT", "POST_ONLY", "CONDITIONAL", "TRAILING_STOP", "TWAP", "SCALE_RATIO"):
            raise ValueError(f"Unsupported execution type: {kind}")
        first, reason = s.order_manager.place_order(
            order_type=kind, symbol=candidate.symbol, side="BUY" if candidate.direction == 1 else "SELL",
            price=candidate.entry_price, margin=candidate.margin, leverage=candidate.leverage,
            quantity=candidate.quantity, stop_loss=candidate.stop_loss, take_profit=candidate.take_profit,
            entry_reason=self.summarize_entry_reason(candidate, self.traces[candidate.order_id]),
            trigger_price=meta.get("trigger_price", getattr(research, "optimal_trigger_price", 0)),
            trigger_condition=meta.get("trigger_condition", getattr(research, "optimal_trigger_cond", "ABOVE")),
            callback_pct=meta.get("callback_pct", getattr(research, "optimal_callback_pct", 0.8)),
            twap_slices=meta.get("twap_slices", getattr(research, "optimal_twap_slices", 5)),
            twap_interval_seconds=meta.get("twap_interval_seconds", 6),
            timeframe=meta.get("timeframe", getattr(candidate, "timeframe", s.active_timeframe)),
            best_bid=s.fee_engine.bid_price, best_ask=s.fee_engine.ask_price,
            execution_group=execution_group or candidate.order_id, group_type=meta.get("group_type", ""),
            client_order_id=candidate.order_id, candidate_payload=asdict(candidate),
            decision_trace=self.traces[candidate.order_id],
        )
        # Gia Maker cham spread (thi truong chay giua chung pipeline duyet va
        # tang dat lenh): lui 1 tick ve phia Maker roi dat lai, thay vi rot han.
        # Chi ap dung 1 lan cho lenh don (khong ap cho SCALE ladder nhieu chan).
        if first is None and kind in ("POST_ONLY", "LIMIT") and isinstance(reason, str) and "BỊ TỪ CHỐI BỞI" in reason:
            try:
                tick = 0.1
                px = float(candidate.entry_price or 0.0)
                if candidate.direction == 1:
                    ask = float(s.fee_engine.ask_price or 0.0)
                    if ask > 0 and px >= ask:
                        candidate.entry_price = round(ask - tick, 1)
                else:
                    bid = float(s.fee_engine.bid_price or 0.0)
                    if bid > 0 and px <= bid:
                        candidate.entry_price = round(bid + tick, 1)
                if float(candidate.entry_price or 0.0) != float(px or 0.0):
                    first, reason = s.order_manager.place_order(
                        order_type=kind, symbol=candidate.symbol,
                        side="BUY" if candidate.direction == 1 else "SELL",
                        price=candidate.entry_price, margin=candidate.margin, leverage=candidate.leverage,
                        quantity=candidate.quantity, stop_loss=candidate.stop_loss, take_profit=candidate.take_profit,
                        entry_reason=self.summarize_entry_reason(candidate, self.traces[candidate.order_id]),
                        trigger_price=meta.get("trigger_price", getattr(research, "optimal_trigger_price", 0)),
                        trigger_condition=meta.get("trigger_condition", getattr(research, "optimal_trigger_cond", "ABOVE")),
                        callback_pct=meta.get("callback_pct", getattr(research, "optimal_callback_pct", 0.8)),
                        twap_slices=meta.get("twap_slices", getattr(research, "optimal_twap_slices", 5)),
                        twap_interval_seconds=meta.get("twap_interval_seconds", 6),
                        timeframe=meta.get("timeframe", getattr(candidate, "timeframe", s.active_timeframe)),
                        best_bid=s.fee_engine.bid_price, best_ask=s.fee_engine.ask_price,
                        execution_group=execution_group or candidate.order_id, group_type=meta.get("group_type", ""),
                        client_order_id=candidate.order_id, candidate_payload=asdict(candidate),
                        decision_trace=self.traces[candidate.order_id],
                    )
                    if first is not None:
                        self.trace(candidate.order_id, "Execution / Maker nudge",
                                   "PASS", f"Lui 1 tick ve Maker ({px} -> {candidate.entry_price}) de tranh khop Taker")
            except Exception:
                pass
        if first is None:
            self.trace(candidate.order_id, "Execution", "VETO", reason)
            self.persist()
            return {"status": "rejected", "reason": reason}
        self.persist()  # Intent exists before any external side effect.
        self.tick()
        return {"status": first.status.lower(), "order_id": first.order_id, "client_order_id": first.client_order_id}

    def queue_grid(self, candidate, legs):
        """Persist all passive grid children beneath one approved parent trace."""
        s = self.state
        if candidate.order_type != "GRID" or candidate.direction != 0 or candidate.quantity <= 0:
            raise ValueError("Only a sized GRID parent can create grid children")
        if not s.last_decision_trace or s.last_decision_trace.order_id != candidate.order_id or s.last_decision_trace.vetoed:
            raise ValueError("Grid parent/trace mismatch")
        if not legs or len(legs) % 2:
            raise ValueError("Hedge grid must contain paired legs")
        self.traces[candidate.order_id] = [asdict(entry) for entry in s.last_decision_trace.entries]
        quantity = math.floor((candidate.quantity / len(legs)) / .001 + 1e-12) * .001
        if quantity <= 0:
            self.trace(candidate.order_id, "Execution / Grid", "VETO", "Grid allocation is below BTCUSDT minimum step")
            self.persist()
            return {"status": "blocked", "reason": "Grid allocation below minimum step"}
        created = []
        for index, leg in enumerate(legs, 1):
            order, reason = s.order_manager.place_order(
                symbol=candidate.symbol, order_type="POST_ONLY", side=leg["side"], price=leg["entry_price"],
                margin=quantity * leg["entry_price"] / candidate.leverage, leverage=candidate.leverage,
                quantity=quantity, stop_loss=leg["stop_loss"], take_profit=leg["take_profit"],
                timeframe=s.active_timeframe, best_bid=s.fee_engine.bid_price, best_ask=s.fee_engine.ask_price,
                execution_group=candidate.order_id, group_type="GRID", position_side="LONG" if leg["side"] == "BUY" else "SHORT",
                client_order_id=f"g-{candidate.order_id[-20:]}-{index}",
                candidate_payload={**asdict(candidate), "metadata": {**candidate.metadata, "grid_leg": leg}},
                decision_trace=self.traces[candidate.order_id],
            )
            if order is None:
                for prior in created:
                    s.order_manager.cancel_order(prior.order_id, f"Rollback lưới GRID: {reason}")
                self.trace(candidate.order_id, "Execution / Grid", "VETO", reason)
                self.persist()
                return {"status": "blocked", "reason": reason}
            created.append(order)
        self.persist()
        self.tick()
        self.trace(candidate.order_id, "Execution / Grid", "PASS", "Grid child intents persisted", children=len(created))
        self.persist()
        return {"status": "active", "grid_plan_id": candidate.order_id, "children": len(created)}

    @staticmethod
    def exchange_type(order):
        if order.order_type in ("MARKET", "TWAP_SLICE"):
            return "MARKET"
        if order.order_type == "SCALE_RATIO":
            return "LIMIT"
        if order.order_type == "TRAILING_STOP":
            return "TRAILING_STOP_MARKET"
        if order.order_type == "CONDITIONAL":
            # BUY below / SELL above is take-profit; the inverse is a stop.
            stop = (order.side == "BUY") == (order.trigger_condition == "ABOVE")
            return "STOP" if stop else "TAKE_PROFIT"
        return order.order_type

    def entry_guard(self, order):
        s = self.state
        if self.entry_exit_barrier:
            return "Full close is reconciling outstanding entry children"
        snap = s.build_market_snapshot()
        issues = snap.freshness_issues()
        if self.live:
            expected_env = "testnet" if s.binance_api.is_testnet else "mainnet"
            if snap.environment != expected_env:
                issues.append(f"market-data/execution mismatch (live={expected_env}, data={snap.environment})")
        if issues:
            return "; ".join(issues)
        metrics = snap.context["hft"]
        is_grid = getattr(order, "group_type", "") == "GRID"
        if not getattr(metrics, "depth_ready", True):
            return "Measured microstructure is not ready"
        if not is_grid and (not getattr(metrics, "vpin_ready", True) and metrics.market_resilience_pct < 25.0):
            return "Measured microstructure is not ready"
        if not is_grid and metrics.liquidity_drought_warning and metrics.market_resilience_pct < 20.0:
            return "Microstructure deteriorated before execution"
        floating = sum(position.get("unrealized_pnl", 0.0) for _, position in self.all_positions())
        target_tp = 0.0 if is_grid else order.take_profit
        approved, reason, _ = s.freqtrade_protections.validate_new_trade(order.price, target_tp, order.direction, s.current_balance, s.current_balance + floating)
        if not approved:
            return reason
        last_trade_closed = s.trades[-1].get("closed_at_ts", 0.0) if s.trades else 0.0
        now_ts = time.time()
        streak = int(s.jesse_engine.compute_metrics().current_consecutive_losses or 0)
        if streak >= 7 and (now_ts - last_trade_closed) < 24 * 3600.0:
            return "Realized-loss halt: 7 consecutive losses (24h review)"
        if streak >= 3 and (now_ts - last_trade_closed) <= 1800.0:
            return "Realized-loss circuit breaker"
        if s.risk_manager.circuit_breaker_active:
            return "Realized-loss circuit breaker"
        price = snap.price if self.exchange_type(order) == "MARKET" else order.price
        if self.exchange_type(order) == "MARKET" and abs(price - order.price) / order.price > 0.0015:
            return "Market moved >0.15% beyond approved execution envelope"
        if not is_grid:
            proposal = s.risk_manager.evaluate_order(order.symbol, order.direction, price, order.stop_loss, order.take_profit, order.leverage)
            if not proposal.approved or order.units > proposal.units + 1e-9:
                return proposal.rejection_reason or "Quantity exceeds current risk envelope"
        hedge_grid = self.is_hedge_grid(order)
        if not hedge_grid and getattr(s, "hedge_positions", {}):
            return "Close hedge-grid exposure before opening a one-way position"
        pos = self.position_for(order.position_side if hedge_grid else None) or {}
        if pos and pos.get("direction") != order.direction:
            return "Opposite exposure already exists"
        if pos and pos.get("leverage", order.leverage) != order.leverage:
            return "Scale-in must use the current position leverage"
        pending = [o for o in s.order_manager.pending_orders if o.order_id != order.order_id]
        if any(o.symbol == order.symbol and o.leverage != order.leverage for o in pending):
            return "Pending symbol intents must share one leverage"
        open_risk = sum(position.get("units", 0) * max(0, (position.get("entry_price", 0) - position.get("stop_loss", 0)) * position.get("direction", 0))
                        for _, position in self.all_positions())
        reserved_risk = sum(max(0, o.units - o.exchange_executed_quantity) * abs(o.price - o.stop_loss) for o in pending if o.status in s.order_manager.OPEN_STATUSES)
        if open_risk + reserved_risk + order.units * abs(price - order.stop_loss) > s.current_balance * s.risk_config.max_account_risk_pct:
            # P6: gan prefix cong de truy vet diem nghen tang khop trong so veto.
            return "[EntryGuard/Risk] Aggregate risk exceeds account envelope"
        if not is_grid and not getattr(order, "candidate_payload", {}).get("metadata", {}).get("reduce_only"):
            # Chốt phân bổ vốn ở tầng khớp cuối: số dư có thể đã trôi kể từ
            # lúc pipeline duyệt. Chỉ từ chối (không tự bóp lệnh ở đây) để
            # tầng nghiên cứu tính lại size cho đúng.
            try:
                allocator = getattr(s, "capital_allocator", None)
                if allocator is not None:
                    meta = getattr(order, "candidate_payload", {}).get("metadata", {}) or {}
                    horizon = meta.get("horizon") or allocator.get_horizon(
                        meta.get("timeframe", getattr(order, "timeframe", "15m")))
                    gov = None
                    try:
                        gov = s.monthly_governor.evaluate(s.current_balance, s.trades)
                    except Exception:
                        gov = None
                    positions_now = [pos for _, pos in self.all_positions()]
                    alloc = allocator.evaluate_allocation(
                        horizon=horizon,
                        requested_margin=float(getattr(order, "margin", 0.0) or 0.0),
                        total_equity=float(s.current_balance or 0.0),
                        active_positions=positions_now,
                        pending_orders=[o for o in pending if o.status in s.order_manager.OPEN_STATUSES],
                        min_trade_margin=10.0 if meta.get("probation") else 30.0,
                        reserve_ratio_override=(gov.reserve_ratio_recommended if (gov is not None and getattr(gov, "enabled", False)) else None),
                    )
                    if not alloc.allowed:
                        return f"[EntryGuard/Alloc] Ngăn vốn {alloc.horizon} đã hết hạn mức tại lúc khớp ({alloc.rationale})"
                    if alloc.is_throttled and alloc.allocated_margin + 1e-9 < float(getattr(order, "margin", 0.0) or 0.0):
                        return (f"[EntryGuard/Alloc] Ngăn vốn {alloc.horizon} chỉ còn ${alloc.allocated_margin:.1f} "
                                f"nhưng lệnh cần ${float(getattr(order, 'margin', 0.0) or 0.0):.1f} — từ chối để tính lại size")
            except Exception:
                pass
        return ""

    def submit(self, order):
        s = self.state
        reason = self.entry_guard(order)
        if reason:
            s.order_manager.cancel_order(order.order_id, f"Tầng khớp từ chối: {reason}")
            self.trace(order.parent_intent_id, "Execution / Revalidation", "VETO", reason)
            self.persist()
            return
        kind = self.exchange_type(order)
        ok, values = s.binance_api.normalize_order_values(
            order.symbol, order.units, order.price if kind in ("LIMIT", "POST_ONLY", "STOP", "TAKE_PROFIT") else None,
            order_type=kind, reference_price=s.live_price,
            price_rounding="down" if order.side == "BUY" else "up",
        )
        if not ok:
            order.status = "REJECTED"
            self.trace(order.parent_intent_id, "Execution / Filters", "VETO", str(values))
            self.persist()
            return
        ok, leverage = s.binance_api.prepare_testnet_trading(order.symbol, order.leverage, hedge=self.is_hedge_grid(order))
        if not ok:
            order.status = "REJECTED"
            self.trace(order.parent_intent_id, "Execution / Leverage", "VETO", str(leverage))
            self.persist()
            return
        pending = [o for o in s.order_manager.pending_orders if o.order_id != order.order_id and o.symbol == order.symbol and o.status in s.order_manager.OPEN_STATUSES]
        account = s.binance_api.test_connection()
        wallet = float(account.get("wallet_balance", 0))
        available_balance = float(account.get("available_balance", 0))
        if not account.get("success") or not account.get("can_trade") or not all(math.isfinite(v) and v > 0 for v in (wallet, available_balance)):
            order.status = "REJECTED"
            self.trace(order.parent_intent_id, "Execution / Account", "VETO", "No verified Testnet wallet and available margin")
            self.persist()
            return
        if self.live:
            # Live nghe số tiền server: ví sàn lệch quá 10% so với vốn bot thì
            # KHÔNG cho sizing trên vốn sai — từ chối để chờ sync_exchange_balance.
            drift = abs(wallet - s.current_balance) / max(wallet, 1e-9)
            if drift > 0.10:
                order.status = "REJECTED"
                self.trace(order.parent_intent_id, "Execution / Account", "VETO",
                           f"Exchange wallet ${wallet:,.2f} diverges {drift * 100:.1f}% from bot capital ${s.current_balance:,.2f}; sync before sizing")
                self.persist()
                return
        open_notional = sum(position.get("units", 0) * s.live_price for _, position in self.all_positions())
        reserved_notional = sum(max(0, o.units - o.exchange_executed_quantity) * max(o.price, s.live_price) for o in pending)
        cro = s.risk_manager.ai_cro.last_verdict
        margin_cap = min(s.current_balance, wallet) * min(.35, cro.max_margin_utilization_pct / 100 if cro else .35)
        notional_cap = min(float(leverage.get("maxNotionalValue", math.inf)), margin_cap * order.leverage, open_notional + reserved_notional + available_balance * order.leverage)
        available = max(0, notional_cap - open_notional - reserved_notional)
        capped_quantity = min(values["quantity"], available / max(order.price, s.live_price))
        metadata = (order.candidate_payload or {}).get("metadata", {})
        # Von nho: tang nghien cuu da nang probation len buoc toi thieu san
        # (co min_size_floor). Tang gui khong duoc cat nho hon buoc do nua —
        # neu khong lenh nao cung chet o normalize "quantity below minQty".
        min_floor = bool(metadata.get("min_size_floor"))
        floor_step = 0.0
        try:
            floor_step = float((values or {}).get("step_size", 0.0) or 0.0)
        except Exception:
            floor_step = 0.0
        risk_pct = .0025 if metadata.get("probation") else (cro.risk_per_trade_pct / 100 if cro else .015)
        fee_fn = getattr(s.fee_engine, "unit_risk_with_fees", None)
        if callable(fee_fn):
            unit_risk = fee_fn(order.price, abs(order.price - order.stop_loss))
        else:
            unit_risk = abs(order.price - order.stop_loss) + order.price * 2.0 * 0.0005
        risk_cap_qty = wallet * risk_pct / unit_risk if unit_risk > 0 else 0.0
        if min_floor and floor_step > 0 and risk_cap_qty < float(values.get("quantity", 0.0) or 0.0):
            # Giu nguyen lenh toi thieu san: ky quy nang san (~$16 o BTC $79k,
            # lev 5x) chi chiem ~16% vi $99, van trong han muc 35%.
            capped_quantity = min(capped_quantity, float(values["quantity"]))
        else:
            capped_quantity = min(capped_quantity, risk_cap_qty)
        if capped_quantity < values["quantity"]:
            ok, values = s.binance_api.normalize_order_values(order.symbol, capped_quantity, values.get("price"), order_type=kind, reference_price=s.live_price)
            if not ok:
                order.status = "REJECTED"
                detail = values.get("msg", values) if isinstance(values, dict) else values
                self.trace(order.parent_intent_id, "Execution / Exposure", "VETO",
                           f"Tang gui cat size con {capped_quantity:.4f} nhung san tu choi: {detail}. "
                           f"(von ${wallet:,.2f}, risk_pct {risk_pct*100:.2f}%, unit_risk ${unit_risk:,.1f})")
                self.persist()
                return
        order.units = values["quantity"]
        order.price = values.get("price") or order.price
        order.margin = order.units * order.price / order.leverage
        if kind in ("STOP", "TAKE_PROFIT", "TRAILING_STOP_MARKET"):
            trigger = order.trigger_price if kind != "TRAILING_STOP_MARKET" else order.price
            ok, trigger_values = s.binance_api.normalize_order_values(order.symbol, order.units, trigger, order_type=kind,
                reference_price=s.live_price, reduce_only=True, price_rounding="up" if order.trigger_condition == "ABOVE" else "down")
            if not ok:
                order.status = "REJECTED"
                self.trace(order.parent_intent_id, "Execution / Trigger Filter", "VETO", str(trigger_values))
                self.persist()
                return
            if kind == "TRAILING_STOP_MARKET":
                order.price = trigger_values["price"]
            else:
                order.trigger_price = trigger_values["price"]
        reason = self.entry_guard(order)  # REST preparation may outlast the data freshness budget.
        if reason:
            order.status = "REJECTED"
            self.trace(order.parent_intent_id, "Execution / Final Revalidation", "VETO", reason)
            self.persist()
            return
        if not s.order_manager.mark_submit_pending(order.order_id):
            return
        self.persist()
        kwargs = dict(symbol=order.symbol, side=order.side, order_type=kind, quantity=order.units,
                      client_order_id=order.client_order_id, position_side=order.position_side)
        if kind in ("LIMIT", "POST_ONLY", "STOP", "TAKE_PROFIT"):
            kwargs["price"] = order.price
        if kind in ("STOP", "TAKE_PROFIT"):
            kwargs["stop_price"] = order.trigger_price
        if kind == "TRAILING_STOP_MARKET":
            kwargs.update(callback_rate=order.callback_pct, activation_price=order.price)
        ok, response = s.binance_api.place_order_live(**kwargs)
        if not ok:
            if self.uncertain_response(response):
                s.order_manager.mark_submit_unknown(order.order_id)
            else:
                order.status = "REJECTED"
                order.exchange_status = "REJECTED"
            self.trace(order.parent_intent_id, "Execution / Submit", "PENDING" if order.status == "SUBMIT_UNKNOWN" else "VETO",
                "Query stable client ID before retry" if order.status == "SUBMIT_UNKNOWN" else "Binance rejected order", response=response)
            self.persist()
            return
        self.apply_response(order, response)

    def apply_response(self, order, response):
        s = self.state
        if not s.order_manager.record_exchange_update(order.order_id, response):
            self.trace(order.parent_intent_id, "Execution / Reconcile", "PENDING", "Invalid or stale cumulative exchange receipt")
            self.persist()
            return
        quantity, price = s.order_manager.pending_fill(order.order_id)
        if quantity > 0:
            self.apply_entry(order, quantity, price)
            s.order_manager.mark_fill_applied(order.order_id)
            # Any partial execution selects the OCO winner, not just a full fill.
            if order.group_type == "OCO":
                for other in s.order_manager.pending_orders:
                    if other.execution_group == order.execution_group and other.direction != order.direction:
                        other.status = "CANCEL_REQUESTED" if self.live else "CANCELED"
            self.persist()  # position + cumulative receipt are committed together
            if self.feedback_pending:
                self.feedback()
            self.ensure_protection()
        self.trace(order.parent_intent_id, "Execution / Exchange", "PASS" if quantity else "PENDING", str(response.get("status", "ACK")), exchange_order_id=order.exchange_order_id, filled_quantity=order.applied_quantity)
        self.persist()

    def apply_entry(self, order, quantity, price):
        s = self.state
        if not math.isfinite(price) or price <= 0:
            raise ValueError("Confirmed fill has no valid execution price")
        hedge_grid = self.is_hedge_grid(order)
        pos = self.position_for(order.position_side if hedge_grid else None)
        if pos and pos["direction"] != order.direction:
            # Synthetic OCO has a race window; account for both actual fills.
            closed = min(quantity, pos["units"])
            self.realize(closed, price, "OCO race offset", f"oco:{order.client_order_id}:{order.applied_quantity + quantity}", commit=False)
            quantity -= closed
            if quantity <= 1e-10:
                return
            pos = self.position_for(order.position_side if hedge_grid else None)
        maker = order.order_type in ("LIMIT", "POST_ONLY", "SCALE_RATIO")
        fee = s.fee_engine.calculate_fee(quantity * price, is_maker=maker)
        s.current_balance -= fee
        s.total_fees += fee
        payload = order.candidate_payload or {}
        slice_reason = str(getattr(order, "entry_reason", "") or "")[:500]
        if not slice_reason:
            try:
                slice_reason = self.summarize_entry_reason(order, self.traces.get(order.parent_intent_id, []))
            except Exception:
                slice_reason = "Lệnh đã qua đủ cổng kiểm duyệt"
        item = {
            "slice_id": f"{order.client_order_id}:{order.applied_quantity + quantity:.8f}",
            "order_id": order.order_id, "client_order_id": order.client_order_id,
            "exchange_order_id": order.exchange_order_id, "parent_intent_id": order.parent_intent_id,
            "execution_group": order.execution_group, "group_type": order.group_type,
            "direction": order.direction, "side": order.side, "order_type": order.order_type,
            "entry_price": price, "units": quantity, "notional": quantity * price,
            "margin": quantity * price / order.leverage, "leverage": order.leverage, "entry_fee": fee,
            "timeframe": order.timeframe, "entry_time": datetime.now(timezone.utc).isoformat(),
            "open_timestamp": time.time(), "entry_context": payload.get("metadata", {}).get("entry_context", {}),
            "entry_reason": slice_reason,
            "unrealized_pnl": 0.0, "roe_pct": 0.0, "fee_estimated": True,
        }
        if not pos:
            pos = {
                "direction": order.direction, "stop_loss": order.stop_loss, "take_profit": order.take_profit,
                "initial_risk": abs(price - order.stop_loss), "peak_price": price, "orders": [],
                "entry_time": item["entry_time"], "open_timestamp": item["open_timestamp"],
                "timeframe": order.timeframe, "is_manual_tpsl": False, "partial_tp_done": False,
                "trailing_status": "Protected execution", "unrealized_pnl": 0.0, "net_upnl": 0.0,
                "leverage": order.leverage, "protection_revision": 0,
                "entry_reason": slice_reason,
                # initial_margin co dinh de ROE chuan Binance = UPNL / initial margin
                "initial_margin": quantity * price / order.leverage,
                "staged_take_profits": payload.get("staged_take_profits", {}),
                "tp_stage_initial_units": 0.0, "tp_stage_filled": {},
            }
            if hedge_grid:
                s.hedge_positions[order.position_side] = pos
            else:
                s.current_position = pos
        pos["tp_stage_initial_units"] = pos.get("tp_stage_initial_units", pos.get("units", 0.0)) + quantity
        pos["orders"].append(item)
        # Adding inventory must not widen the existing loss boundary.
        pos["stop_loss"] = max(pos["stop_loss"], order.stop_loss) if order.direction == 1 else min(pos["stop_loss"], order.stop_loss)
        self.aggregate(order.position_side if hedge_grid else None)
        s.risk_manager.update_balance(s.current_balance)

    def aggregate(self, position_side=None):
        s = self.state
        pos = self.position_for(position_side)
        if not pos:
            return
        for field in ("units", "margin", "notional", "entry_fee"):
            pos[field] = sum(item.get(field, 0.0) for item in pos["orders"])
        # initial_margin giu co dinh theo tung slice luc vao (khong giam khi chot mot phan),
        # de ROE chuan = UPNL / initial margin nhu Binance.
        if not pos.get("initial_margin"):
            pos["initial_margin"] = pos.get("margin", 0.0)
        if pos["units"] <= 1e-10:
            if position_side in ("LONG", "SHORT"):
                s.hedge_positions.pop(position_side, None)
            else:
                s.current_position = None
            return
        pos["entry_price"] = pos["notional"] / pos["units"]
        pos["breakeven_price"] = s.fee_engine.calculate_breakeven_price(pos["entry_price"], pos["direction"])
        try:
            pos["liq_price"] = s.risk_manager.calculate_liquidation_price(
                pos["entry_price"],
                pos["direction"],
                pos.get("leverage", 3),
                notional_usdt=pos["notional"],
            )
        except TypeError as exc:
            # Keep lightweight test doubles and older integrations compatible;
            # the real risk manager always accepts the notional tier input.
            if "notional_usdt" not in str(exc):
                raise
            pos["liq_price"] = s.risk_manager.calculate_liquidation_price(
                pos["entry_price"], pos["direction"], pos.get("leverage", 3)
            )

    def query_ref(self, intent):
        client = intent["client_order_id"]
        if intent["order_type"] in ("STOP", "TAKE_PROFIT", "STOP_MARKET", "TAKE_PROFIT_MARKET", "TRAILING_STOP_MARKET"):
            client = "algo:" + client
        return dict(order_id=intent.get("order_id") or None, client_order_id=client)

    @staticmethod
    def uncertain_response(response):
        return not isinstance(response, dict) or response.get("code") in (-999, -1001, -1006, -1007)

    def send_reduce(self, intent):
        s = self.state
        intent["status"] = "SUBMIT_UNKNOWN"
        self.persist()
        position_side = intent.get("position_side", "BOTH")
        # Hedge-mode ``closePosition`` is an all-or-nothing exchange primitive.
        # Protective staged take-profits carry a partial quantity, so they must
        # stay quantity-based; only an explicitly marked full stop may use it.
        native_close = bool(intent.get("close_position", False))
        ok, response = s.binance_api.place_order_live(
            s.symbol, intent["side"], intent["order_type"], intent["quantity"],
            stop_price=intent.get("trigger"), reduce_only=position_side == "BOTH", close_position=native_close,
            position_side=position_side, client_order_id=intent["client_order_id"],
        )
        if ok:
            intent["order_id"] = str(response.get("orderId", ""))
            ok = self.apply_exit_response(intent, response)
        elif not self.uncertain_response(response):
            intent["status"] = "REJECTED"
            intent["error"] = response
        self.persist()
        return ok

    def ensure_protection(self):
        s = self.state
        if not self.live or any(e.get("status") not in ("FILLED", "CANCELED", "EXPIRED", "REJECTED") for e in self.exits):
            return
        for position_side, pos in list(self.all_positions()):
            self._ensure_protection_for(pos, position_side)

    def _ensure_protection_for(self, pos, position_side):
        s = self.state
        signature = [pos["units"], pos["stop_loss"], pos["take_profit"]]
        current_ids = set(pos.get("protective_order_ids", []))
        active = [p for p in self.protective if p.get("order_id") in current_ids and p.get("status") in ("NEW", "PARTIALLY_FILLED", "PENDING_TRIGGER") and not p.get("cancel_requested")]
        stop_coverage = sum(max(0, p["quantity"] - p.get("applied", 0)) for p in active if p["order_type"] == "STOP_MARKET")
        take_coverage = sum(max(0, p["quantity"] - p.get("applied", 0)) for p in active if p["order_type"] == "TAKE_PROFIT_MARKET")
        if pos.get("protected_signature") == signature and current_ids and len(active) == len(current_ids) and min(stop_coverage, take_coverage) + 1e-10 >= pos["units"]:
            return
        if any(p.get("status") == "SUBMIT_UNKNOWN" for p in self.protective):
            s.execution_blocker = "Protective order acknowledgement unresolved"
            return
        old = [p for p in self.protective if p.get("position_side", "BOTH") == position_side and p.get("status") in ("NEW", "PARTIALLY_FILLED", "PENDING_TRIGGER")]
        revision = pos.get("protection_revision", 0) + 1
        pos["protection_revision"] = revision
        side = "SELL" if pos["direction"] == 1 else "BUY"
        plan = [("STOP_MARKET", pos["stop_loss"], pos["units"], None)]
        staged = pos.get("staged_take_profits") or {}
        remaining = pos["units"]
        for stage in ("tp1", "tp2"):
            qty = min(remaining, self.staged_tp_remaining(pos, stage))
            trigger = staged.get(stage, 0)
            if qty > 0 and trigger > 0:
                if (trigger - s.live_price) * pos["direction"] <= 0:
                    self.request_close(qty / pos["units"], f"{stage.upper()} staged exit", tp_stage=stage, position_side=position_side)
                    return
                plan.append(("TAKE_PROFIT_MARKET", trigger, qty, stage))
                remaining -= qty
        if remaining > 1e-10:
            plan.append(("TAKE_PROFIT_MARKET", pos["take_profit"], remaining, None))
        created = []
        for kind, trigger, quantity, tp_stage in plan:
            ok, values = s.binance_api.normalize_order_values(s.symbol, quantity, trigger, order_type=kind, reference_price=s.live_price,
                reduce_only=position_side == "BOTH", price_rounding="up" if kind == "STOP_MARKET" and pos["direction"] == 1 else "down")
            if not ok:
                break
            intent = dict(client_order_id=f"p-{uuid.uuid4().hex[:22]}", order_type=kind, side=side, quantity=values["quantity"], trigger=values["price"], status="CREATED", applied=0.0, quote_applied=0.0,
                reason=f"{tp_stage.upper()} staged exit" if tp_stage else "Protective SL/TP", tp_stage=tp_stage, position_side=position_side,
                close_position=(kind == "STOP_MARKET" and quantity >= pos["units"] - 1e-10))
            self.protective.append(intent)
            if not self.send_reduce(intent):
                break
            created.append(intent)
        if len(created) == len(plan) and self.position_for(position_side) is pos:
            previous_signature = pos.get("protected_signature")
            pos["protected_signature"] = signature
            pos["protective_order_ids"] = [p["order_id"] for p in created]
            self.confirm_protection_flags(pos, previous_signature[2] if previous_signature else None)
            for previous in old:
                previous["cancel_requested"] = True
            s.execution_blocker = ""
        else:
            s.execution_blocker = "Protective setup failed; reducing tracked Testnet exposure"
            self.request_close(1.0, s.execution_blocker, position_side=position_side)
        self.persist()

    @staticmethod
    def staged_tp_remaining(pos, stage):
        """Fixed entry-quantity stage budgets, reduced only by attributed fills."""
        staged = pos.get("staged_take_profits") or {}
        ratio = staged.get(f"{stage}_ratio", 0)
        initial = pos.get("tp_stage_initial_units", pos.get("units", 0))
        target = math.floor(initial * max(0, min(1, ratio)) / .001 + 1e-9) * .001
        filled = pos.get("tp_stage_filled", {}).get(stage, 0)
        return min(pos.get("units", 0), max(0.0, target - filled))

    def cancel_pending_entries(self, reason):
        """Persist cancellation intent before a full close; never discard unknown fills."""
        s = self.state
        self.entry_exit_barrier = reason
        for order in list(s.order_manager.pending_orders):
            if order.symbol != s.symbol:
                continue
            if order.status in ("PENDING", "ACTIVE") and not order.exchange_order_id and not order.exchange_status:
                s.order_manager.cancel_order(order.order_id, reason)
            elif order.status in s.order_manager.OPEN_STATUSES:
                order.status = "CANCEL_REQUESTED"
            order.cancel_reason = reason

    def reconcile_close_barrier(self):
        if not self.entry_exit_barrier:
            return
        s = self.state
        self.cancel_pending_entries(self.entry_exit_barrier)
        if s.current_position:
            self.request_close(1.0, self.entry_exit_barrier)
        elif not s.order_manager.pending_orders and not any(
                e.get("status") not in ("FILLED", "CANCELED", "EXPIRED", "REJECTED") for e in self.exits + self.protective):
            self.entry_exit_barrier = ""
        self.persist()

    def request_close(self, ratio, reason, slice_id=None, tp_stage=None, after_stop_loss=None, position_side=None):
        s = self.state
        pos = self.position_for(position_side)
        if not pos or not math.isfinite(ratio) or not 0 < ratio <= 1:
            return False
        if after_stop_loss is not None and (not math.isfinite(after_stop_loss) or after_stop_loss <= 0):
            return False
        if ratio == 1 and not slice_id:
            if position_side in ("LONG", "SHORT"):
                for order in s.order_manager.pending_orders:
                    if order.group_type == "GRID" and order.position_side == position_side:
                        self.cancel(order)
            else:
                self.cancel_pending_entries(reason)
            self.persist()
        if any(e.get("status") not in ("FILLED", "CANCELED", "EXPIRED", "REJECTED") for e in self.exits):
            return False
        target = next((item for item in pos["orders"] if str(item["slice_id"]) == str(slice_id)), None) if slice_id else None
        if slice_id and not target:
            return False
        quantity = target["units"] if target else pos["units"] * ratio
        if self.live:
            ok, values = s.binance_api.normalize_order_values(s.symbol, quantity, order_type="MARKET", reference_price=s.live_price, reduce_only=True)
            if not ok:
                return False
            quantity = values["quantity"]
        intent = dict(client_order_id=f"x-{uuid.uuid4().hex[:22]}", order_type="MARKET", side="SELL" if pos["direction"] == 1 else "BUY", quantity=quantity, status="CREATED", applied=0.0, quote_applied=0.0, reason=reason, slice_id=slice_id, tp_stage=tp_stage,
            after_stop_loss=after_stop_loss, after_stop_applied=False, position_key=pos.get("open_timestamp", pos.get("entry_time")), position_side=position_side or "BOTH")
        self.exits.append(intent)
        if not self.live:
            self.persist()
            self.apply_exit_response(intent, {"orderId": f"paper:{intent['client_order_id']}", "status": "FILLED", "executedQty": quantity, "avgPrice": s.live_price})
            return intent["status"] == "FILLED"
        self.send_reduce(intent)
        return intent["status"] == "FILLED"

    def apply_exit_response(self, intent, response):
        try:
            qty = float(response.get("executedQty", 0) or 0)
            avg = float(response.get("avgPrice", 0) or 0)
            quote = float(response.get("cumQuote", 0) or 0) or qty * avg
        except (AttributeError, TypeError, ValueError, OverflowError):
            return False
        status = response.get("status", intent["status"])
        if (status not in ("NEW", "PARTIALLY_FILLED", "FILLED", "CANCELED", "EXPIRED", "REJECTED", "PENDING_TRIGGER")
                or any(not math.isfinite(v) or v < 0 for v in (qty, avg, quote))
                or qty > intent["quantity"] + 1e-9
                or qty < intent.get("applied", 0) or quote < intent.get("quote_applied", 0)
                or (qty > 0 and quote <= 0) or (status == "FILLED" and qty <= 0)):
            return False
        if intent["status"] in ("FILLED", "CANCELED", "EXPIRED", "REJECTED") and status in ("NEW", "PARTIALLY_FILLED", "PENDING_TRIGGER"):
            return False
        delta = qty - intent.get("applied", 0.0)
        if delta > 1e-10:
            price = (quote - intent.get("quote_applied", 0)) / delta
            if not math.isfinite(price) or price <= 0:
                return False
            self.realize(delta, price, intent["reason"], f"{intent['client_order_id']}:{qty:.8f}", intent.get("slice_id"), commit=False,
                         tp_stage=intent.get("tp_stage"), position_side=intent.get("position_side"))
            intent["applied"] = qty
            intent["quote_applied"] = quote
        intent["status"] = status
        intent["order_id"] = str(response.get("orderId", intent.get("order_id", "")))
        self.persist()
        if self.feedback_pending:
            self.feedback()
        self.apply_after_close_protection(intent)
        return True

    def apply_after_close_protection(self, intent):
        """A persisted partial-exit follow-up runs only after a confirmed fill."""
        stop = intent.get("after_stop_loss")
        if stop is None or intent.get("after_stop_applied") or intent.get("applied", 0) <= 0:
            return
        if intent.get("position_side") in ("LONG", "SHORT"):
            return
        pos = self.state.current_position
        if not pos or pos.get("open_timestamp", pos.get("entry_time")) != intent.get("position_key"):
            intent["after_stop_applied"] = True  # Never apply a stale follow-up to a new position.
            self.persist()
            return
        stop = max(stop, pos["stop_loss"]) if pos["direction"] == 1 else min(stop, pos["stop_loss"])
        ok, _ = self.replace_protection(sl=stop)
        if ok:
            intent["after_stop_applied"] = True
            self.persist()

    def realize(self, quantity, price, reason, execution_id, slice_id=None, commit=True, tp_stage=None, position_side=None):
        s = self.state
        if not all(math.isfinite(v) and v > 0 for v in (quantity, price)):
            raise ValueError("Invalid realized fill quantity/price")
        pos = self.position_for(position_side)
        if not pos or execution_id in self.processed_execution_ids:
            return
        ordered = sorted(pos["orders"], key=lambda item: 0 if str(item["slice_id"]) == str(slice_id) else 1)
        remaining = min(quantity, pos["units"])
        realized_quantity = remaining
        if remaining >= pos["units"] - 1e-10:
            if position_side in ("LONG", "SHORT"):
                for order in s.order_manager.pending_orders:
                    if order.group_type == "GRID" and order.position_side == position_side:
                        self.cancel(order)
            else:
                self.cancel_pending_entries(reason)
        total_take = 0.0
        total_entry_fee = 0.0
        total_margin = 0.0
        total_gross = 0.0
        sum_entry_val = 0.0
        slice_ids = []
        earliest_entry_time = None
        earliest_open_ts = None
        first_ctx = {}
        parent_intent_id = ""
        exec_group = ""
        grp_type = ""
        timeframe = pos.get("timeframe", s.active_timeframe)

        for item in ordered:
            if remaining <= 1e-10:
                break
            take = min(remaining, item["units"])
            ratio = take / item["units"]
            entry_fee = item["entry_fee"] * ratio
            margin = item["margin"] * ratio
            gross = (price - item["entry_price"]) * take * pos["direction"]

            total_take += take
            total_entry_fee += entry_fee
            total_margin += margin
            total_gross += gross
            sum_entry_val += item["entry_price"] * take
            slice_ids.append(str(item.get("slice_id", "")))
            if earliest_entry_time is None or (item.get("entry_time") and item.get("entry_time") < earliest_entry_time):
                earliest_entry_time = item.get("entry_time")
                earliest_open_ts = item.get("open_timestamp")
            if not first_ctx and item.get("entry_context"):
                first_ctx = item.get("entry_context")
            if not parent_intent_id and item.get("parent_intent_id"):
                parent_intent_id = item.get("parent_intent_id")
            if not exec_group and item.get("execution_group"):
                exec_group = item.get("execution_group")
            if not grp_type and item.get("group_type"):
                grp_type = item.get("group_type")
            if item.get("timeframe"):
                timeframe = item.get("timeframe")

            for field in ("units", "notional", "margin", "entry_fee"):
                item[field] *= 1 - ratio
            remaining -= take

        if total_take > 1e-10:
            avg_entry_price = sum_entry_val / total_take
            # Phi thoat tinh theo taker (bao thu, nhat quan voi unrealized tru phi uoc tinh).
            # entry_fee da tru vao balance luc mo lenh (apply_entry) nen khong tru lan 2 o day,
            # nhung van cong vao tong phi hien thi va net de PnL phan anh dung chi phi that.
            exit_fee = s.fee_engine.calculate_fee(price * total_take, is_maker=False)
            net = total_gross - total_entry_fee - exit_fee
            combined_slice_id = ":".join(slice_ids) if len(slice_ids) <= 2 else f"{slice_ids[0]}+({len(slice_ids)-1})"
            trade = dict(
                id=max([t["id"] for t in s.trades] or [0]) + 1,
                execution_id=f"{execution_id}:{combined_slice_id}",
                symbol=s.symbol,
                timeframe=timeframe,
                direction="LONG" if pos["direction"] == 1 else "SHORT",
                entry_time=earliest_entry_time or datetime.now(timezone.utc).isoformat(),
                open_timestamp=earliest_open_ts,
                entry_price=round(avg_entry_price, 2),
                breakeven_price=pos["breakeven_price"],
                exit_time=datetime.now(timezone.utc).isoformat(),
                closed_at_ts=time.time(),
                exit_price=price,
                units=round(total_take, 8),
                fee=round(total_entry_fee + exit_fee, 4),
                pnl=round(net, 4),
                return_pct=round(net / total_margin * 100, 2) if total_margin else 0.0,
                reason=reason,
                entry_context=first_ctx,
                parent_intent_id=parent_intent_id,
                execution_group=exec_group,
                group_type=grp_type,
                fee_estimated=True
            )
            s.current_balance += total_gross - exit_fee
            s.total_fees += exit_fee
            s.trades.append(trade)
            s.jesse_engine.record_trade(net)
            s.risk_manager.ai_cro.record_trade_result(net)
            s.freqtrade_protections.on_trade_closed(trade)
            if first_ctx:
                s.trade_memory.record_trade_outcome(
                    trade_id=trade["id"],
                    direction=pos["direction"],
                    entry_price=avg_entry_price,
                    indicators=first_ctx.get("indicators", {}),
                    of_data=first_ctx.get("of_data", {}),
                    smc_data=first_ctx.get("smc_data", {}),
                    vwap_data=first_ctx.get("vwap_data", {}),
                    net_pnl=net,
                    exit_reason=reason
                )
            shadow = s.shadow_account.analyze_trade_history(s.trades, s.initial_balance)
            self.trace(trade["parent_intent_id"], "Stage 6 / Shadow Account", "PASS", "Realized fill feedback", **asdict(shadow))
        pos["orders"] = [item for item in pos["orders"] if item["units"] > 1e-10]
        if tp_stage in ("tp1", "tp2"):
            stages = pos.setdefault("tp_stage_filled", {})
            stages[tp_stage] = stages.get(tp_stage, 0.0) + realized_quantity
        pos["partial_tp_done"] = True
        self.aggregate(position_side)
        s.risk_manager.update_balance(s.current_balance)
        s.last_trade_closed_time = time.time()
        self.processed_execution_ids.add(execution_id)
        self.feedback_pending = True
        # Commit realized fills before Monthly reads the database.
        if commit:
            self.persist()
            self.feedback()

    def feedback(self):
        if not self.feedback_pending:
            return
        s = self.state
        governor = s.monthly_governor.evaluate(s.current_balance, s.trades)
        if s.trades:
            trade = s.trades[-1]
            self.trace(trade.get("parent_intent_id", ""), "Stage 7 / Monthly Governor", "PASS", "Multiplier for next candidate", **asdict(governor))
        self.feedback_pending = False
        self.persist()

    def replace_protection(self, sl=None, tp=None, manual=False):
        s = self.state
        pos = s.current_position
        if not pos:
            return False, "No open position"
        stop, target = pos["stop_loss"] if sl is None else sl, pos["take_profit"] if tp is None else tp
        if not all(math.isfinite(v) and v > 0 for v in (stop, target)):
            return False, "SL/TP must be finite and positive"
        if (s.live_price - stop) * pos["direction"] <= 0 or (target - s.live_price) * pos["direction"] <= 0:
            return False, "SL/TP are on the wrong side of current price"
        if (stop - pos["stop_loss"]) * pos["direction"] < 0:
            return False, "Widening risk beyond the approved stop is prohibited"
        previous = (pos["stop_loss"], pos["take_profit"])
        pos["stop_loss"], pos["take_profit"] = stop, target
        self.ensure_protection()
        if self.live and pos.get("protected_signature") != [pos["units"], stop, target]:
            pos["stop_loss"], pos["take_profit"] = previous
            return False, "Replacement not confirmed; prior protection retained"
        if s.current_position is not pos:
            return False, "Position closed while replacement was being acknowledged"
        self.confirm_protection_flags(pos, previous[1])
        pos["is_manual_tpsl"] = manual
        self.persist()
        return True, "Protection updated"

    @staticmethod
    def confirm_protection_flags(pos, previous_target=None):
        """Flags describe acknowledged protection, not a model's proposal."""
        direction = pos["direction"]
        pos["is_risk_free"] = (pos["stop_loss"] - pos.get("breakeven_price", pos["entry_price"])) * direction >= -1e-9
        if previous_target is not None and (pos["take_profit"] - previous_target) * direction > 1e-9:
            pos["tp_expanded"] = True

    def cancel(self, order):
        s = self.state
        if not self.live or order.status in ("PENDING", "ACTIVE"):
            result = s.order_manager.cancel_order(order.order_id)
            self.persist()
            return result
        order.status = "CANCEL_REQUESTED"
        self.persist()
        ref = {"order_type": self.exchange_type(order), "order_id": order.exchange_order_id, "client_order_id": order.client_order_id}
        ok, response = s.binance_api.cancel_order(order.symbol, **self.query_ref(ref))
        if not ok:
            # A cancel can lose the race to a fill; query before leaving it unresolved.
            ok, response = s.binance_api.query_order(order.symbol, **self.query_ref(ref))
        if ok:
            self.apply_response(order, response)
            if response.get("status") in ("CANCELED", "EXPIRED", "REJECTED"):
                s.order_manager.mark_exchange_cancelled(order.order_id, "Exchange cancellation confirmed")
                self.persist()
                return True
        return False

    def verify_startup_inventory(self):
        if self.startup_reconciled:
            return True
        s = self.state
        ok, positions = s.binance_api._send_request("GET", "/fapi/v3/positionRisk", {"symbol": s.symbol})
        reason = "Startup exchange inventory reconciliation unavailable"
        if ok and isinstance(positions, list):
            try:
                relevant = [p for p in positions if p.get("symbol") == s.symbol]
                actual = {"BOTH": 0.0, "LONG": 0.0, "SHORT": 0.0}
                for item in relevant:
                    side = str(item.get("positionSide", "BOTH")).upper()
                    if side not in actual:
                        raise ValueError("unknown position side")
                    actual[side] += float(item["positionAmt"])
                expected = {"BOTH": (s.current_position or {}).get("units", 0) * (s.current_position or {}).get("direction", 0),
                            "LONG": getattr(s, "hedge_positions", {}).get("LONG", {}).get("units", 0),
                            "SHORT": -getattr(s, "hedge_positions", {}).get("SHORT", {}).get("units", 0)}
                valid = all(math.isfinite(v) for v in actual.values())
                if valid and all(abs(actual[side] - expected[side]) <= 1e-9 for side in actual):
                    self.startup_reconciled = True
                    if getattr(s, "execution_blocker", "").startswith("Startup exchange inventory"):
                        s.execution_blocker = ""
                    self.persist()
                    return True
                reason = "Startup exchange inventory differs from durable ledger; no automatic adoption"
            except (TypeError, KeyError, ValueError, OverflowError):
                pass
        s.execution_blocker = reason
        self.persist()
        return False

    def tick(self):
        s = self.state
        if not self.live:
            self.startup_reconciled = False
            self.reconcile_close_barrier()
            for intent in self.exits:
                self.apply_after_close_protection(intent)
            for order in list(s.order_manager.due_orders()):
                guard_reason = self.entry_guard(order)
                if guard_reason:
                    s.order_manager.cancel_order(order.order_id, f"Lệnh chờ hết hiệu lực: {guard_reason}")
            for order in s.order_manager.match_orders(s.live_price, s.fee_engine.bid_price, s.fee_engine.ask_price):
                self.apply_response(order, {"orderId": f"paper:{order.order_id}", "status": "FILLED",
                    "executedQty": order.units, "avgPrice": order.price})
            self.persist()
            return
        if time.time() - self.last_reconcile < 1.0:
            return
        self.last_reconcile = time.time()
        for intent in self.protective + self.exits:
            if intent.get("status") in ("FILLED", "CANCELED", "EXPIRED", "REJECTED"):
                continue
            if not self.position_for(intent.get("position_side")) or intent.get("cancel_requested") or (self.entry_exit_barrier and intent in self.protective):
                ok, response = s.binance_api.cancel_order(s.symbol, **self.query_ref(intent))
                if not ok:
                    ok, response = s.binance_api.query_order(s.symbol, **self.query_ref(intent))
            else:
                ok, response = s.binance_api.query_order(s.symbol, **self.query_ref(intent))
            if ok:
                self.apply_exit_response(intent, response)
        for intent in self.exits:
            self.apply_after_close_protection(intent)
        if self.feedback_pending:
            self.feedback()
        for order in list(s.order_manager.pending_orders):
            if order.status == "CANCEL_REQUESTED":
                self.cancel(order)
            elif order.status not in ("PENDING", "ACTIVE"):
                ref = dict(order_type=self.exchange_type(order), order_id=order.exchange_order_id, client_order_id=order.client_order_id)
                ok, response = s.binance_api.query_order(order.symbol, **self.query_ref(ref))
                if ok:
                    self.apply_response(order, response)
        self.reconcile_close_barrier()
        self.ensure_protection()
        inventory_verified = self.verify_startup_inventory()
        unknown_entries = any(order.status in ("SUBMIT_PENDING", "SUBMIT_UNKNOWN") for order in s.order_manager.pending_orders)
        if not getattr(s, "execution_blocker", "") and inventory_verified and not unknown_entries and not self.entry_exit_barrier:
            for order in list(s.order_manager.due_orders()):
                self.submit(order)
                if order.status in ("SUBMIT_PENDING", "SUBMIT_UNKNOWN"):
                    break
