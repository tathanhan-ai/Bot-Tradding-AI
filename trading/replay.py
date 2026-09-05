"""Offline, fixed-parameter replay through the live decision and execution code.

Run in a SEPARATE process (the replay clock temporarily patches module clocks)::

    python -m trading.replay D:\\recordings\\BTCUSDT.jsonl --split 0.7

Each JSONL line is a full market snapshot: captured_at (UTC epoch seconds),
snapshot_id, symbol=BTCUSDT, exchange=binance, price, source_times (depth,
agg_trade, kline_1m), bids/asks (20 actual [price, quantity] levels), frames
{timeframe: [{time: candle OPEN epoch seconds, open, high, low, close, volume,
is_closed: true}]}, and incremental agg_trades [{id, timestamp, price, quantity,
is_buyer_maker}]. Include warmup trades in the first snapshot of EACH split.
provenance.kind must be 'recorded' or 'test_only'; the latter NEVER qualifies
as performance evidence. Frames must include 1m/5m/15m/1h closed OHLCV.

No depth/flow is fabricated from OHLCV. No network, credentials, real account
state, fitting, or parameter changes are used. Both windows start flat with
fresh risk memory; supplied historical closed bars provide indicator warmup.
LLM is intentionally unavailable, so this measures the deterministic fallback.
Taker fills consume recorded L2 with VWAP; maker fills retain the shared paper
touch model (queue priority is unknown). Funding, latency and the path between
snapshots are not modeled. Consequently this is a validation harness, not proof
of profitability or approval to deploy/tune parameters.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack, contextmanager, redirect_stdout
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import importlib
import io
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from trading.pipeline import DecisionTrace, MarketSnapshot, StageOutcome, candle_end


class ReplayDataError(ValueError):
    """Recording is insufficient for an honest replay; never synthesize a fix."""


def _seconds(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ReplayDataError("timestamps must be finite positive UTC epochs")
    return result / 1000 if result > 1e11 else result


def snapshot_from_event(event: dict) -> MarketSnapshot:
    """Validate every supplied candle (not only a tail), without repairing data."""
    try:
        now = _seconds(event["captured_at"])
        frames = {}
        for tf, rows in event["frames"].items():
            if not rows or any(row.get("is_closed") is not True for row in rows):
                raise ReplayDataError(f"{tf}: explicit closed-candle evidence is required")
            frame = pd.DataFrame(rows)
            frame.index = pd.to_datetime([_seconds(row["time"]) for row in rows], unit="s")
            if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
                raise ReplayDataError(f"{tf}: candles must be unique and chronological")
            if any(candle_end(ts, tf) > pd.Timestamp(now, unit="s") for ts in frame.index):
                raise ReplayDataError(f"{tf}: look-ahead / forming candle")
            values = frame[["open", "high", "low", "close", "volume"]].astype(float)
            if any(not math.isfinite(v) for v in values.to_numpy().flat):
                raise ReplayDataError(f"{tf}: non-finite OHLCV")
            if ((values[["open", "high", "low", "close"]] <= 0).any().any()
                    or (values.volume < 0).any()
                    or (values.high < values[["open", "close", "low"]].max(axis=1)).any()
                    or (values.low > values[["open", "close", "high"]].min(axis=1)).any()):
                raise ReplayDataError(f"{tf}: invalid OHLCV geometry")
            frames[tf] = values
        if event.get("provenance", {}).get("kind") not in ("recorded", "test_only"):
            raise ReplayDataError("explicit provenance.kind recorded/test_only is required")
        if event.get("symbol") != "BTCUSDT":
            raise ReplayDataError("this baseline is scoped to BTCUSDT")
        sources = {key: _seconds(value) for key, value in event["source_times"].items()}
        if any(value > now for value in sources.values()):
            raise ReplayDataError("future source timestamp")
        snapshot = MarketSnapshot(
            snapshot_id=str(event["snapshot_id"]), symbol=event["symbol"],
            exchange=event["exchange"], captured_at=now, price=float(event["price"]),
            source_times=sources, bids=[[float(p), float(q)] for p, q in event.get("bids", [])],
            asks=[[float(p), float(q)] for p, q in event.get("asks", [])], frames=frames,
            environment="paper", context={"active_timeframe": "15m", "required_frames": ["1m", "5m", "15m", "1h"]},
        )
        issues = snapshot.freshness_issues(now)
        if issues:
            raise ReplayDataError("; ".join(issues))
        return snapshot
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        if isinstance(exc, ReplayDataError):
            raise
        raise ReplayDataError(f"invalid snapshot: {exc}") from exc


def load_events(path) -> list[dict]:
    events = []
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
                snapshot_from_event(event)
                events.append(event)
            except (json.JSONDecodeError, ReplayDataError) as exc:
                raise ReplayDataError(f"line {line_number}: {exc}") from exc
    return events


class ReplayStorage:
    """Per-run memory only. Never restore a user's SQLite account/credentials."""
    def __init__(self):
        self.settings = {"is_live_enabled": False, "is_testnet": True, "active_exchange": "binance"}
        self.runtime = None
        self.trades = {}

    def get_all_settings(self):
        return deepcopy(self.settings)

    def get_setting(self, key, default=None):
        return deepcopy(self.settings.get(key, default))

    def save_setting(self, key, value):
        self.settings[key] = deepcopy(value)

    def get_vibe_config(self):
        return {"enabled": False, "min_votes": 3}

    def load_account_state(self):
        return None

    def load_trades(self, limit=200):
        return deepcopy(list(self.trades.values())[-limit:])

    def load_trade_memory_records(self, limit=40):
        return []

    def load_execution_runtime(self):
        return deepcopy(self.runtime)

    def save_execution_runtime(self, payload):
        self.runtime = deepcopy(payload)
        before = len(self.trades)
        for trade in payload.get("trades", []):
            self.trades.setdefault(trade["execution_id"], deepcopy(trade))
        return len(self.trades) - before

    def get_monthly_realized_pnl(self, year_month):
        return sum(t["pnl"] for t in self.trades.values()
                   if datetime.fromtimestamp(t["closed_at_ts"], timezone.utc).strftime("%Y-%m") == year_month)


def _state_class():
    # server's global state is initialized at import; isolate even that state.
    if "ui.server" not in sys.modules:
        from data.persistent_storage import PersistentStorageManager
        with patch("data.persistent_storage.PersistentStorageManager", return_value=ReplayStorage()):
            server = importlib.import_module("ui.server")
        server.PersistentStorageManager = PersistentStorageManager
    return sys.modules["ui.server"].LiveTradingState


@contextmanager
def replay_clock(clock):
    """Offline only: restore all clocks/network guards, including on exceptions."""
    real_datetime = datetime

    class ReplayDateTime(real_datetime):
        @classmethod
        def now(cls, tz=None):
            value = cls.fromtimestamp(clock.value, timezone.utc)
            return value.astimezone(tz) if tz else value.replace(tzinfo=None)

        @classmethod
        def utcnow(cls):
            return cls.fromtimestamp(clock.value, timezone.utc).replace(tzinfo=None)

    with ExitStack() as stack:
        stack.enter_context(patch("time.time", lambda: clock.value))
        stack.enter_context(patch("socket.socket.connect", side_effect=RuntimeError("replay prohibits network access")))
        for name, module in list(sys.modules.items()):
            if name.startswith(("ui.", "risk.", "strategy.", "trading.")) and getattr(module, "datetime", None) is real_datetime:
                stack.enter_context(patch.object(module, "datetime", ReplayDateTime))
        yield


def depth_vwap(levels, quantity):
    remaining, quote = float(quantity), 0.0
    if not math.isfinite(remaining) or remaining <= 0:
        raise ReplayDataError("invalid execution quantity")
    for price, available in levels:
        take = min(remaining, available)
        quote += take * price
        remaining -= take
        if remaining <= 1e-10:
            return quote / quantity
    raise ReplayDataError("recorded L2 is insufficient for fill; no extrapolation allowed")


class RecordedExecution:
    """Price adapter only; shared lifecycle remains the owner of fills/feedback."""
    def __init__(self, state):
        self.state = state
        self.fills = []
        self.maker_fills = 0
        self.book_timestamp = None
        self.bids, self.asks = [], []
        original_entry, original_exit = state.execution.apply_entry, state.execution.realize

        def entry(order, quantity, price):
            maker = order.order_type in ("LIMIT", "POST_ONLY", "SCALE_RATIO")
            fill_price = price if maker else self.price(order.direction, quantity)
            original_entry(order, quantity, fill_price)
            self.record(order.direction, quantity, fill_price, maker)

        def realize(quantity, price, reason, execution_id, slice_id=None, commit=True, tp_stage=None):
            if not state.current_position or execution_id in state.execution.processed_execution_ids:
                return
            quantity = min(quantity, state.current_position["units"])
            direction = -state.current_position["direction"]
            fill_price = self.price(direction, quantity)
            original_exit(quantity, fill_price, reason, execution_id, slice_id, commit, tp_stage=tp_stage)
            self.record(direction, quantity, fill_price, False)

        state.execution.apply_entry, state.execution.realize = entry, realize

    def price(self, direction, quantity):
        timestamp = self.state.market_source_times["depth"]
        if timestamp != self.book_timestamp:
            self.bids, self.asks = deepcopy(self.state.latest_l2_bids), deepcopy(self.state.latest_l2_asks)
            self.book_timestamp = timestamp
        levels = self.asks if direction == 1 else self.bids
        price = depth_vwap(levels, quantity)
        remaining = quantity
        for level in levels:
            take = min(remaining, level[1])
            level[1] -= take
            remaining -= take
            if remaining <= 1e-10:
                break
        return price

    def record(self, direction, quantity, price, maker):
        mid = (self.state.latest_l2_asks[0][0] + self.state.latest_l2_bids[0][0]) / 2
        self.fills.append({"quantity": quantity, "price": price, "notional": quantity * price,
                           "slippage_usdt": (price - mid) * quantity * direction, "maker": maker})
        self.maker_fills += int(maker)


def performance_metrics(initial_balance, final_balance, equity, trades, fills):
    peak, max_dd = initial_balance, 0.0
    for balance in equity:
        peak = max(peak, balance)
        max_dd = max(max_dd, (peak - balance) / peak if peak > 0 else 0)
    gains = sum(max(0, t["pnl"]) for t in trades)
    losses = -sum(min(0, t["pnl"]) for t in trades)
    notional = sum(f["notional"] for f in fills)
    slippage = sum(f["slippage_usdt"] for f in fills)
    return {"net_return_pct": (final_balance / initial_balance - 1) * 100,
            "profit_factor": gains / losses if losses > 0 else None,
            "profit_factor_status": "DEFINED" if losses else "NO_LOSSES_OR_NO_TRADES",
            "max_drawdown_pct": max_dd * 100, "turnover_multiple": notional / initial_balance,
            "traded_notional_usdt": notional, "slippage_usdt": slippage,
            "slippage_bps": slippage / notional * 1e4 if notional else 0,
            "fees_usdt": sum(t.get("fee", 0) for t in trades), "closed_fills": len(trades)}


def _feed(state, snapshot, event, seen, clock):
    latest = state.market_source_times.get("agg_trade", 0)
    for trade in event.get("agg_trades", []):
        try:
            key = str(trade["id"])
            ts = _seconds(trade["timestamp"])
            price, quantity = float(trade["price"]), float(trade["quantity"])
            maker = trade["is_buyer_maker"]
        except (KeyError, ValueError, TypeError) as exc:
            raise ReplayDataError(f"invalid aggTrade schema: {exc}") from exc
        if ts > snapshot.captured_at:
            raise ReplayDataError("future aggTrade / look-ahead")
        if key in seen:
            continue
        if ts < latest:
            raise ReplayDataError("aggTrades must be chronological, without unseen late arrivals")
        clock.value = ts
        if not state.order_flow_engine.add_trade(price, quantity, maker, ts * 1000):
            raise ReplayDataError("invalid recorded aggTrade")
        state.visual_hft.update_trade(price, quantity, maker, ts)
        seen.add(key)
        latest = ts
    clock.value = snapshot.captured_at
    if not seen or abs(snapshot.source_times["agg_trade"] - latest) > 0.001:
        raise ReplayDataError("raw aggTrade missing or source timestamp inconsistent (warm up each split)")
    state.data_map = {tf: frame.copy(deep=True) for tf, frame in snapshot.frames.items()}
    state.market_source_times = deepcopy(snapshot.source_times)
    state.latest_l2_bids, state.latest_l2_asks = deepcopy(snapshot.bids), deepcopy(snapshot.asks)
    state.visual_hft.update_order_book(snapshot.bids, snapshot.asks, snapshot.source_times["depth"])
    state.fee_engine.update_book(snapshot.bids[0][0], snapshot.asks[0][0])
    state.order_flow_verdict = state.order_flow_engine.evaluate(snapshot.price)
    state.on_tick(snapshot.price)


def _baseline_row(state, snapshot):
    bar = snapshot.frames["15m"].index[-1]
    if getattr(state, "_replay_baseline_bar", None) != bar:
        signals = state.strategy.generate_signals(snapshot.frames)
        state._replay_baseline_bar = bar
        state._replay_baseline_row = None if signals.empty else signals.iloc[-1]
    return state._replay_baseline_row


def _baseline_tick(state, snapshot):
    pos = state.current_position
    if not pos:
        return
    price, direction = snapshot.price, pos["direction"]
    row = _baseline_row(state, snapshot)
    if ((price - pos["stop_loss"]) * direction <= 0
            or (price - pos["take_profit"]) * direction >= 0):
        state.execution.request_close(1.0, "MTF-ATR SL/TP")
        return
    if row is not None and int(row["signal"]) == -direction:
        state.execution.request_close(1.0, "MTF-ATR reversal")
        return
    pos["peak_price"] = max(price, pos["peak_price"]) if direction == 1 else min(price, pos["peak_price"])
    if row is not None and (pos["peak_price"] - pos["entry_price"]) * direction >= pos["initial_risk"]:
        trailing = pos["peak_price"] - direction * float(row["atr"])
        # Same 1R activation / one ATR trailing as the repository baseline,
        # but only observed snapshot prices, never retrospective candle highs.
        stop = (max(pos["stop_loss"], pos["entry_price"], trailing) if direction == 1
                else min(pos["stop_loss"], pos["entry_price"], trailing))
        state.execution.replace_protection(sl=stop)


def _baseline_decision(state, snapshot, last_bar):
    bar = snapshot.frames["15m"].index[-1]
    if bar == last_bar:
        return last_bar
    row = _baseline_row(state, snapshot)
    if row is None or int(row["signal"]) == 0:
        return bar
    direction = int(row["signal"])
    candidate = state.build_candidate_order(direction, "MARKET", snapshot.price, float(row.stop_loss), float(row.take_profit), "baseline")
    candidate.leverage = state.risk_config.default_leverage
    proposal = state.risk_manager.evaluate_order(state.symbol, direction, snapshot.price, candidate.stop_loss, candidate.take_profit, candidate.leverage)
    if not proposal.approved:
        return bar
    candidate.quantity, candidate.margin = proposal.units, proposal.required_margin
    # The comparator intentionally replaces alpha/sizing, not data/execution guards.
    state.last_decision_trace = DecisionTrace(candidate.order_id, snapshot.snapshot_id)
    state.last_decision_trace.add("Baseline / MTF-ATR", StageOutcome.pass_("Fixed baseline signal and FuturesRiskManager sizing; NOT seven-stage approval"))
    state.queue_approved_candidate(candidate)
    return bar


def _run_window(events, mode, initial_balance, state_type):
    clock = SimpleNamespace(value=_seconds(events[0]["captured_at"]))
    with replay_clock(clock):
        state = state_type(symbol="BTCUSDT", balance=initial_balance, storage=ReplayStorage())
        state.vibe_swarm.enabled = False
        if state.execution.live:
            raise RuntimeError("replay must never enable exchange execution")
        pricing = RecordedExecution(state)
        seen, equity, traces, vetoes = set(), [initial_balance], [], Counter()
        last_bar = None
        for event in events:
            snapshot = snapshot_from_event(event)
            _feed(state, snapshot, event, seen, clock)
            state.execution.tick()
            if mode == "pipeline":
                state.process_execution_tick(snapshot.price)
                prior = state.last_decision_trace
                state.evaluate_ensemble_automated_decision(snapshot.price)
                if state.last_decision_trace is not prior:
                    trace = state.last_decision_trace
                    for entry in trace.entries:
                        entry.timestamp = clock.value
                        if entry.verdict == "VETO":
                            vetoes[entry.stage + ": " + entry.reason] += 1
                    traces.append(asdict(trace))
            else:
                _baseline_tick(state, snapshot)
                if not state.current_position and not state.order_manager.pending_orders:
                    last_bar = _baseline_decision(state, snapshot, last_bar)
            pos = state.current_position
            unrealized = (snapshot.price - pos["entry_price"]) * pos["units"] * pos["direction"] if pos else 0
            equity.append(state.current_balance + unrealized)
        for order in list(state.order_manager.pending_orders):
            state.execution.cancel(order)
        if state.current_position:
            state.execution.request_close(1.0, "Replay boundary liquidation at last observed L2")
        equity.append(state.current_balance)
        metrics = performance_metrics(initial_balance, state.current_balance, equity, state.trades, pricing.fills)
        return {"metrics": metrics, "decision_traces": traces, "veto_counts": dict(vetoes),
                "maker_touch_fills": pricing.maker_fills, "equity": equity,
                "trades": deepcopy(state.trades), "fills": pricing.fills,
                "fees": {"maker_rate": state.fee_engine.maker_fee_rate, "taker_rate": state.fee_engine.taker_fee_rate}}


def run_replay(events, split_fraction=0.7, initial_balance=5000.0):
    """Return IS/OOS comparisons; fail on missing data, never repair/invent it."""
    if len(events) < 2 or not 0 < split_fraction < 1 or not math.isfinite(initial_balance) or initial_balance <= 0:
        raise ReplayDataError("need >=2 snapshots, 0<split<1 and positive finite capital")
    times, closed_history = [], {}
    for event in events:
        snapshot = snapshot_from_event(event)
        times.append(snapshot.captured_at)
        for tf, frame in snapshot.frames.items():
            for row in frame.itertuples():
                key, values = (tf, row.Index), tuple(row[1:])
                if key in closed_history and values != closed_history[key]:
                    raise ReplayDataError(f"{tf}: previously observed closed candle was rewritten")
                closed_history[key] = values
    if any(a >= b for a, b in zip(times, times[1:])):
        raise ReplayDataError("snapshots must be strictly chronological")
    split = times[0] + (times[-1] - times[0]) * split_fraction
    cut = next(i for i, timestamp in enumerate(times) if timestamp >= split)
    if cut == 0:
        raise ReplayDataError("empty in-sample window")
    with redirect_stdout(io.StringIO()):
        state_type = _state_class()
        results = {name: {mode: _run_window(window, mode, initial_balance, state_type)
                          for mode in ("pipeline", "mtf_atr")}
                   for name, window in (("in_sample", events[:cut]), ("out_of_sample", events[cut:]))}
    a, b = (results["out_of_sample"][mode]["metrics"] for mode in ("pipeline", "mtf_atr"))
    test_only = any(event["provenance"]["kind"] == "test_only" for event in events)
    improves = (a["net_return_pct"] > b["net_return_pct"]
                and a["max_drawdown_pct"] <= b["max_drawdown_pct"]
                and a["closed_fills"] > 0 and b["closed_fills"] > 0)
    return {"status": "TEST_ONLY_NOT_PERFORMANCE_EVIDENCE" if test_only else "RECORDED_REPLAY",
            "split_timestamp": split, "split_policy": "chronological, flat reset per window, no fitted parameters",
            "llm_policy": "LLM_UNAVAILABLE: deterministic fallback only", "results": results,
            "comparison": {"oos_return_improved_without_worse_drawdown": improves,
                           "parameters_changed": False, "tuning_performed": False, "promotion_allowed": False},
            "limitations": ["Input provenance is declared by recorder, not independently authenticated.",
                            "Maker queue priority, funding and latency are not modeled; no profitability claim.",
                            "Unobserved price paths between snapshots cannot trigger protective exits.",
                            "No tuning/promotion evidence from test fixtures; validate recorded multi-regime OOS first."],
            "max_snapshot_gap_seconds": max(b - a for a, b in zip(times, times[1:]))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--split", type=float, default=0.7)
    parser.add_argument("--balance", type=float, default=5000.0)
    args = parser.parse_args(argv)
    try:
        report = run_replay(load_events(args.recording), args.split, args.balance)
    except (OSError, ReplayDataError) as exc:
        print(json.dumps({"status": "DATA_UNAVAILABLE", "reason": str(exc), "promotion_allowed": False}))
        return 2
    print(json.dumps(report, ensure_ascii=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
