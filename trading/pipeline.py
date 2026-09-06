"""Fail-closed seven-stage order pipeline shared by automated and manual flow."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import math
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd


class DecisionMode(str, Enum):
    AI_REQUIRED = "AI_REQUIRED"
    DETERMINISTIC_ONLY = "DETERMINISTIC_ONLY"
    EXIT_ONLY = "EXIT_ONLY"

    def __str__(self) -> str:
        return self.value


@dataclass
class DecisionTraceEntry:
    stage: str
    verdict: str
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class DecisionTrace:
    order_id: str
    snapshot_id: str
    entries: List[DecisionTraceEntry] = field(default_factory=list)

    @property
    def vetoed(self) -> bool:
        return any(entry.verdict == "VETO" for entry in self.entries)

    def add(self, stage: str, outcome: "StageOutcome") -> None:
        self.entries.append(DecisionTraceEntry(stage, outcome.verdict, outcome.reason, {**outcome.details, **outcome.updates}))


@dataclass
class MarketSnapshot:
    snapshot_id: str
    symbol: str
    exchange: str
    captured_at: float
    price: float
    source_times: Dict[str, float]
    bids: List[List[float]]
    asks: List[List[float]]
    cvd: float = 0.0
    vpin: float = 0.0
    frames: Dict[str, pd.DataFrame] = field(default_factory=dict, repr=False)
    context: Dict[str, Any] = field(default_factory=dict, repr=False)
    environment: str = "paper"

    def freshness_issues(self, now: Optional[float] = None) -> List[str]:
        now = time.time() if now is None else now
        issues: List[str] = []
        exchange = self.exchange.lower()
        if exchange not in ("binance", "mexc"):
            issues.append(f"unsupported exchange={self.exchange}")
        if exchange == "mexc" and self.environment != "paper":
            issues.append("MEXC market data is paper-only")
        if not math.isfinite(self.price) or self.price <= 0:
            issues.append("invalid price")
        if len(self.bids) < 20 or len(self.asks) < 20:
            issues.append("real 20-level L2 depth is incomplete")
        if self.bids and self.asks:
            if self.bids[0][0] >= self.asks[0][0]:
                issues.append("crossed L2 book")
            if any(not math.isfinite(v) or v <= 0 for level in self.bids + self.asks for v in level):
                issues.append("invalid L2 price/quantity")
            if any(a[0] <= b[0] for a, b in zip(self.bids, self.bids[1:])) or any(a[0] >= b[0] for a, b in zip(self.asks, self.asks[1:])):
                issues.append("L2 levels are not unique and sorted")
        for tf in self.context.get("required_frames", []):
            frame = self.frames.get(tf)
            if frame is None or len(frame) < 50:
                issues.append(f"insufficient closed {tf} history")
                continue
            recent = frame.tail(50)
            ends = [candle_end(ts, tf) for ts in recent.index]
            now_ts = pd.Timestamp(now, unit="s")
            last_closed_at = ends[-1]
            if last_closed_at > now_ts + pd.Timedelta(seconds=5):
                issues.append(f"forming/future {tf} candle")
            tf_delta = candle_end(last_closed_at, tf) - last_closed_at
            allowed_delay = pd.Timedelta(seconds=5)
            if last_closed_at < now_ts - tf_delta - allowed_delay:
                issues.append(f"stale {tf} history")
            if any(end != next_open for end, next_open in zip(ends, recent.index[1:])):
                issues.append(f"gap in {tf} closed candles")
            values = recent[["open", "high", "low", "close", "volume"]]
            if (values.isna().any().any() or any(not math.isfinite(float(v)) for v in values.to_numpy().flat)
                    or (values[["open", "high", "low", "close"]] <= 0).any().any() or (values["volume"] < 0).any()
                    or (values["high"] < values[["open", "close", "low"]].max(axis=1)).any()
                    or (values["low"] > values[["open", "close", "high"]].min(axis=1)).any()):
                issues.append(f"invalid {tf} OHLCV")
        for name, maximum_age in (("depth", 3.0), ("agg_trade", 3.0), ("kline_1m", 90.0)):
            timestamp = self.source_times.get(name)
            if timestamp is None or not math.isfinite(timestamp) or timestamp <= 0:
                issues.append(f"missing {name} timestamp")
                continue
            timestamp = timestamp / 1000.0 if timestamp > 100_000_000_000 else timestamp
            age = now - timestamp
            if age < -2.0 or age > maximum_age:
                issues.append(f"{name} age {age:.1f}s exceeds {maximum_age:.0f}s")
        return issues


@dataclass
class CandidateOrder:
    order_id: str
    symbol: str
    direction: int
    order_type: str
    entry_price: float
    stop_loss: float
    take_profit: float
    leverage: int = 1
    quantity: float = 0.0
    margin: float = 0.0
    source: str = "auto"
    alpha_score: float = 0.0
    confidence: float = 0.0
    regime: str = ""
    staged_take_profits: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StageOutcome:
    verdict: str
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)
    updates: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def pass_(cls, reason: str, **updates: Any) -> "StageOutcome":
        return cls("PASS", reason, updates=updates)

    @classmethod
    def veto(cls, reason: str, **details: Any) -> "StageOutcome":
        return cls("VETO", reason, details=details)

    @classmethod
    def unavailable(cls, reason: str, **details: Any) -> "StageOutcome":
        return cls("LLM_UNAVAILABLE", reason, details=details)


@dataclass
class PipelineDecision:
    candidate: CandidateOrder
    trace: DecisionTrace

    @property
    def approved(self) -> bool:
        return not self.trace.vetoed


StageGate = Callable[[CandidateOrder, MarketSnapshot], StageOutcome]


class SevenStagePipeline:
    """Runs one mutable candidate through the mandatory decision order."""

    def __init__(
        self,
        defense_gates: Sequence[Tuple[str, StageGate]],
        stage2_gates: Sequence[Tuple[str, StageGate]] = (),
        council: Optional[StageGate] = None,
        sizing: Optional[StageGate] = None,
        memory: Optional[StageGate] = None,
        decision_mode: str = DecisionMode.AI_REQUIRED,
    ):
        self.defense_gates = list(defense_gates)
        self.stage2_gates = list(stage2_gates)
        self.council = council
        self.sizing = sizing
        self.memory = memory
        if isinstance(decision_mode, DecisionMode):
            self.decision_mode = decision_mode
        elif isinstance(decision_mode, str):
            clean = decision_mode.replace("DecisionMode.", "")
            try:
                self.decision_mode = DecisionMode(clean)
            except ValueError:
                self.decision_mode = DecisionMode.AI_REQUIRED
        else:
            self.decision_mode = DecisionMode.AI_REQUIRED

    def decide(self, candidate: CandidateOrder, snapshot: MarketSnapshot, now: Optional[float] = None, trace: Optional[DecisionTrace] = None) -> PipelineDecision:
        trace = trace or DecisionTrace(candidate.order_id, snapshot.snapshot_id)
        is_risk_reducing = (
            candidate.metadata.get("is_exit") is True
            or candidate.metadata.get("reduce_only") is True
            or candidate.source == "exit"
            or candidate.metadata.get("intent") in ("REDUCE", "EXIT", "CLOSE_ALL", "EMERGENCY_CLOSE")
        )

        if self.decision_mode == DecisionMode.EXIT_ONLY and not is_risk_reducing:
            trace.add("Safety Governor", StageOutcome.veto("Pipeline in EXIT_ONLY mode; risk-increasing entries are prohibited"))
            return PipelineDecision(candidate, trace)

        snapshot_issues = snapshot.freshness_issues(now)
        if candidate.symbol != snapshot.symbol:
            snapshot_issues.append("candidate and snapshot symbols differ")
        is_grid_plan = candidate.source in ("manual-grid", "auto-grid") and candidate.order_type == "GRID" and candidate.direction == 0
        if candidate.direction not in (-1, 1) and not (candidate.source in ("auto", "auto-grid") and candidate.direction == 0) and not is_grid_plan:
            snapshot_issues.append("invalid candidate direction")
        if any(not math.isfinite(x) or x < 0 for x in (candidate.entry_price, candidate.stop_loss, candidate.take_profit)):
            snapshot_issues.append("non-finite or negative candidate price")
        if snapshot_issues:
            trace.add("Stage 0 / Market Snapshot", StageOutcome.veto("; ".join(snapshot_issues)))
            return PipelineDecision(candidate, trace)
        trace.add("Stage 0 / Market Snapshot", StageOutcome.pass_("real-time snapshot is fresh", snapshot_id=snapshot.snapshot_id, captured_at=snapshot.captured_at, source_times=snapshot.source_times, environment=snapshot.environment))

        if not self.defense_gates:
            trace.add("Stage 1 / Defense Gates", StageOutcome.veto("defense gates are not configured"))
            return PipelineDecision(candidate, trace)
        for name, gate in self.defense_gates:
            if self._run_gate(trace, f"Stage 1 / {name}", gate, candidate, snapshot):
                return PipelineDecision(candidate, trace)

        if not self.stage2_gates:
            trace.add("Stage 2 / Alpha & Regime", StageOutcome.veto("alpha/regime gates are not configured"))
            return PipelineDecision(candidate, trace)
        for name, gate in self.stage2_gates:
            if self._run_gate(trace, f"Stage 2 / {name}", gate, candidate, snapshot):
                return PipelineDecision(candidate, trace)

        if is_risk_reducing:
            trace.add("Stage 3 / AI Council", StageOutcome.pass_("Risk-reducing operation; AI Council evaluation bypassed for safe exit"))
        elif self.decision_mode == DecisionMode.DETERMINISTIC_ONLY:
            trace.add("Stage 3 / AI Council", StageOutcome.pass_("DETERMINISTIC_ONLY mode active: AI Council bypassed"))
        elif self.council is None:
            trace.add("Stage 3 / AI Council", StageOutcome.veto("Fail-Closed: AI Council is required in AI_REQUIRED mode but not configured"))
            return PipelineDecision(candidate, trace)
        else:
            try:
                council_outcome = self.council(candidate, snapshot)
            except Exception as exc:
                council_outcome = StageOutcome.veto(f"Fail-Closed: AI Council error: {type(exc).__name__}")
            if council_outcome.verdict != "PASS":
                council_outcome = StageOutcome.veto(council_outcome.reason or "Fail-Closed: AI Council did not PASS", **council_outcome.details)
                self._apply(candidate, council_outcome)
                trace.add("Stage 3 / AI Council", council_outcome)
                return PipelineDecision(candidate, trace)
            self._apply(candidate, council_outcome)
            trace.add("Stage 3 / AI Council", council_outcome)

        if self.sizing is None:
            trace.add("Stage 4 / Carver + Risk", StageOutcome.veto("sizing gate is not configured"))
            return PipelineDecision(candidate, trace)
        if self._run_gate(trace, "Stage 4 / Carver + Risk", self.sizing, candidate, snapshot):
            return PipelineDecision(candidate, trace)

        if self.memory is None:
            trace.add("Stage 5 / Trade Memory", StageOutcome.veto("trade-memory gate is not configured"))
            return PipelineDecision(candidate, trace)
        self._run_gate(trace, "Stage 5 / Trade Memory", self.memory, candidate, snapshot)
        return PipelineDecision(candidate, trace)

    @staticmethod
    def _apply(candidate: CandidateOrder, outcome: StageOutcome) -> None:
        for key, value in outcome.updates.items():
            if key in CandidateOrder.__dataclass_fields__:
                setattr(candidate, key, value)
            else:
                candidate.metadata[key] = value

    def _run_gate(
        self,
        trace: DecisionTrace,
        stage: str,
        gate: StageGate,
        candidate: CandidateOrder,
        snapshot: MarketSnapshot,
    ) -> bool:
        try:
            outcome = gate(candidate, snapshot)
            if not isinstance(outcome, StageOutcome):
                raise TypeError("gate did not return StageOutcome")
            if outcome.verdict not in ("PASS", "VETO"):
                outcome = StageOutcome.veto(f"invalid mandatory gate verdict: {outcome.verdict}")
        except Exception as exc:
            outcome = StageOutcome.veto(f"gate error: {type(exc).__name__}", error=str(exc)[:200])
        self._apply(candidate, outcome)
        trace.add(stage, outcome)
        return outcome.verdict == "VETO"


def apply_closed_kline(frames: Dict[str, pd.DataFrame], timeframe: str, kline: Dict[str, Any]) -> bool:
    """Upsert a confirmed Binance OHLCV bar; forming or synthetic bars are ignored."""
    required = ("time", "open", "high", "low", "close", "volume", "quote_volume")
    if not kline.get("is_closed") or any(key not in kline for key in required):
        return False
    try:
        values = {key: float(kline[key]) for key in required if key != "time"}
        timestamp = float(kline["time"])
    except (TypeError, ValueError):
        return False
    if timestamp <= 0 or any(not math.isfinite(value) or value < 0 for value in values.values()):
        return False
    if values["high"] < max(values["open"], values["close"]) or values["low"] > min(values["open"], values["close"]):
        return False

    index = pd.to_datetime(int(timestamp), unit="s", utc=True).tz_localize(None)
    row = pd.DataFrame([values], index=[index])
    frame = frames.get(timeframe)
    if frame is None or frame.empty:
        frames[timeframe] = row
    else:
        frames[timeframe] = pd.concat([frame.drop(index=index, errors="ignore"), row]).sort_index().iloc[-500:]
    return True


def candle_end(open_time, timeframe):
    start = pd.Timestamp(open_time)
    if timeframe == "1M":
        return start + pd.DateOffset(months=1)
    unit = {"m": "minutes", "h": "hours", "d": "days", "w": "weeks"}[timeframe[-1]]
    return start + pd.Timedelta(**{unit: int(timeframe[:-1])})


def resample_closed_candles(frame: pd.DataFrame, timeframe: str, now: float) -> pd.DataFrame:
    """Aggregate contiguous closed 1m OHLCV; incomplete buckets never enter alpha."""
    rule = {"1M": "MS", "1w": "W-MON"}.get(timeframe, timeframe.replace("m", "min"))
    grouped = frame.resample(rule, label="left", closed="left")
    result = grouped.agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "quote_volume": "sum"})
    counts = grouped["close"].count()
    valid = []
    for start in result.index:
        end = candle_end(start, timeframe)
        expected = int((end - start).total_seconds() / 60)
        valid.append(end <= pd.Timestamp(now, unit="s") and counts.loc[start] == expected)
    return result.loc[valid].dropna()
