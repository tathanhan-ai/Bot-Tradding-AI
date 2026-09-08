# -*- coding: utf-8 -*-
"""Four independent, role-scoped 9Router council calls."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
import urllib.request


@dataclass
class AgentVote:
    agent_id: str
    agent_name: str
    icon: str
    model_used: str
    vote: str                 # APPROVE, REJECT, or ABSTAIN
    confidence: int
    thesis: str
    timestamp: str = ""


@dataclass
class SwarmCouncilVerdict:
    council_verdict: str      # APPROVED, REJECTED, or LLM_UNAVAILABLE
    consensus_passed: bool
    approvals_count: int
    total_agents: int = 4
    min_votes_required: int = 3
    votes: List[AgentVote] = field(default_factory=list)
    council_rationale: str = ""
    timestamp: str = ""
    available: bool = False
    availability_reason: str = ""
    order_id: str = ""
    snapshot_id: str = ""
    can_negotiate: bool = False
    rejection_categories: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def approved(self) -> bool:
        return self.consensus_passed

    @property
    def approved_votes(self) -> int:
        return self.approvals_count

    @property
    def total_votes(self) -> int:
        return self.total_agents


class VibeSwarmCouncil:
    AGENT_META = {
        "macro": ("🌐 Macro & Regime Agent", "🌐"),
        "quant": ("🔬 Quant Alpha Agent", "🔬"),
        "risk": ("🛡️ Risk Governor Agent", "🛡️"),
        "execution": ("⚡ Execution Specialist", "⚡"),
    }

    def __init__(
        self,
        gateway_url: str = "http://127.0.0.1:8039/v1",
        default_model: str = "ag/gemini-3.8-flash",
        api_key: Optional[str] = None,
        min_votes_required: int = 3,
        enabled: bool = True,
    ):
        self.gateway_url = gateway_url.rstrip("/")
        self.default_model = default_model
        import os
        # P0: khong key cung trong source. Key lay tu tham so > settings (UI) > moi truong.
        # Thieu key -> AI_REQUIRED tu choi entry (fail-closed), chi chay deterministic khi dat ro mode.
        self.api_key = api_key or self._load_key_from_settings() or os.environ.get("NINEROUTER_API_KEY")
        self.min_votes_required = max(2, min(4, min_votes_required))
        self.enabled = enabled
        self.agent_models = {agent_id: default_model for agent_id in self.AGENT_META}
        self.last_verdict: Optional[SwarmCouncilVerdict] = None
        self.last_completed_at = 0.0
        self._background_pending = False
        self._background_lock = threading.Lock()
        self._queued_kwargs: Dict[str, Any] = {}
        self._has_queued = False

    @property
    def latest_verdict(self) -> Optional[SwarmCouncilVerdict]:
        return self.last_verdict

    @latest_verdict.setter
    def latest_verdict(self, value: Optional[SwarmCouncilVerdict]) -> None:
        self.last_verdict = value

    @property
    def consensus_threshold(self) -> int:
        return self.min_votes_required

    @consensus_threshold.setter
    def consensus_threshold(self, value: int) -> None:
        self.min_votes_required = max(2, min(4, int(value)))

    @staticmethod
    def _load_key_from_settings() -> Optional[str]:
        # Doc key 9Router do nguoi dung nhap trong Settings (luu qua SecretProvider, khong trong source).
        try:
            from strategy.ninerouter_key import load_ninerouter_key
            return load_ninerouter_key()
        except Exception:
            return None

    def refresh_api_key(self, api_key: Optional[str] = None) -> None:
        import os
        self.api_key = api_key or self._load_key_from_settings() or os.environ.get("NINEROUTER_API_KEY")

    def update_config(
        self,
        enabled: Optional[bool] = None,
        min_votes: Optional[int] = None,
        macro_model: Optional[str] = None,
        quant_model: Optional[str] = None,
        risk_model: Optional[str] = None,
        exec_model: Optional[str] = None,
        default_model: Optional[str] = None,
    ) -> None:
        if enabled is not None:
            self.enabled = bool(enabled)
        if min_votes is not None:
            self.min_votes_required = max(2, min(4, int(min_votes)))
        if default_model:
            self.default_model = default_model
        for agent_id, model in {
            "macro": macro_model,
            "quant": quant_model,
            "risk": risk_model,
            "execution": exec_model,
        }.items():
            if model:
                self.agent_models[agent_id] = model

    def evaluate_council(
        self,
        current_price: float,
        indicators: Dict[str, Any],
        ai_verdict: Any,
        ensemble_result: Any,
        order_research: Any,
        alpha_zoo_metrics: Optional[Any] = None,
        visual_hft_metrics: Optional[Any] = None,
        jesse_metrics: Optional[Any] = None,
        octobot_metrics: Optional[Any] = None,
        current_position: Optional[Dict[str, Any]] = None,
        user_instruction: str = "",
        candidate: Optional[Any] = None,
        snapshot: Optional[Any] = None,
    ) -> SwarmCouncilVerdict:
        if not self.enabled:
            verdict = self._unavailable_verdict("AI Council disabled")
            verdict.order_id = getattr(candidate, "order_id", "")
            verdict.snapshot_id = getattr(snapshot, "snapshot_id", "")
            self.last_verdict = verdict
            self.last_completed_at = time.time()
            return verdict

        contexts = self._role_contexts(
            current_price, indicators, ai_verdict, ensemble_result, order_research,
            alpha_zoo_metrics, visual_hft_metrics, jesse_metrics, octobot_metrics,
        )
        if candidate:
            proposal = {key: getattr(candidate, key) for key in ("order_id", "symbol", "direction", "order_type", "entry_price", "stop_loss", "take_profit", "leverage", "quantity")}
            for context in contexts.values():
                context["candidate"] = proposal
            contexts["quant"]["alpha_score"] = candidate.alpha_score
            contexts["quant"]["proposed_side"] = "BUY" if candidate.direction == 1 else "SELL"
            contexts["risk"]["order"] = proposal
            contexts["execution"]["order"] = proposal
            if snapshot:
                contexts["execution"]["depth20"] = {"bids": snapshot.bids, "asks": snapshot.asks}
                contexts["execution"]["snapshot_id"] = snapshot.snapshot_id
        results: Dict[str, Tuple[AgentVote, bool]] = {}
        with ThreadPoolExecutor(max_workers=4, thread_name_prefix="vibe-council") as pool:
            futures = {
                pool.submit(self._request_agent, agent_id, contexts[agent_id]): agent_id
                for agent_id in self.AGENT_META
            }
            for future in as_completed(futures):
                agent_id = futures[future]
                try:
                    results[agent_id] = future.result()
                except Exception as exc:
                    results[agent_id] = (self._abstain_vote(agent_id, f"9Router error: {type(exc).__name__}"), False)

        votes = [results[agent_id][0] for agent_id in self.AGENT_META]
        all_available = all(results[agent_id][1] for agent_id in self.AGENT_META)
        if not all_available:
            verdict = self._unavailable_verdict("one or more independent 9Router calls failed", votes)
            verdict.order_id = getattr(candidate, "order_id", "")
            verdict.snapshot_id = getattr(snapshot, "snapshot_id", "")
            self.last_verdict = verdict
            # Quick recovery: allow next council run after only 4 seconds if recovering from timeout/failure
            self.last_completed_at = time.time() - 26.0
            return verdict

        approvals = sum(vote.vote == "APPROVE" for vote in votes)
        passed = approvals >= self.min_votes_required
        rejection_categories: Dict[str, List[str]] = {
            "risk": [],
            "macro": [],
            "execution": [],
            "quant": []
        }
        if not passed:
            for v in votes:
                if v.vote == "REJECT":
                    agent = v.agent_id.lower()
                    cat = "risk" if "risk" in agent else ("macro" if "macro" in agent else ("execution" if ("exec" in agent or "execution" in agent) else "quant"))
                    rejection_categories[cat].append(v.thesis)

        verdict = SwarmCouncilVerdict(
            council_verdict="APPROVED" if passed else "REJECTED",
            consensus_passed=passed,
            approvals_count=approvals,
            min_votes_required=self.min_votes_required,
            votes=votes,
            council_rationale=f"Independent council: {approvals}/4 APPROVE votes (threshold {self.min_votes_required}/4).",
            timestamp=datetime.now().strftime("%H:%M:%S"),
            available=True,
            can_negotiate=not passed,
            rejection_categories=rejection_categories,
        )
        self.last_verdict = verdict
        self.last_completed_at = time.time()
        verdict.order_id = getattr(candidate, "order_id", "")
        verdict.snapshot_id = getattr(snapshot, "snapshot_id", "")
        return verdict

    def schedule_council(self, **kwargs: Any) -> Optional[SwarmCouncilVerdict]:
        """Run optional LLM review off the market-data event loop."""
        if not self.enabled:
            return self._unavailable_verdict("AI Council disabled")
        with self._background_lock:
            if self._background_pending:
                # Giữ lại yêu cầu mới nhất thay vì drop: đánh dấu pending để worker lấy sau khi xong
                self._queued_kwargs = kwargs
                self._has_queued = True
                return None
            self._background_pending = True

        def run(first_kwargs: Dict[str, Any]) -> None:
            try:
                self.evaluate_council(**first_kwargs)
            finally:
                with self._background_lock:
                    queued = self._has_queued
                    next_kwargs = self._queued_kwargs
                    self._queued_kwargs = {}
                    self._has_queued = False
                    if not queued:
                        self._background_pending = False
            if queued:
                try:
                    self.evaluate_council(**next_kwargs)
                finally:
                    with self._background_lock:
                        self._background_pending = False

        threading.Thread(target=run, args=(kwargs,), name="vibe-council-async", daemon=True).start()
        return None

    def _role_contexts(
        self,
        current_price: float,
        indicators: Dict[str, Any],
        ai_verdict: Any,
        ensemble_result: Any,
        order_research: Any,
        alpha_zoo_metrics: Any,
        visual_hft_metrics: Any,
        jesse_metrics: Any,
        octobot_metrics: Any,
    ) -> Dict[str, Dict[str, Any]]:
        tactical_formation = getattr(order_research, "tactical_formation", "DEFENSIVE_SNIPER") if order_research else "DEFENSIVE_SNIPER"
        staged_exits = getattr(order_research, "staged_exits", []) if order_research else []
        sleeve_allocation = getattr(order_research, "sleeve_allocation", {}) if order_research else {}
        fee_tier = getattr(order_research, "fee_tier", "MAKER (0.02%)") if order_research else "MAKER (0.02%)"

        order = {
            "side": getattr(order_research, "recommended_side", "UNKNOWN"),
            "type": getattr(order_research, "recommended_type", "UNKNOWN"),
            "entry": getattr(order_research, "optimal_price", current_price),
            "stop_loss": getattr(order_research, "structural_sl", 0.0),
            "take_profit": getattr(order_research, "structural_tp", 0.0),
            "risk_reward": getattr(order_research, "rr_ratio", 0.0),
            "leverage": getattr(order_research, "optimal_leverage", 0),
            "margin": getattr(order_research, "optimal_margin", 0.0),
            "tactical_formation": tactical_formation,
            "staged_exits": staged_exits,
            "sleeve_allocation": sleeve_allocation,
            "fee_tier": fee_tier,
        }
        def serialise(value: Any) -> Dict[str, Any]:
            return asdict(value) if value and hasattr(value, "__dataclass_fields__") else {}
        return {
            "macro": {
                "price": current_price,
                "regime": getattr(ai_verdict, "regime", "UNKNOWN"),
                "mtf_radar": getattr(ai_verdict, "mtf_radar", {}),
                "ensemble": {"score": getattr(ensemble_result, "consensus_score", 0.0), "verdict": getattr(ensemble_result, "consensus_verdict", "UNKNOWN")},
                "octobot": {"state": getattr(octobot_metrics, "consensus_state", "UNKNOWN"), "direction": getattr(octobot_metrics, "recommended_direction", 0)},
                "tactical_formation": tactical_formation,
                "proposed_side": order["side"],
            },
            "quant": {
                "price": current_price,
                "indicators": {key: indicators.get(key) for key in ("rsi", "adx", "atr", "hurst", "gk_vol")},
                "alpha_zoo": serialise(alpha_zoo_metrics),
                "order_flow": {"vpin": getattr(visual_hft_metrics, "vpin", None), "lob_imbalance_20": getattr(visual_hft_metrics, "lob_imbalance_20", None)},
                "proposed_side": order["side"],
                "tactical_formation": tactical_formation,
            },
            "risk": {
                "order": order,
                "jesse": serialise(jesse_metrics),
                "risk_status": {"consecutive_losses": getattr(jesse_metrics, "current_consecutive_losses", 0), "expectancy": getattr(jesse_metrics, "expectancy_usdt", 0.0)},
                "tactical_formation": tactical_formation,
                "sleeve_allocation": sleeve_allocation,
            },
            "execution": {
                "order": order,
                "microstructure": {"vpin": getattr(visual_hft_metrics, "vpin", None), "resilience": getattr(visual_hft_metrics, "market_resilience_pct", None), "lob_imbalance_20": getattr(visual_hft_metrics, "lob_imbalance_20", None)},
                "carver": getattr(order_research, "carver_output", {}),
                "fee_tier": fee_tier,
            },
        }

    def _request_agent(self, agent_id: str, context: Dict[str, Any], max_attempts: int = 2) -> Tuple[AgentVote, bool]:
        configured_model = self.agent_models[agent_id]
        fallback_model = self.default_model if self.default_model != configured_model else "ag/gemini-3.8-flash"
        last_exc: Optional[Exception] = None

        for attempt in range(1, max_attempts + 1):
            current_model = configured_model if attempt == 1 else fallback_model
            timeout = 18.0 if attempt == 1 else 10.0
            payload = {
                "model": current_model,
                "temperature": 0.0,
                "stream": False,
                "messages": [
                    {"role": "system", "content": f"You are the {agent_id} trading reviewer. Use only the supplied role-scoped data. Return JSON only: {{\"vote\": \"APPROVE|REJECT|ABSTAIN\", \"confidence\": 0-100, \"thesis\": \"short reason\"}}."},
                    {"role": "user", "content": json.dumps(context, ensure_ascii=False, separators=(",", ":"))},
                ],
            }
            try:
                request = urllib.request.Request(
                    f"{self.gateway_url}/chat/completions",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
                )
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    body_bytes = response.read()
                raw = ""
                try:
                    response_data = json.loads(body_bytes.decode("utf-8"))
                    raw = response_data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                except Exception:
                    # Fallback for SSE streaming
                    text = body_bytes.decode("utf-8", errors="replace")
                    parts = []
                    for line in text.splitlines():
                        if line.startswith("data: ") and line.strip() != "data: [DONE]":
                            try:
                                chunk = json.loads(line[6:])
                                c_part = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                if c_part:
                                    parts.append(c_part)
                            except Exception:
                                pass
                    raw = "".join(parts).strip()

                raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
                parsed = json.loads(raw)
                vote = str(parsed.get("vote", "ABSTAIN")).upper()
                vote = vote if vote in ("APPROVE", "REJECT", "ABSTAIN") else "ABSTAIN"
                try:
                    confidence = max(0, min(100, int(parsed.get("confidence", 0))))
                except (TypeError, ValueError):
                    confidence = 0
                thesis = str(parsed.get("thesis", "No thesis returned"))
                if attempt > 1:
                    thesis = f"[Retry qua {current_model}] " + thesis
                return self._vote(agent_id, vote, confidence, thesis, model_used=current_model), True
            except Exception as exc:
                last_exc = exc
                if attempt < max_attempts:
                    try:
                        print(f"[9Router Auto-Retry] Agent '{agent_id}' ({current_model}) bi {type(exc).__name__}. Retrying qua '{fallback_model}' ngay va luon...", flush=True)
                    except Exception:
                        pass
                    continue

        return self._abstain_vote(agent_id, f"9Router unavailable: {type(last_exc).__name__}", model_used=configured_model), False

    def _vote(self, agent_id: str, vote: str, confidence: int, thesis: str, model_used: Optional[str] = None) -> AgentVote:
        name, icon = self.AGENT_META[agent_id]
        m = model_used or self.agent_models[agent_id]
        return AgentVote(agent_id, name, icon, f"9router/{m}", vote, confidence, thesis[:500], datetime.now().strftime("%H:%M:%S"))

    def _abstain_vote(self, agent_id: str, thesis: str, model_used: Optional[str] = None) -> AgentVote:
        return self._vote(agent_id, "ABSTAIN", 0, thesis, model_used=model_used)

    def _unavailable_verdict(self, reason: str, votes: Optional[List[AgentVote]] = None) -> SwarmCouncilVerdict:
        votes = votes or [self._abstain_vote(agent_id, reason) for agent_id in self.AGENT_META]
        return SwarmCouncilVerdict(
            council_verdict="LLM_UNAVAILABLE",
            consensus_passed=False,
            approvals_count=sum(vote.vote == "APPROVE" for vote in votes),
            min_votes_required=self.min_votes_required,
            votes=votes,
            council_rationale="LLM unavailable; no synthetic approvals were created.",
            timestamp=datetime.now().strftime("%H:%M:%S"),
            available=False,
            availability_reason=reason,
        )

    def _fallback_cognitive_swarm(self, *args: Any, **kwargs: Any) -> SwarmCouncilVerdict:
        """Compatibility entry point; deterministic logic must not impersonate LLM votes."""
        return self._unavailable_verdict("9Router fallback is intentionally ABSTAIN-only")
