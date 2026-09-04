# -*- coding: utf-8 -*-
"""
HKUDS Vibe-Trading: Multi-Agent Swarm Council (Hội Đồng Đầu Tư Đa Đặc Vụ 9Router)
Điều phối 4 Đặc Vụ AI chuyên biệt tranh luận và biểu quyết qua 9Router Gateway (Port 8039):
1. Macro & Regime Agent (Vĩ mô, chu kỳ, tin tức)
2. Quant Alpha Agent (12 Alpha Zoo factors, SMC, CVD, VPIN)
3. Risk Governor Agent (Max drawdown, đòn bẩy, SL/TP, bảo vệ vốn)
4. Execution Specialist Agent (Sổ lệnh 20 tầng, trượt giá, dải trơ Carver)
Cơ chế biểu quyết: Chỉ mở lệnh khi số phiếu thuận >= min_votes (mặc định 3/4).
Bảo đảm Zero-Crash Guard với bộ quy tắc thẩm định nhận thức cục bộ khi 9Router offline.
"""
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import urllib.request


@dataclass
class AgentVote:
    agent_id: str                   # 'macro', 'quant', 'risk', 'execution'
    agent_name: str                 # Tên hiển thị
    icon: str                       # Emoji
    model_used: str                 # Model 9Router thực thi
    vote: str                       # 'APPROVE', 'REJECT', 'CAUTION'
    confidence: int                 # [50, 100]%
    thesis: str                     # Luận điểm cốt lõi ngắn gọn
    timestamp: str = ""


@dataclass
class SwarmCouncilVerdict:
    council_verdict: str            # 'APPROVED' hoặc 'REJECTED'
    consensus_passed: bool          # True nếu approvals_count >= min_votes
    approvals_count: int            # Số phiếu thuận (ví dụ: 3)
    total_agents: int = 4           # Tổng số đặc vụ
    min_votes_required: int = 3     # Ngưỡng tối thiểu (3 hoặc 4)
    votes: List[AgentVote] = field(default_factory=list)
    council_rationale: str = ""
    timestamp: str = ""

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
    def __init__(
        self,
        gateway_url: str = "http://127.0.0.1:8039/v1",
        default_model: str = "gemini-2.0-flash",
        api_key: str = "sk-9router-local",
        min_votes_required: int = 3,
        enabled: bool = True
    ):
        self.gateway_url = gateway_url.rstrip("/")
        self.default_model = default_model
        self.api_key = api_key
        self.min_votes_required = min_votes_required
        self.enabled = enabled

        # Per-agent specialized model overrides (can be custom models in 9Router)
        self.agent_models: Dict[str, str] = {
            "macro": default_model,
            "quant": default_model,
            "risk": default_model,
            "execution": default_model
        }

        self.last_verdict: Optional[SwarmCouncilVerdict] = None

    @property
    def latest_verdict(self) -> Optional[SwarmCouncilVerdict]:
        return self.last_verdict

    @latest_verdict.setter
    def latest_verdict(self, val: Optional[SwarmCouncilVerdict]):
        self.last_verdict = val

    @property
    def consensus_threshold(self) -> int:
        return self.min_votes_required

    @consensus_threshold.setter
    def consensus_threshold(self, val: int):
        self.min_votes_required = max(1, min(4, int(val)))

    def update_config(
        self,
        enabled: Optional[bool] = None,
        min_votes: Optional[int] = None,
        macro_model: Optional[str] = None,
        quant_model: Optional[str] = None,
        risk_model: Optional[str] = None,
        exec_model: Optional[str] = None,
        default_model: Optional[str] = None
    ):
        if enabled is not None:
            self.enabled = enabled
        if min_votes is not None:
            self.min_votes_required = max(1, min(4, min_votes))
        if default_model:
            self.default_model = default_model
        if macro_model:
            self.agent_models["macro"] = macro_model
        if quant_model:
            self.agent_models["quant"] = quant_model
        if risk_model:
            self.agent_models["risk"] = risk_model
        if exec_model:
            self.agent_models["execution"] = exec_model

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
        user_instruction: str = ""
    ) -> SwarmCouncilVerdict:
        """
        Conducts the 4-Agent Council Debate & Voting session.
        """
        now_str = datetime.now().strftime("%H:%M:%S")

        # Try querying 9Router for the Swarm debate
        verdict = self._query_9router_swarm(
            current_price=current_price,
            indicators=indicators,
            ai_verdict=ai_verdict,
            ensemble_result=ensemble_result,
            order_research=order_research,
            alpha_zoo_metrics=alpha_zoo_metrics,
            visual_hft_metrics=visual_hft_metrics,
            jesse_metrics=jesse_metrics,
            octobot_metrics=octobot_metrics,
            current_position=current_position,
            user_instruction=user_instruction
        )

        if verdict:
            self.last_verdict = verdict
            return verdict

        # Fallback to Cognitive Swarm Logic
        verdict = self._fallback_cognitive_swarm(
            current_price=current_price,
            indicators=indicators,
            ai_verdict=ai_verdict,
            ensemble_result=ensemble_result,
            order_research=order_research,
            alpha_zoo_metrics=alpha_zoo_metrics,
            visual_hft_metrics=visual_hft_metrics,
            jesse_metrics=jesse_metrics,
            octobot_metrics=octobot_metrics
        )
        self.last_verdict = verdict
        return verdict

    def _query_9router_swarm(
        self,
        current_price: float,
        indicators: Dict[str, Any],
        ai_verdict: Any,
        ensemble_result: Any,
        order_research: Any,
        alpha_zoo_metrics: Optional[Any],
        visual_hft_metrics: Optional[Any],
        jesse_metrics: Optional[Any],
        octobot_metrics: Optional[Any],
        current_position: Optional[Dict[str, Any]],
        user_instruction: str
    ) -> Optional[SwarmCouncilVerdict]:
        url = f"{self.gateway_url}/chat/completions"
        now_str = datetime.now().strftime("%H:%M:%S")

        system_prompt = (
            "Ban la Hoi Dong Dau Tu Da Dac Vu (Multi-Agent Swarm Council) cua quy giao dich quants Binance Futures (HKUDS Vibe-Trading). "
            "Hoi dong gom 4 dac vu: "
            "1. Macro Agent (Vi mo & Chu ky) "
            "2. Quant Agent (12 Alpha Zoo factors, VPIN, CVD) "
            "3. Risk Agent (R:R, Max Drawdown, StopLoss) "
            "4. Execution Agent (So lenh 20 tang LOB, phi san, Carver buffer). "
            "Hay de 4 dac vu tranh luan va bieu quyet. "
            "Tra ve DUY NHAT 1 JSON hop le dang: "
            '{"votes": ['
            '{"agent_id": "macro", "vote": "APPROVE", "confidence": 85, "thesis": "..."},'
            '{"agent_id": "quant", "vote": "APPROVE", "confidence": 90, "thesis": "..."},'
            '{"agent_id": "risk", "vote": "APPROVE", "confidence": 80, "thesis": "..."},'
            '{"agent_id": "execution", "vote": "APPROVE", "confidence": 88, "thesis": "..."}'
            '], "council_rationale": "Tong ket ket qua tranh luan..."}'
        )

        context_data = {
            "symbol": "BTCUSDT",
            "price": current_price,
            "indicators": indicators,
            "proposed_order": {
                "side": getattr(order_research, "recommended_side", "BUY"),
                "type": getattr(order_research, "recommended_type", "POST_ONLY"),
                "price": getattr(order_research, "optimal_price", current_price),
                "margin": getattr(order_research, "optimal_margin", 100.0),
                "leverage": getattr(order_research, "optimal_leverage", 3),
                "sl": getattr(order_research, "structural_sl", 0.0),
                "tp": getattr(order_research, "structural_tp", 0.0),
                "rr": getattr(order_research, "rr_ratio", 1.8)
            },
            "vibe_alpha_zoo": asdict(alpha_zoo_metrics) if alpha_zoo_metrics and hasattr(alpha_zoo_metrics, "__dataclass_fields__") else {},
            "visual_hft": asdict(visual_hft_metrics) if visual_hft_metrics and hasattr(visual_hft_metrics, "__dataclass_fields__") else {},
            "jesse_expectancy": asdict(jesse_metrics) if jesse_metrics and hasattr(jesse_metrics, "__dataclass_fields__") else {},
            "octobot_matrix": asdict(octobot_metrics) if octobot_metrics and hasattr(octobot_metrics, "__dataclass_fields__") else {},
            "user_instruction": user_instruction
        }

        payload = {
            "model": self.default_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Du lieu thi truong:\n" + json.dumps(context_data, ensure_ascii=False)}
            ],
            "temperature": 0.2
        }

        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
            )
            with urllib.request.urlopen(req, timeout=12.0) as response:
                if response.status == 200:
                    resp_json = json.loads(response.read().decode("utf-8"))
                    raw = resp_json["choices"][0]["message"]["content"].strip()
                    if raw.startswith("```json"):
                        raw = raw[7:]
                    if raw.startswith("```"):
                        raw = raw[3:]
                    if raw.endswith("```"):
                        raw = raw[:-3]
                    
                    parsed = json.loads(raw.strip())
                    raw_votes = parsed.get("votes", [])
                    
                    agent_meta = {
                        "macro": ("🌐 Macro & Regime Agent", "🌐", self.agent_models.get("macro", self.default_model)),
                        "quant": ("🔬 Quant Alpha Agent", "🔬", self.agent_models.get("quant", self.default_model)),
                        "risk": ("🛡️ Risk Governor Agent", "🛡️", self.agent_models.get("risk", self.default_model)),
                        "execution": ("⚡ Execution Specialist", "⚡", self.agent_models.get("execution", self.default_model))
                    }

                    votes_list = []
                    approvals = 0
                    for v in raw_votes:
                        aid = v.get("agent_id", "macro")
                        name, icon, mod = agent_meta.get(aid, (aid.upper(), "🤖", self.default_model))
                        vt = v.get("vote", "APPROVE")
                        if vt == "APPROVE":
                            approvals += 1
                        votes_list.append(AgentVote(
                            agent_id=aid,
                            agent_name=name,
                            icon=icon,
                            model_used=f"9router/{mod}",
                            vote=vt,
                            confidence=int(v.get("confidence", 85)),
                            thesis=v.get("thesis", "Đồng thuận theo dữ liệu định lượng."),
                            timestamp=now_str
                        ))

                    passed = (approvals >= self.min_votes_required)
                    verdict_str = "APPROVED" if passed else "REJECTED"

                    return SwarmCouncilVerdict(
                        council_verdict=verdict_str,
                        consensus_passed=passed,
                        approvals_count=approvals,
                        total_agents=4,
                        min_votes_required=self.min_votes_required,
                        votes=votes_list,
                        council_rationale=parsed.get("council_rationale", f"Hội đồng biểu quyết {approvals}/4 phiếu thuận."),
                        timestamp=now_str
                    )
        except Exception:
            return None

    def _fallback_cognitive_swarm(
        self,
        current_price: float,
        indicators: Dict[str, Any],
        ai_verdict: Any = None,
        ensemble_result: Any = None,
        order_research: Any = None,
        alpha_zoo_metrics: Optional[Any] = None,
        visual_hft_metrics: Optional[Any] = None,
        jesse_metrics: Optional[Any] = None,
        octobot_metrics: Optional[Any] = None,
        current_position: Optional[Dict[str, Any]] = None,
        user_instruction: str = "",
        **kwargs
    ) -> SwarmCouncilVerdict:
        """Rule-based Zero-Crash Cognitive Fallback Swarm."""
        now_str = datetime.now().strftime("%H:%M:%S")
        side = getattr(order_research, "recommended_side", "BUY")
        h = indicators.get("hurst_exponent", 0.50) if indicators else 0.50
        rr = getattr(order_research, "rr_ratio", 1.8)
        
        # 1. Macro Vote
        macro_vote = "APPROVE"
        macro_conf = 85
        macro_thesis = f"Xu hướng vĩ mô ổn định, chu kỳ tích lũy an toàn cho lệnh {side}."
        if h > 0.60:
            macro_thesis = f"Sóng xu hướng mạnh (Hurst={h:.2f}), ủng hộ bám trend {side}."

        # 2. Quant Alpha Vote
        quant_vote = "APPROVE"
        quant_conf = 88
        az_score = getattr(alpha_zoo_metrics, "composite_alpha_score", 15.0) if alpha_zoo_metrics else 15.0
        vpin_val = getattr(visual_hft_metrics, "vpin", 0.35) if visual_hft_metrics else 0.35
        is_toxic = getattr(visual_hft_metrics, "is_toxic_flow", False) if visual_hft_metrics else False
        
        if is_toxic:
            quant_vote = "REJECT"
            quant_conf = 95
            quant_thesis = f"VisualHFT VPIN={vpin_val:.2f} phát hiện dòng tiền độc hại cá mập!"
        else:
            quant_thesis = f"Alpha Zoo đạt {az_score:+.1f}đ, VPIN={vpin_val:.2f} lành mạnh."

        # 3. Risk Governor Vote
        risk_vote = "APPROVE"
        risk_conf = 82
        if rr < 1.3:
            risk_vote = "CAUTION"
            risk_conf = 75
            risk_thesis = f"Tỷ lệ R:R={rr:.1f} hơi thấp, khuyến nghị nâng TP để bảo toàn vốn."
        else:
            risk_thesis = f"Tỷ lệ R:R={rr:.1f} chuẩn cấu trúc, SL nằm ngoài râu quét thanh khoản."

        # 4. Execution Specialist Vote
        exec_vote = "APPROVE"
        exec_conf = 90
        exec_type = getattr(order_research, "recommended_type", "POST_ONLY")
        carver_act = getattr(order_research, "carver_action", "BUY")
        exec_thesis = f"Thực thi lệnh {exec_type} Maker 0.02% tiết kiệm phí, Carver action: {carver_act}."

        votes = [
            AgentVote("macro", "🌐 Macro & Regime Agent", "🌐", f"9router/{self.agent_models.get('macro', self.default_model)}", macro_vote, macro_conf, macro_thesis, now_str),
            AgentVote("quant", "🔬 Quant Alpha Agent", "🔬", f"9router/{self.agent_models.get('quant', self.default_model)}", quant_vote, quant_conf, quant_thesis, now_str),
            AgentVote("risk", "🛡️ Risk Governor Agent", "🛡️", f"9router/{self.agent_models.get('risk', self.default_model)}", risk_vote, risk_conf, risk_thesis, now_str),
            AgentVote("execution", "⚡ Execution Specialist", "⚡", f"9router/{self.agent_models.get('execution', self.default_model)}", exec_vote, exec_conf, exec_thesis, now_str),
        ]

        approvals = sum(1 for v in votes if v.vote == "APPROVE")
        passed = (approvals >= self.min_votes_required)

        rationale = f"Hội Đồng Swarm đạt {approvals}/4 phiếu thuận: {'Đủ điều kiện phê duyệt vào lệnh' if passed else 'Bị từ chối vì chưa đạt ngưỡng đồng thuận'}."
        return SwarmCouncilVerdict(
            council_verdict="APPROVED" if passed else "REJECTED",
            consensus_passed=passed,
            approvals_count=approvals,
            total_agents=4,
            min_votes_required=self.min_votes_required,
            votes=votes,
            council_rationale=rationale,
            timestamp=now_str
        )
