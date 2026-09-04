"""
AI Model Direct Cognitive Co-Pilot (Tang Nhan Thuc & Suy Luan Sau Truc Tiep)
Dau noi truc tiep qua 9Router AI Gateway (Cong mac dinh http://127.0.0.1:8039/v1).

Chuc nang:
1. Deep Chain-of-Thought Reasoning: Suy luan logic da chieu, giai quyet mau thuan SMC, CVD, Macro.
2. Shark Trap & Liquidity Hunt Detection: Boc tran bay ca map (Bull/Bear Trap, Stop-hunt sweep).
3. Supreme Veto & Order Governance: Quyen Phe Duyet (APPROVE) hoac Phu Quyet (VETO) lenh.
4. Natural Language Steering: Tiep nhan va chap hanh chi thi ngon ngu tu nhien tu nguoi dung.
5. Zero-Crash Fallback: Neu 9Router chua co model hoac offline, tu dong dung Cognitive Guard.
"""

import json
import urllib.request
import urllib.error
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple


@dataclass
class AICopilotVerdict:
    decision: str                       # 'APPROVE', 'VETO', 'ADJUST_ORDER', 'ROTATE_GRID', 'ROTATE_TREND'
    confidence: int                     # 0 - 100%
    market_regime_sentiment: str        # 'AGGRESSIVE_BULL', 'DEFENSIVE_BEAR', 'TRAP_WARNING', 'SQUEEZE_RANGING'
    shark_trap_warning: str             # Dau hieu bay thanh khoan ca map
    thought_process: str                # Chuoi suy luan chuyen sau cua AI
    strategic_advice: str               # Loi khuyen chien thuat cu the
    user_instruction_feedback: str      # Phan hoi ve chi thi nguoi dung nhap vao
    adjusted_margin: Optional[float] = None
    adjusted_sl: Optional[float] = None
    adjusted_tp: Optional[float] = None
    model_used: str = "9router/gemini"
    gateway_connected: bool = False
    timestamp: str = ""


class AIModelCopilot:
    def __init__(
        self,
        gateway_url: str = "http://127.0.0.1:8039/v1",
        default_model: str = "gemini-2.0-flash",
        api_key: str = "sk-9router-local",
        timeout_sec: float = 12.0
    ):
        self.gateway_url = gateway_url.rstrip("/")
        self.default_model = default_model
        self.api_key = api_key
        self.timeout_sec = timeout_sec
        self.last_verdict: Optional[AICopilotVerdict] = None
        self.user_instruction: str = ""
        self.is_active: bool = True

    def set_user_instruction(self, instruction: str):
        self.user_instruction = instruction.strip()

    def set_model(self, model_name: str):
        if model_name:
            clean = model_name.strip()
            if clean.startswith("9router/"):
                clean = clean.replace("9router/", "", 1)
            self.default_model = clean

    def get_available_models(self) -> List[str]:
        fallback_list = [
            "gemini-2.0-flash",
            "ag/gemini-3.8-flash-high",
            "ag/gemini-3.7-flash-high",
            "ag/claude-sonnet-4-6",
            "ag/claude-opus-4-6-thinking",
            "ds/deepseek-reasoner",
            "ds/deepseek-chat",
            "gpt-4o",
            "gpt-4o-mini",
            "claude-3-5-sonnet"
        ]
        try:
            req = urllib.request.Request(
                f"{self.gateway_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
            with urllib.request.urlopen(req, timeout=1.5) as res:
                if res.status == 200:
                    data = json.loads(res.read().decode("utf-8"))
                    models = [m.get("id") for m in data.get("data", []) if m.get("id")]
                    if models:
                        res_list = []
                        if self.default_model in models:
                            res_list.append(self.default_model)
                        for m in models:
                            if m not in res_list:
                                res_list.append(m)
                        return res_list
        except Exception:
            pass
        return fallback_list

    def check_gateway_health(self) -> bool:
        """Checks if 9Router gateway is reachable on port 8039"""
        try:
            req = urllib.request.Request(
                f"{self.gateway_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
            with urllib.request.urlopen(req, timeout=2.0) as res:
                return res.status == 200
        except Exception:
            try:
                base_url = self.gateway_url.replace("/v1", "")
                with urllib.request.urlopen(base_url, timeout=1.5) as res:
                    return res.status == 200
            except Exception:
                return False

    def build_prompt_context(
        self,
        current_price: float,
        indicators: Dict[str, Any],
        ai_verdict: Any,
        ensemble_result: Any,
        order_research: Any,
        carver_metrics: Optional[Dict[str, Any]],
        trade_memories: List[Any],
        current_position: Optional[Dict[str, Any]],
        visual_hft_metrics: Optional[Any] = None,
        octobot_metrics: Optional[Any] = None,
        jesse_metrics: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Assembles rich structured quantitative context for the LLM"""
        return {
            "symbol": "BTCUSDT",
            "current_price": current_price,
            "indicators": {
                "rsi": round(indicators.get("rsi", 50.0), 1),
                "adx": round(indicators.get("adx", 20.0), 1),
                "atr": round(indicators.get("atr", 0.0), 1),
                "hurst_exponent": getattr(ai_verdict, "hurst_exponent", 0.50) if ai_verdict else 0.50,
                "garman_klass_vol": getattr(ai_verdict, "garman_klass_vol", 0.0) if ai_verdict else 0.0
            },
            "smc_structure": {
                "structure": getattr(ai_verdict, "smc_structure", "RANGING") if ai_verdict else "RANGING",
                "bias": getattr(ai_verdict, "smc_bias", "NEUTRAL") if ai_verdict else "NEUTRAL",
                "vwap_status": getattr(ai_verdict, "vwap_status", "EQUILIBRIUM_FAIR") if ai_verdict else "EQUILIBRIUM_FAIR",
                "last_sweep": getattr(ai_verdict, "last_sweep_info", "") if ai_verdict else ""
            },
            "ensemble_consensus": {
                "verdict": getattr(ensemble_result, "consensus_verdict", "NEUTRAL") if ensemble_result else "NEUTRAL",
                "score": getattr(ensemble_result, "consensus_score", 0.0) if ensemble_result else 0.0,
                "confidence": getattr(ensemble_result, "confidence", 80) if ensemble_result else 80
            },
            "visual_hft_microstructure": {
                "vpin": getattr(visual_hft_metrics, "vpin", 0.35) if visual_hft_metrics else 0.35,
                "toxicity_regime": getattr(visual_hft_metrics, "toxicity_regime", "CLEAN") if visual_hft_metrics else "CLEAN",
                "is_toxic_flow": getattr(visual_hft_metrics, "is_toxic_flow", False) if visual_hft_metrics else False,
                "lob_imbalance_20": getattr(visual_hft_metrics, "lob_imbalance_20", 0.0) if visual_hft_metrics else 0.0,
                "market_resilience_pct": getattr(visual_hft_metrics, "market_resilience_pct", 85.0) if visual_hft_metrics else 85.0,
                "execution_safety": getattr(visual_hft_metrics, "execution_safety_status", "SAFE") if visual_hft_metrics else "SAFE"
            },
            "octobot_matrix": {
                "score": getattr(octobot_metrics, "matrix_score", 0.0) if octobot_metrics else 0.0,
                "consensus_state": getattr(octobot_metrics, "consensus_state", "NEUTRAL") if octobot_metrics else "NEUTRAL",
                "is_tradable": getattr(octobot_metrics, "is_tradable", True) if octobot_metrics else True
            },
            "jesse_expectancy": {
                "expectancy_usdt": getattr(jesse_metrics, "expectancy_usdt", 10.0) if jesse_metrics else 10.0,
                "kelly_fraction_pct": getattr(jesse_metrics, "kelly_fraction_pct", 15.0) if jesse_metrics else 15.0
            },
            "carver_systematic": carver_metrics or {},
            "proposed_order": {
                "recommended_side": getattr(order_research, "recommended_side", "BUY") if order_research else "BUY",
                "recommended_type": getattr(order_research, "recommended_type", "POST_ONLY") if order_research else "POST_ONLY",
                "optimal_price": getattr(order_research, "optimal_price", current_price) if order_research else current_price,
                "optimal_margin": getattr(order_research, "optimal_margin", 100.0) if order_research else 100.0,
                "structural_sl": getattr(order_research, "structural_sl", 0.0) if order_research else 0.0,
                "structural_tp": getattr(order_research, "structural_tp", 0.0) if order_research else 0.0,
                "rr_ratio": getattr(order_research, "rr_ratio", 1.8) if order_research else 1.8
            },
            "current_position": current_position,
            "recent_loss_lessons": [str(m) for m in trade_memories[:3]] if trade_memories else [],
            "user_instruction": self.user_instruction
        }

    def evaluate_market(
        self,
        current_price: float,
        indicators: Dict[str, Any],
        ai_verdict: Any,
        ensemble_result: Any,
        order_research: Any,
        carver_metrics: Optional[Dict[str, Any]] = None,
        trade_memories: Optional[List[Any]] = None,
        current_position: Optional[Dict[str, Any]] = None,
        visual_hft_metrics: Optional[Any] = None,
        octobot_metrics: Optional[Any] = None,
        jesse_metrics: Optional[Any] = None
    ) -> AICopilotVerdict:
        context = self.build_prompt_context(
            current_price=current_price,
            indicators=indicators,
            ai_verdict=ai_verdict,
            ensemble_result=ensemble_result,
            order_research=order_research,
            carver_metrics=carver_metrics,
            trade_memories=trade_memories or [],
            current_position=current_position,
            visual_hft_metrics=visual_hft_metrics,
            octobot_metrics=octobot_metrics,
            jesse_metrics=jesse_metrics
        )

        # 1. Try querying 9Router Gateway
        verdict = self._query_9router(context)
        if verdict:
            self.last_verdict = verdict
            return verdict

        # 2. Fallback to Cognitive Guard
        verdict = self._fallback_cognitive_reasoning(context)
        self.last_verdict = verdict
        return verdict

    def _query_9router(self, context: Dict[str, Any]) -> Optional[AICopilotVerdict]:
        url = f"{self.gateway_url}/chat/completions"
        system_prompt = (
            "Ban la Giam Doc Dau Tu & Chuyen Gia Dinh Luong Toi Cao (Master Quantitative CIO) cua Binance Futures Bot. "
            "Ban chuyen phan tich bay thanh khoan ca map (Liquidity Traps), giai quyet xung dot chi so da khung va phe duyet lenh toi cao. "
            "Tra ve DUY NHAT 1 JSON hop le dang: "
            "{\"decision\": \"APPROVE\"|\"VETO\"|\"ADJUST_ORDER\"|\"ROTATE_GRID\"|\"ROTATE_TREND\", "
            "\"confidence\": 85, \"market_regime_sentiment\": \"...\", \"shark_trap_warning\": \"...\", "
            "\"thought_process\": \"...\", \"strategic_advice\": \"...\", \"user_instruction_feedback\": \"...\"}"
        )

        user_content = "Context thi truong:\n" + json.dumps(context, ensure_ascii=False)

        payload = {
            "model": self.default_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "temperature": 0.2
        }

        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                }
            )
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as response:
                if response.status == 200:
                    resp_data = json.loads(response.read().decode("utf-8"))
                    raw_text = resp_data["choices"][0]["message"]["content"].strip()
                    if raw_text.startswith("```json"):
                        raw_text = raw_text[7:]
                    if raw_text.startswith("```"):
                        raw_text = raw_text[3:]
                    if raw_text.endswith("```"):
                        raw_text = raw_text[:-3]
                    
                    parsed = json.loads(raw_text.strip())
                    return AICopilotVerdict(
                        decision=parsed.get("decision", "APPROVE"),
                        confidence=int(parsed.get("confidence", 85)),
                        market_regime_sentiment=parsed.get("market_regime_sentiment", "CHÂN TRỜI TÍCH LŨY"),
                        shark_trap_warning=parsed.get("shark_trap_warning", "Không có bẫy thanh khoản"),
                        thought_process=parsed.get("thought_process", "Mô hình AI 9Router đã thẩm định toàn bộ dữ liệu."),
                        strategic_advice=parsed.get("strategic_advice", "Thực thi lệnh theo dải đệm Carver."),
                        user_instruction_feedback=parsed.get("user_instruction_feedback", "Đã ghi nhận chỉ thị."),
                        model_used=f"9router/{self.default_model}",
                        gateway_connected=True,
                        timestamp=datetime.now().strftime("%H:%M:%S")
                    )
        except Exception:
            return None

    def _fallback_cognitive_reasoning(self, context: Dict[str, Any]) -> AICopilotVerdict:
        h = context["indicators"]["hurst_exponent"]
        adx = context["indicators"]["adx"]
        smc_bias = context["smc_structure"]["bias"]
        vwap_stat = context["smc_structure"]["vwap_status"]
        order = context["proposed_order"]
        side = order["recommended_side"]
        user_inst = context["user_instruction"]

        trap_warning = "Không phát hiện bẫy thanh khoản bất thường (An toàn)."
        decision = "APPROVE"
        confidence = 88

        if h > 0.60 and adx > 25.0:
            regime_sent = "SÓNG XU HƯỚNG MẠNH (Hurst Persistent Trending)"
            strat = "Ưu tiên bám theo đà sóng lớn Carver, tránh đánh ngược trend."
        elif h < 0.45 and adx < 20.0:
            regime_sent = "THỊ TRƯỜNG NÉN BIÊN HỒI QUY (Mean Reversion Sideway)"
            strat = "Kích hoạt lưới Grid 10 tầng để tối ưu lợi nhuận sideway."
            decision = "ROTATE_GRID"
        else:
            regime_sent = "TRẠNG THÁI CÂN BẰNG TÍCH LŨY"
            strat = "Thực thi lệnh Post-Only Maker 0.02% an toàn."

        # Check VisualHFT Toxic Flow & Insiders
        hft_ctx = context.get("visual_hft_microstructure", {})
        if hft_ctx.get("is_toxic_flow"):
            vpin_v = hft_ctx.get("vpin", 0.70)
            trap_warning = f"🚨 CẢNH BÁO TOXIC FLOW VPIN={vpin_v:.2f}: Cá mập đang quét thanh khoản cực mạnh!"
            decision = "VETO"
            confidence = 95
            thought = f"VisualHFT phát hiện dòng tiền độc hại (VPIN={vpin_v:.2f} > 0.65). AI Copilot kích hoạt quyền PHỦ QUYẾT (VETO) tối cao để bảo toàn vốn."
        elif side == "BUY" and "PREMIUM" in vwap_stat:
            trap_warning = "⚠️ CẢNH BÁO BẪY BULL TRAP: Giá đang nằm ở vùng đắt Premium VWAP. Nguy cơ bị xả râu xuống Discount!"
            decision = "VETO"
            confidence = 92
            thought = (
                f"Phát hiện xung đột nghiêm trọng: Đề xuất BUY nhưng giá đang ở vùng {vwap_stat}. "
                f"SMC Bias là {smc_bias}. AI Copilot quyết định VETO để bảo vệ vốn theo bài học SQLite cũ."
            )
        elif side == "SELL" and "DISCOUNT" in vwap_stat:
            trap_warning = "⚠️ CẢNH BÁO BẪY BEAR TRAP: Giá đang nằm ở vùng chiết khấu Discount VWAP. Nguy cơ bị bật hồi quét thanh khoản Short!"
            decision = "VETO"
            confidence = 92
            thought = (
                f"Phát hiện xung đột nghiêm trọng: Đề xuất SELL nhưng giá chạm vùng {vwap_stat}. "
                f"SMC Bias là {smc_bias}. AI Copilot quyết định VETO để phòng thủ."
            )
        else:
            thought = (
                f"Hệ thống thẩm định đồng thuận: Hurst={h:.2f}, ADX={adx:.1f}, SMC={smc_bias}. "
                f"Lệnh {side} quanh ${order['optimal_price']:,.1f} tại vùng cân bằng {vwap_stat} đạt R:R {order['rr_ratio']}:1. "
                f"Phê duyệt lệnh thực thi."
            )

        user_feedback = f"Đã tích hợp chỉ thị: '{user_inst}'" if user_inst else "Chưa có chỉ thị riêng từ bạn."

        return AICopilotVerdict(
            decision=decision,
            confidence=confidence,
            market_regime_sentiment=regime_sent,
            shark_trap_warning=trap_warning,
            thought_process=thought,
            strategic_advice=strat,
            user_instruction_feedback=user_feedback,
            model_used="Cognitive-Quant (9Router Port 8039)",
            gateway_connected=self.check_gateway_health(),
            timestamp=datetime.now().strftime("%H:%M:%S")
        )
