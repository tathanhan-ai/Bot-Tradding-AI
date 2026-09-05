import io
import json
import unittest
from unittest.mock import patch

from strategy.vibe_swarm_council import VibeSwarmCouncil


class _Response:
    status = 200

    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class VibeCouncilIndependenceTest(unittest.TestCase):
    def test_each_agent_uses_its_own_model_and_only_approve_counts(self):
        council = VibeSwarmCouncil(min_votes_required=3)
        council.update_config(macro_model="macro-model", quant_model="quant-model", risk_model="risk-model", exec_model="exec-model")
        payloads = []

        def fake_urlopen(request, timeout):
            payload = json.loads(request.data.decode("utf-8"))
            payloads.append(payload)
            vote = "ABSTAIN" if payload["model"] == "exec-model" else "APPROVE"
            content = json.dumps({"vote": vote, "confidence": 80, "thesis": payload["model"]})
            return _Response({"choices": [{"message": {"content": content}}]})

        with patch("strategy.vibe_swarm_council.urllib.request.urlopen", fake_urlopen):
            verdict = council.evaluate_council(
                current_price=60_000,
                indicators={"rsi": 55, "adx": 25},
                ai_verdict=None,
                ensemble_result=None,
                order_research=None,
            )

        self.assertTrue(verdict.available)
        self.assertTrue(verdict.approved)
        self.assertEqual(verdict.approved_votes, 3)
        self.assertEqual({payload["model"] for payload in payloads}, {"macro-model", "quant-model", "risk-model", "exec-model"})
        self.assertEqual({vote.vote for vote in verdict.votes}, {"APPROVE", "ABSTAIN"})
        self.assertTrue(all(len(payload["messages"]) == 2 for payload in payloads))

    def test_router_failure_returns_unavailable_without_synthetic_votes(self):
        council = VibeSwarmCouncil(min_votes_required=3)

        with patch("strategy.vibe_swarm_council.urllib.request.urlopen", side_effect=OSError("router down")):
            verdict = council.evaluate_council(60_000, {}, None, None, None)

        self.assertFalse(verdict.available)
        self.assertFalse(verdict.approved)
        self.assertEqual(verdict.council_verdict, "LLM_UNAVAILABLE")
        self.assertEqual([vote.vote for vote in verdict.votes], ["ABSTAIN"] * 4)


if __name__ == "__main__":
    unittest.main()
