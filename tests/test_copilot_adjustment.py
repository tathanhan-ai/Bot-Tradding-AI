import json
import unittest
from unittest.mock import patch

from strategy.ai_model_copilot import AIModelCopilot


class _Response:
    status = 200

    def read(self):
        return json.dumps({"choices": [{"message": {"content": json.dumps({
            "decision": "ADJUST_ORDER", "confidence": 72,
            "adjusted_margin": 40, "adjusted_sl": 59900, "adjusted_tp": 61500,
        })}}]}).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class CopilotAdjustmentTest(unittest.TestCase):
    def test_router_adjustment_preserves_explicit_risk_fields(self):
        copilot = AIModelCopilot()
        with patch("strategy.ai_model_copilot.urllib.request.urlopen", return_value=_Response()):
            verdict = copilot._query_9router({})

        self.assertEqual(verdict.decision, "ADJUST_ORDER")
        self.assertEqual(verdict.adjusted_margin, 40.0)
        self.assertEqual(verdict.adjusted_sl, 59900.0)
        self.assertEqual(verdict.adjusted_tp, 61500.0)


if __name__ == "__main__":
    unittest.main()
