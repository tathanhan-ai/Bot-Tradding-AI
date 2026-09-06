# -*- coding: utf-8 -*-
import json
import unittest
from unittest.mock import patch

from strategy.vibe_swarm_council import VibeSwarmCouncil


class _FakeResponse:
    def __init__(self, content: str):
        self._body = json.dumps({'choices': [{'message': {'content': content}}]}).encode('utf-8')

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class VibeRetryOnTimeoutTest(unittest.TestCase):
    def test_timeout_on_first_attempt_triggers_immediate_fallback_retry(self):
        council = VibeSwarmCouncil(min_votes_required=3, default_model='ag/gemini-3.8-flash')
        council.update_config(risk_model='ds/deepseek-reasoner')

        call_count = {'risk': 0}

        def fake_urlopen(request, timeout):
            payload = json.loads(request.data.decode('utf-8'))
            model = payload['model']
            if 'risk' in payload['messages'][0]['content']:
                call_count['risk'] += 1
                if call_count['risk'] == 1:
                    raise TimeoutError('9Router connection timed out')
                self.assertEqual(model, 'ag/gemini-3.8-flash')
                return _FakeResponse(json.dumps({'vote': 'APPROVE', 'confidence': 85, 'thesis': 'Recovered risk check'}))
            
            return _FakeResponse(json.dumps({'vote': 'APPROVE', 'confidence': 80, 'thesis': f'Approved by {model}'}))

        with patch('strategy.vibe_swarm_council.urllib.request.urlopen', fake_urlopen):
            verdict = council.evaluate_council(
                current_price=80000.0,
                indicators={'rsi': 50, 'atr': 100},
                ai_verdict=None,
                ensemble_result=None,
                order_research=None,
            )

        self.assertTrue(verdict.available, 'Council should be available because retry succeeded')
        self.assertTrue(verdict.approved)
        self.assertEqual(call_count['risk'], 2, 'Risk agent should have retried once immediately')
        risk_vote = next(v for v in verdict.votes if v.agent_id == 'risk')
        self.assertEqual(risk_vote.vote, 'APPROVE')
        self.assertIn('Retry qua ag/gemini-3.8-flash', risk_vote.thesis)

    def test_fast_recovery_cooldown_when_unavailable(self):
        council = VibeSwarmCouncil(min_votes_required=3)

        with patch('strategy.vibe_swarm_council.urllib.request.urlopen', side_effect=TimeoutError('total timeout')):
            verdict = council.evaluate_council(
                current_price=80000.0,
                indicators={'rsi': 50, 'atr': 100},
                ai_verdict=None,
                ensemble_result=None,
                order_research=None,
            )

        self.assertFalse(verdict.available)
        import time
        time_since_completed = time.time() - council.last_completed_at
        self.assertGreaterEqual(time_since_completed, 25.0, 'Cooldown should be primed for near-instant retry')


if __name__ == '__main__':
    unittest.main()
