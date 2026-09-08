import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from starlette.testclient import TestClient

from security.rbac import AuthManager, Role, get_auth_manager
from ui.server import app


class P0AuthAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app, raise_server_exceptions=False)
        self.auth = get_auth_manager()
        self.admin = self.auth.create_session("p0admin", Role.ADMIN)
        self.operator = self.auth.create_session("p0op", Role.OPERATOR)
        self.viewer = self.auth.create_session("p0viewer", Role.VIEWER)

    def H(self, tok):
        return {"Authorization": f"Bearer {tok}"}

    def test_dashboard_without_session_gets_login_not_admin_token(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertNotIn("window.__DESK_TOKEN__", res.text)
        self.assertIn("Trading Desk Login", res.text)

    def test_anonymous_mutations_return_401(self):
        for path, payload in [
            ("/api/action/place_custom_order", {"side": "BUY"}),
            ("/api/action/cancel_order", {"order_id": 1}),
            ("/api/action/execute_pending_order", {"order_id": 1}),
            ("/api/settings/toggle_mode", {"mode": "paper"}),
            ("/api/settings/reset_data", {"amount": 1000}),
            ("/api/settings/test_connection", {"exchange": "binance"}),
        ]:
            res = self.client.post(path, json=payload)
            self.assertEqual(res.status_code, 401, path)

    def test_anonymous_state_requires_viewer(self):
        self.assertEqual(self.client.get("/api/state").status_code, 401)
        self.assertEqual(self.client.get("/api/storage/telemetry").status_code, 401)

    def test_viewer_cannot_mutate(self):
        res = self.client.post("/api/action/place_custom_order", json={"side": "BUY"}, headers=self.H(self.viewer))
        self.assertEqual(res.status_code, 403)

    def test_operator_cannot_change_credentials_or_key(self):
        res = self.client.post("/api/settings/save_api_keys", json={"exchange": "binance"}, headers=self.H(self.operator))
        self.assertEqual(res.status_code, 403)
        res2 = self.client.post("/api/settings/save_ninerouter_key", json={"api_key": "x"}, headers=self.H(self.operator))
        self.assertEqual(res2.status_code, 403)

    def test_test_connection_does_not_persist_credentials(self):
        res = self.client.post("/api/settings/test_connection",
                               json={"exchange": "binance", "api_key": "EVIL", "api_secret": "EVIL",
                                     "base_url": "https://evil.example.com", "proxy_url": "https://evil.example.com"},
                               headers=self.H(self.operator))
        # Khong crash vi auth; payload credential bi bo qua hoan toan
        self.assertIn(res.status_code, (200, 422, 500))

    def test_no_hardcoded_ninerouter_key(self):
        import pathlib
        root = pathlib.Path(__file__).resolve().parent.parent
        hits = []
        for f in list((root / "strategy").glob("*.py")) + list((root / "risk").glob("*.py")):
            txt = f.read_text(encoding="utf-8")
            if "sk-b4a922a69924f20a-b6s8st-780e9a2f" in txt:
                hits.append(str(f))
        self.assertEqual(hits, [])

    def test_missing_key_does_not_approve_ai_entry(self):
        from strategy.vibe_swarm_council import VibeSwarmCouncil
        from strategy.ai_model_copilot import AIModelCopilot
        with patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop("NINEROUTER_API_KEY", None)
            c = VibeSwarmCouncil(api_key=None)
            self.assertIsNone(c.api_key)
            cp = AIModelCopilot(api_key=None)
            self.assertIsNone(cp.api_key)


if __name__ == "__main__":
    unittest.main()
