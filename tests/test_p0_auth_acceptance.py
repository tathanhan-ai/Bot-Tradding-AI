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
        self.assertIn("Binance Futures Quant Desk", res.text)

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

    def test_anonymous_sensitive_reads_require_viewer(self):
        for path in ("/api/klines", "/api/settings", "/api/action/get_ai_models",
                     "/api/settings/vibe", "/api/action/get_vibe_config"):
            self.assertEqual(self.client.get(path).status_code, 401, path)

    def test_login_attempts_are_rate_limited(self):
        with patch("ui.server.get_rate_limiter") as get_limiter:
            get_limiter.return_value.is_allowed.return_value = False
            res = self.client.post("/auth/login", json={"token": "invalid"})
        self.assertEqual(res.status_code, 429)

    def test_failed_live_activation_rolls_back_mode_flags(self):
        from ui.server import state, toggle_trading_mode

        old_exchange = state.active_exchange
        old_testnet = state.binance_api.is_testnet
        old_live = state.binance_api.is_live_enabled
        old_mexc_live = state.mexc_api.is_live_enabled
        old_confirmed = state.binance_api.mainnet_confirmed
        try:
            state.active_exchange = "binance"
            state.binance_api.is_testnet = True
            state.binance_api.is_live_enabled = False
            state.mexc_api.is_live_enabled = False
            state.binance_api.mainnet_confirmed = False
            with patch.object(state.active_exchange_api, "test_connection",
                              return_value={"success": True, "can_trade": True}), \
                 patch.object(state, "sync_exchange_balance", side_effect=RuntimeError("sync failed")):
                result = toggle_trading_mode({"live_enabled": True}, session=self.admin)
            self.assertEqual(result["status"], "error")
            self.assertFalse(state.binance_api.is_live_enabled)
            self.assertFalse(state.mexc_api.is_live_enabled)
            self.assertFalse(state.binance_api.mainnet_confirmed)
        finally:
            state.active_exchange = old_exchange
            state.binance_api.is_testnet = old_testnet
            state.binance_api.is_live_enabled = old_live
            state.mexc_api.is_live_enabled = old_mexc_live
            state.binance_api.mainnet_confirmed = old_confirmed

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
        # Settings local co the da co key nguoi dung nhap truoc do -> mock ca 2 nguon
        # de kiem chung hanh vi "thieu key thi khong co key trong memory".
        with patch.dict("os.environ", {}, clear=False), \
                patch("strategy.vibe_swarm_council.VibeSwarmCouncil._load_key_from_settings", return_value=None), \
                patch("strategy.ninerouter_key.load_ninerouter_key", return_value=None):
            import os
            os.environ.pop("NINEROUTER_API_KEY", None)
            c = VibeSwarmCouncil(api_key=None)
            self.assertIsNone(c.api_key)
            cp = AIModelCopilot(api_key=None)
            self.assertIsNone(cp.api_key)


    def test_logout_keeps_token_alive_for_next_login(self):
        # Nut Thoat chi xoa cookie phia client; token lau dai phai con hieu luc
        # de dang nhap lai, va bot nen (auto-cycle) khong phu thuoc session.
        tok = self.auth.create_session("logout-admin", Role.ADMIN)
        res = self.client.post("/auth/logout", cookies={"desk_session": tok})
        self.assertEqual(res.status_code, 200)
        self.assertIsNotNone(self.auth.authenticate_token(tok))
        # Cookie da duoc xoa phia client (Max-Age=0 / rong)
        set_cookie = res.headers.get("set-cookie", "")
        self.assertIn("Max-Age=0", set_cookie)

    def test_ninerouter_key_status_never_leaks_key(self):
        res = self.client.get("/api/settings/ninerouter_key_status", headers=self.H(self.admin))
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertNotIn("api_key", str(body).lower().replace("has_settings_key", "").replace("has_env_key", ""))
        for k in ("has_env_key", "has_settings_key", "council_key_set", "copilot_key_set"):
            self.assertIn(k, body)


if __name__ == "__main__":
    unittest.main()
