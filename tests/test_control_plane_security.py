import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from starlette.testclient import TestClient

from data.mexc_api_manager import validate_mexc_url
from data.persistent_storage import PersistentStorageManager
from security.rbac import AuthManager, RateLimiter, Role, StructuredAuditLogger, verify_trusted_origin
from security.secret_provider import (
    EncryptedFileSecretProvider,
    LocalMemorySecretProvider,
    sanitize_settings_for_export,
    validate_settings_for_import,
)
from ui.server import app, state


class ControlPlaneSecurityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="sec-test-")
        self.tmp_dir = Path(self.tmp.name)
        self.client = TestClient(app)
        from security.rbac import get_auth_manager
        self.auth = get_auth_manager()
        self.admin_token = self.auth.create_session("admin_user", Role.ADMIN)
        self.operator_token = self.auth.create_session("op_user", Role.OPERATOR)
        self.viewer_token = self.auth.create_session("view_user", Role.VIEWER)

    def tearDown(self):
        self.tmp.cleanup()

    def test_rbac_hierarchy(self):
        self.assertTrue(Role.has_permission(Role.ADMIN, Role.VIEWER))
        self.assertTrue(Role.has_permission(Role.ADMIN, Role.OPERATOR))
        self.assertTrue(Role.has_permission(Role.ADMIN, Role.ADMIN))
        self.assertTrue(Role.has_permission(Role.SYSTEM, Role.ADMIN))

        self.assertTrue(Role.has_permission(Role.OPERATOR, Role.VIEWER))
        self.assertTrue(Role.has_permission(Role.OPERATOR, Role.OPERATOR))
        self.assertFalse(Role.has_permission(Role.OPERATOR, Role.ADMIN))

        self.assertTrue(Role.has_permission(Role.VIEWER, Role.VIEWER))
        self.assertFalse(Role.has_permission(Role.VIEWER, Role.OPERATOR))
        self.assertFalse(Role.has_permission(Role.VIEWER, Role.ADMIN))

    def test_token_authentication_variants(self):
        # Bearer header
        user = self.auth.authenticate(f"Bearer {self.admin_token}")
        self.assertIsNotNone(user)
        self.assertEqual(user.username, "admin_user")
        self.assertEqual(user.role, Role.ADMIN)

        # Raw token string
        user2 = self.auth.authenticate(self.operator_token)
        self.assertIsNotNone(user2)
        self.assertEqual(user2.role, Role.OPERATOR)

        # Invalid token
        self.assertIsNone(self.auth.authenticate("invalid-fake-token"))
        self.assertIsNone(self.auth.authenticate(""))

    def test_origin_validation(self):
        self.assertTrue(verify_trusted_origin("http://127.0.0.1:8000"))
        self.assertTrue(verify_trusted_origin("http://localhost:8000"))
        self.assertTrue(verify_trusted_origin(None))  # Direct/curl request
        self.assertFalse(verify_trusted_origin("http://malicious-site.com"))
        self.assertFalse(verify_trusted_origin("http://evil.127.0.0.1.nip.io:8000"))

    def test_mexc_url_validation_and_ssrf_protection(self):
        # Valid official MEXC endpoints
        self.assertEqual(validate_mexc_url("https://contract.mexc.com"), "https://contract.mexc.com")
        self.assertEqual(validate_mexc_url("https://contract.mexc.co/api/v1"), "https://contract.mexc.co")
        self.assertEqual(validate_mexc_url("https://api.mexc.com"), "https://api.mexc.com")

        # Invalid schemes or hosts
        with self.assertRaises(ValueError):
            validate_mexc_url("http://contract.mexc.com")  # Non-TLS rejected
        with self.assertRaises(ValueError):
            validate_mexc_url("https://evil-attacker.com")  # SSRF target
        with self.assertRaises(ValueError):
            validate_mexc_url("https://contract.mexc.com.attacker.com")  # Subdomain trick
        with self.assertRaises(ValueError):
            validate_mexc_url("ftp://contract.mexc.com")

    def test_unauthenticated_api_mutations_return_401(self):
        # Without any token
        res = self.client.post("/api/action/manual_order", json={"direction": 1})
        self.assertEqual(res.status_code, 401)

        res2 = self.client.post("/api/settings/save_api_keys", json={"exchange": "binance", "api_key": "k", "api_secret": "s"})
        self.assertEqual(res2.status_code, 401)

        res3 = self.client.post("/api/settings/set_decision_mode", json={"mode": "DETERMINISTIC_ONLY"})
        self.assertEqual(res3.status_code, 401)

    def test_insufficient_role_returns_403(self):
        headers_viewer = {"Authorization": f"Bearer {self.viewer_token}"}
        headers_operator = {"Authorization": f"Bearer {self.operator_token}"}

        # Viewer cannot trigger manual orders (requires OPERATOR)
        res = self.client.post("/api/action/manual_order", json={"direction": 1}, headers=headers_viewer)
        self.assertEqual(res.status_code, 403)

        # Viewer and Operator cannot modify API keys (requires ADMIN)
        res2 = self.client.post("/api/settings/save_api_keys", json={"exchange": "binance", "api_key": "k", "api_secret": "s"}, headers=headers_operator)
        self.assertEqual(res2.status_code, 403)

        # Viewer and Operator cannot change decision mode (requires ADMIN)
        res3 = self.client.post("/api/settings/set_decision_mode", json={"mode": "DETERMINISTIC_ONLY"}, headers=headers_operator)
        self.assertEqual(res3.status_code, 403)

    def test_websocket_unauthenticated_connection_closed_with_4401(self):
        with self.assertRaises(Exception):
            with self.client.websocket_connect("/ws") as ws:
                ws.receive_json()

    def test_secret_isolation_in_sqlite(self):
        db_path = self.tmp_dir / "test_sec.sqlite"
        json_path = self.tmp_dir / "test_sec.json"
        mock_provider = LocalMemorySecretProvider()

        storage = PersistentStorageManager(db_path, json_path, secret_provider=mock_provider)
        storage.save_setting("binance_api_key", "super_secret_binance_key_12345")
        storage.save_setting("binance_api_secret", "ultra_secret_binance_secret_67890")

        # In SQLite, raw secret string MUST NOT appear
        with open(db_path, "rb") as f:
            raw_db_content = f.read().decode("latin1")
            self.assertNotIn("super_secret_binance_key_12345", raw_db_content)
            self.assertNotIn("ultra_secret_binance_secret_67890", raw_db_content)

        # Stored setting in DB is a keyring ref
        raw_val = storage.get_setting("binance_api_key", resolve_secrets=False)
        self.assertTrue(raw_val.startswith("keyring://credentials/"))

        # Resolved setting returns plaintext
        resolved_val = storage.get_setting("binance_api_key", resolve_secrets=True)
        self.assertEqual(resolved_val, "super_secret_binance_key_12345")

    def test_export_sanitization_and_import_rejection(self):
        settings = {
            "symbol": "BTCUSDT",
            "binance_api_key": "raw_secret_key",
            "binance_api_secret": "raw_secret_val",
            "mexc_api_key": "raw_mexc_key",
            "regular_setting": "safe_value",
        }
        sanitized = sanitize_settings_for_export(settings)
        self.assertEqual(sanitized["binance_api_key"], "***")
        self.assertEqual(sanitized["binance_api_secret"], "***")
        self.assertEqual(sanitized["mexc_api_key"], "***")
        self.assertEqual(sanitized["regular_setting"], "safe_value")

        # Import rejecting raw secrets in payload
        dirty_payload = {"settings": {"apiKey": "raw_unencrypted_secret"}}
        ok, msg = validate_settings_for_import(dirty_payload)
        self.assertFalse(ok)
        self.assertIn("forbidden raw secret", msg)

        clean_payload = {"settings": {"symbol": "BTCUSDT", "regular_setting": "val"}}
        ok_clean, _ = validate_settings_for_import(clean_payload)
        self.assertTrue(ok_clean)

    def test_rate_limiter(self):
        limiter = RateLimiter(max_requests=3, window_seconds=1.0)
        client_ip = "192.168.1.100"

        self.assertTrue(limiter.allow_request(client_ip))
        self.assertTrue(limiter.allow_request(client_ip))
        self.assertTrue(limiter.allow_request(client_ip))
        self.assertFalse(limiter.allow_request(client_ip))  # Rate limit exceeded

    def test_structured_audit_logger_redaction(self):
        log_file = self.tmp_dir / "audit.jsonl"
        logger = StructuredAuditLogger(log_path=log_file)
        logger.log_event(
            action="API_KEY_UPDATE",
            actor="admin",
            role=Role.ADMIN,
            resource="credentials",
            status="SUCCESS",
            ip="127.0.0.1",
            details={"api_key": "AKIA1234567890SECRET", "authorization": "Bearer secret-token-xyz"},
        )

        with open(log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 1)
            record = json.loads(lines[0])
            self.assertEqual(record["action"], "API_KEY_UPDATE")
            self.assertEqual(record["details"]["api_key"], "***REDACTED***")
            self.assertEqual(record["details"]["authorization"], "***REDACTED***")


if __name__ == "__main__":
    unittest.main()
