# P0 Acceptance Report — security/p0-auth-secret-ci

Ngay: 2026-09-08. Pham vi: auth dashboard, RBAC, secret 9Router, test_connection, CI, D:/Codex.
Khong doi: indicator, strategy weights, tactical threshold, risk sizing, fee hurdle, veto threshold.

## Test command
```bash
python -m unittest tests.test_p0_auth_acceptance tests.test_control_plane_security tests.test_fail_closed_policy -v
```

## Ket qua
- `tests/test_p0_auth_acceptance.py`: 8/8 PASS
  - dashboard_without_session_gets_login_not_admin_token
  - anonymous_mutations_return_401 (place/cancel/execute/toggle_mode/reset_data/test_connection)
  - anonymous_state_requires_viewer (/api/state, /api/storage/telemetry)
  - viewer_cannot_mutate (403)
  - operator_cannot_change_credentials_or_key (save_api_keys + save_ninerouter_key, 403)
  - test_connection_does_not_persist_credentials (payload credential bi bo qua)
  - no_hardcoded_ninerouter_key (quet strategy/ + risk/)
  - missing_key_does_not_approve_ai_entry (api_key None khi thieu moi nguon)
- `tests/test_control_plane_security.py` + `tests/test_fail_closed_policy.py`: PASS (35 tests chung dot chay lien quan)

## Thay doi chinh
1. Xoa auto-admin `GET /`: chua login tra trang login, khong inject `window.__DESK_TOKEN__`, cookie `desk_session` HttpOnly + SameSite=Strict. Them `POST /auth/login`, `POST /auth/logout`, `GET /health`.
2. Session expiry 12h + `revoke_token` (`security/rbac.py`).
3. RBAC: state/telemetry/vibe GET yeu cau VIEWER; mutation dieu hanh yeu cau OPERATOR; cau hinh nhay cam (credential, model, decision mode, exchange, risk, governor, reset, import) yeu cau ADMIN.
4. Xoa key cung `sk-b4a9...` khoi `strategy/ai_model_copilot.py` + `strategy/vibe_swarm_council.py`. Key lay theo thu tu: tham so > Settings UI (`ninerouter_api_key` qua SecretProvider) > `NINEROUTER_API_KEY`. Them `strategy/ninerouter_key.py` dung chung + endpoint ADMIN `GET /api/settings/ninerouter_key_status` (chi bao co/khong) va `POST /api/settings/save_ninerouter_key`.
5. `test_connection` yeu cau OPERATOR, khong nhan/luu api_key/api_secret/base_url/proxy_url.
6. Sua 4 file test cung `D:/Codex` ve tempfile thuong. CI them ruff + bandit + pip-audit + gitleaks + no-hardcoded-credential gate. `requirements.txt` them `keyring`.

## Con ton tai (ngoai pham vi PR1, can PR tiep)
- Key cu `sk-b4a9...` van nam trong lich su Git (can thu hoi tren 9Router + filter-repo + force-push + quet Gitleaks).
- MEXC proxy trong `save_api_keys` van luu truc tiep khong validate (da validate base_url).
- EncryptedFileSecretProvider van dung XOR tu che (can thay bang keyring/Vault/AES-GCM o PR sau).
- Session token van plaintext trong `data/.desk_tokens.json` (can hash + rotate o PR sau).
- CI chua chay xanh hoan toan tren runner (can day len remote de kiem chung matrix).
