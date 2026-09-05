# Security Threat Model: Control-Plane & Trading Engine Hardening

**Classification**: Confidential / Operational Architecture  
**Target Systems**: Binance Futures & MEXC Quantitative Desk (`Bot-Tradding-AI`)  
**Scope**: Control-Plane API, UI Server, WebSocket Streams, Secret Management, Network Boundary

---

## 1. Executive Summary & Asset Inventory

The trading engine executes real-time risk calculations, order submissions, and position management across exchange APIs (Binance Testnet Futures and MEXC Market Data). Protecting the integrity of the control plane and safeguarding exchange credentials is a Tier-0 requirement.

### Key Protected Assets

1. **Exchange Credentials & Private Keys**:
   - Binance API Key & Secret (HMAC-SHA256).
   - MEXC Access Keys & Session Identifiers.
   - Webhook & 9Router API authentication tokens.
2. **Order Execution & Position Lifecycle**:
   - Order submission, modification, cancellation, and emergency liquidation controls.
   - Position sizing, leverage parameters, and risk governor controls.
3. **Control-Plane Telemetry & Audit Integrity**:
   - Real-time market data snapshots, L2 book states, and AI Council verdicts.
   - Immutable audit logs recording all state mutations and human interventions.

---

## 2. Threat Actors & Threat Scenarios

| Threat Actor | Vector | Impact | Likelihood | Risk Level |
| :--- | :--- | :--- | :--- | :--- |
| **Malicious LAN Actor / Rogue Process** | Unauthenticated REST or WebSocket API access on `0.0.0.0:8000` | Unauthorized order placement, position liquidation, capital draining | High | **P0 (Critical)** |
| **Cross-Origin Browser Script (CSRF/XSS)** | Wildcard CORS (`*`) + no Origin validation on state-modifying POSTs | Browser automatically sends unauthorized trade requests | High | **P0 (Critical)** |
| **Local Disk / Backup Exfiltration** | Plaintext API keys stored in SQLite database (`user_settings`) or export JSON | Extraction of exchange credentials and offline misuse | Medium | **P0 (Critical)** |
| **SSRF / Rogue Gateway Attack** | User-editable exchange URLs (e.g. `mexc_base_url`, `mexc_proxy_url`) | Request hijacking, internal network port scanning, token theft | Medium | **P1 (High)** |
| **Brute-force / DoS on Control Endpoints** | Unrestricted POST rates on order entry or settings update | Exchange API rate-limit bans (IP ban) and process memory exhaustion | Medium | **P1 (High)** |

---

## 3. Security Architecture & Remediation Controls (PR 1)

### 3.1 Role-Based Access Control (RBAC)

Every request to the control-plane REST API and WebSocket gateway is evaluated against four strict privilege tiers:

- **`VIEWER`**: Read-only access to `/api/status`, `/api/trades`, `/api/history`, and telemetry streams.
- **`OPERATOR`**: Operational authority to trigger `/api/pause`, `/api/resume`, manual order placement, order cancellation, and emergency position exits.
- **`ADMIN`**: Full administrative access including credential storage, API key rotation, system reconfiguration, and database backups.
- **`SYSTEM`**: Internal service token used exclusively by background orchestrators and pipeline loops.

### 3.2 WebSocket Security

- WebSocket connections to `/ws` MUST be authenticated before the protocol handshake is accepted.
- Handshake validates `?token=<access_token>` query parameter or `Authorization: Bearer <token>` header.
- Unauthenticated or invalid token requests are immediately closed with WebSocket Close Code `4401` (Unauthorized) without sending internal state payloads.

### 3.3 CORS, Origin Validation & CSRF Protection

- **No Wildcards**: `allow_origins=["*"]` is strictly eliminated.
- **Explicit Origin Allowlist**: Allowed origins are restricted to configured loopback hosts (`http://127.0.0.1:8000`, `http://localhost:8000`) and explicitly configured operator origins.
- **Origin & Referer Verification**: All state-modifying requests (`POST`, `PUT`, `DELETE`) require verification of the `Origin` or `Referer` header against the allowlist.
- **CSRF Defense**: State mutations require an application-generated CSRF token or Bearer authorization header, preventing ambient browser credential abuse.

### 3.4 Rate Limiting

- Sensitive endpoints (authentication, manual orders, settings updates, key management) enforce token bucket rate limiting:
  - Login/Auth: 5 requests / minute per IP.
  - Manual Orders & Emergency Exits: 10 requests / second burst, 30 requests / minute sustained.
  - Settings & Key Updates: 5 requests / minute.

### 3.5 Structured Audit Logging

Every state-modifying event is recorded to a dedicated, append-only structured audit log:
```json
{
  "timestamp": "2026-09-05T04:30:00.123456Z",
  "request_id": "req-9a8f2c3d",
  "actor_id": "operator-01",
  "role": "OPERATOR",
  "action": "EMERGENCY_CLOSE",
  "resource": "position/BTCUSDT",
  "source_ip": "127.0.0.1",
  "parameters": {"symbol": "BTCUSDT", "reason": "Operator manual trigger"},
  "result": "SUCCESS"
}
```
Raw secrets, passwords, Bearer tokens, or HMAC secrets are NEVER included in log records (enforced by automated log redaction filters).

### 3.6 Secret Isolation & `SecretProvider` Protocol

To prevent plaintext credential exposure in database backups or file copies:

1. **Protocol Definition**:
   A uniform `SecretProvider` protocol specifies `get_secret(key_id)`, `set_secret(key_id, secret_value)`, and `delete_secret(key_id)`.
2. **Backends**:
   - **Primary**: OS Keyring (`keyring` library targeting Windows Credential Locker, macOS Keychain, or Linux Secret Service).
   - **Fallback**: Local encrypted file with PBKDF2/AES-GCM.
3. **Database Decoupling**:
   - SQLite `user_settings` table NEVER stores raw `api_key` or `api_secret`.
   - SQLite stores only an opaque reference string: `credential_ref: "keyring://binance_testnet/default"`.
4. **Strict Export / Import Denylist**:
   - Any export payload (`/api/settings/export`) scrubs all fields matching `*key*`, `*secret*`, `*token*`, `*password*`, `*credential*`, `*private*`.
   - Any import payload (`/api/settings/import`) containing raw secret fields is immediately rejected with HTTP 422 Unprocessable Entity.

### 3.7 Target Host Hardening (MEXC URL Restrictions)

- The editable `mexc_base_url` and `mexc_proxy_url` configuration fields in the public UI are locked down.
- Enforced strict hostname allowlist:
  - Allowed Hosts: `{"contract.mexc.com"}`.
  - Allowed Scheme: `https` only.
  - Forbidden: Private IP ranges (`10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16`, `127.0.0.0/8`), port numbers other than default 443, credentials in URL (`user:pass@host`), and URL redirects.

---

## 4. Verification & Compliance Checklist

- [x] RBAC enforcement unit tests (`tests/test_control_plane_security.py`).
- [x] Unauthenticated WebSocket rejection test (code 4401).
- [x] Secret export denylist and import rejection tests.
- [x] MEXC URL host validation and SSRF rejection tests.
- [x] No plaintext secrets logged or persisted in SQLite.
