"""
Role-Based Access Control (RBAC), Authentication, Origin Verification, and Structured Audit Logging
"""
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum
import hashlib
import hmac
import json
import logging
from pathlib import Path
import secrets
import time
from typing import Any, Callable, Dict, List, Optional, Set
import uuid

from fastapi import Depends, HTTPException, Request, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer


class Role(str, Enum):
    VIEWER = "VIEWER"
    OPERATOR = "OPERATOR"
    ADMIN = "ADMIN"
    SYSTEM = "SYSTEM"

    @classmethod
    def has_permission(cls, user_role: "Role", required_role: "Role") -> bool:
        return ROLE_HIERARCHY.get(user_role, 0) >= ROLE_HIERARCHY.get(required_role, 0)


# Numerical hierarchy for role comparisons
ROLE_HIERARCHY = {
    Role.VIEWER: 1,
    Role.OPERATOR: 2,
    Role.ADMIN: 3,
    Role.SYSTEM: 4,
}


@dataclass
class UserSession:
    user_id: str
    role: Role
    token: str
    created_at: float

    @property
    def username(self) -> str:
        return self.user_id


class AuthManager:
    """Manages active tokens and roles for the trading desk."""

    def __init__(self, secret_provider=None):
        self.secret_provider = secret_provider
        self.sessions: Dict[str, UserSession] = {}
        self._init_default_tokens()

    def _init_default_tokens(self):
        # Generate stable or default tokens for the three human tiers
        # If specified in environment, use those; otherwise generate cryptographically secure tokens.
        import os
        admin_tok = os.environ.get("DESK_ADMIN_TOKEN")
        operator_tok = os.environ.get("DESK_OPERATOR_TOKEN")
        viewer_tok = os.environ.get("DESK_VIEWER_TOKEN")

        if not admin_tok:
            admin_tok = "admin-" + secrets.token_hex(16)
        if not operator_tok:
            operator_tok = "operator-" + secrets.token_hex(16)
        if not viewer_tok:
            viewer_tok = "viewer-" + secrets.token_hex(16)

        self.register_token(admin_tok, "admin-user", Role.ADMIN)
        self.register_token(operator_tok, "operator-user", Role.OPERATOR)
        self.register_token(viewer_tok, "viewer-user", Role.VIEWER)

    def create_session(self, user_id: str, role: Role) -> str:
        token = f"{role.value.lower()}-{secrets.token_hex(16)}"
        self.register_token(token, user_id, role)
        return token

    def register_token(self, token: str, user_id: str, role: Role) -> UserSession:
        session = UserSession(
            user_id=user_id,
            role=role,
            token=token,
            created_at=time.time()
        )
        self.sessions[token] = session
        return session

    def authenticate_token(self, token: Optional[str]) -> Optional[UserSession]:
        if not token:
            return None
        token = token.strip()
        if token.startswith("Bearer "):
            token = token[len("Bearer "):].strip()
        return self.sessions.get(token)

    def authenticate(self, token: Optional[str]) -> Optional[UserSession]:
        return self.authenticate_token(token)


# Global Auth Manager
_AUTH_MANAGER = AuthManager()


def get_auth_manager() -> AuthManager:
    return _AUTH_MANAGER


# ---------------------------------------------------------------------------
# Structured Audit Logger
# ---------------------------------------------------------------------------

class StructuredAuditLogger:
    """Records high-fidelity immutable structured JSON logs of all control-plane actions."""

    def __init__(self, log_path: Optional[Path] = None):
        if log_path is None:
            base_dir = Path(__file__).resolve().parent.parent / "data"
            base_dir.mkdir(parents=True, exist_ok=True)
            log_path = base_dir / "audit_log.jsonl"
        self.log_path = Path(log_path)

    def log(
        self,
        actor_id: str,
        role: str,
        action: str,
        resource: str,
        request_id: str,
        result: str,
        source_ip: str = "127.0.0.1",
        parameters: Optional[Dict[str, Any]] = None
    ):
        # Ensure no raw secrets are logged
        from security.secret_provider import is_secret_key, sanitize_settings_for_export
        safe_params = {}
        for k, v in (parameters or {}).items():
            if is_secret_key(k):
                safe_params[k] = "***REDACTED***"
            elif isinstance(v, dict):
                safe_params[k] = sanitize_settings_for_export(v, mask=True)
            else:
                safe_params[k] = v

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id,
            "actor_id": actor_id,
            "role": str(role),
            "action": action,
            "resource": resource,
            "source_ip": source_ip,
            "parameters": safe_params,
            "details": safe_params,
            "result": result
        }
        line = json.dumps(entry, ensure_ascii=False)
        try:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception as e:
            print(f"[AuditLog Error] Failed to write audit log: {e}")
        # Also print to stdout
        print(f"[AUDIT] {entry['timestamp']} | {actor_id} ({role}) -> {action} [{resource}] => {result}", flush=True)

    def log_event(
        self,
        action: str,
        actor: str,
        role: Any,
        resource: str,
        status: str,
        ip: str = "127.0.0.1",
        details: Optional[Dict[str, Any]] = None,
        request_id: str = "manual",
    ):
        self.log(
            actor_id=actor,
            role=str(role),
            action=action,
            resource=resource,
            request_id=request_id,
            result=status,
            source_ip=ip,
            parameters=details,
        )


_AUDIT_LOGGER = StructuredAuditLogger()


def get_audit_logger() -> StructuredAuditLogger:
    return _AUDIT_LOGGER


# ---------------------------------------------------------------------------
# Fast Rate Limiter (Token Bucket / Sliding Window)
# ---------------------------------------------------------------------------

class RateLimiter:
    """Sliding window in-memory rate limiter per IP address."""

    def __init__(self, max_requests: int = 10, window_seconds: float = 1.0):
        self.default_max_requests = max_requests
        self.default_window_seconds = window_seconds
        self.history: Dict[str, List[float]] = {}

    def is_allowed(self, key: str, max_requests: int, window_seconds: float) -> bool:
        now = time.time()
        timestamps = self.history.setdefault(key, [])
        # Prune older than window
        cutoff = now - window_seconds
        self.history[key] = [ts for ts in timestamps if ts > cutoff]
        if len(self.history[key]) >= max_requests:
            return False
        self.history[key].append(now)
        return True

    def allow_request(self, key: str, max_requests: Optional[int] = None, window_seconds: Optional[float] = None) -> bool:
        max_req = max_requests if max_requests is not None else self.default_max_requests
        win_sec = window_seconds if window_seconds is not None else self.default_window_seconds
        return self.is_allowed(key, max_req, win_sec)


_RATE_LIMITER = RateLimiter()


def get_rate_limiter() -> RateLimiter:
    return _RATE_LIMITER


# ---------------------------------------------------------------------------
# FastAPI Dependencies for RBAC
# ---------------------------------------------------------------------------

security_bearer = HTTPBearer(auto_error=False)


def get_token_from_request(request: Request) -> Optional[str]:
    # 1. Header Authorization: Bearer <token>
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        return auth_header[len("Bearer "):].strip()
    # 2. Header X-Desk-Token
    desk_token = request.headers.get("X-Desk-Token")
    if desk_token:
        return desk_token.strip()
    # 3. Query parameter ?token=...
    query_token = request.query_params.get("token")
    if query_token:
        return query_token.strip()
    # 4. Cookie 'desk_token'
    cookie_token = request.cookies.get("desk_token")
    if cookie_token:
        return cookie_token.strip()
    return None


def require_role(minimum_role: Role):
    """
    FastAPI dependency that enforces authentication and RBAC role hierarchy.
    """
    async def dependency(request: Request) -> UserSession:
        token = get_token_from_request(request)
        if not token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication token required"
            )
        session = get_auth_manager().authenticate_token(token)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired authentication token"
            )

        user_level = ROLE_HIERARCHY.get(session.role, 0)
        req_level = ROLE_HIERARCHY.get(minimum_role, 0)
        if user_level < req_level:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Forbidden: role '{session.role.value}' lacks required privilege '{minimum_role.value}'"
            )
        return session

    return dependency


DEFAULT_ALLOWED_ORIGINS = {
    "http://127.0.0.1:8000",
    "http://localhost:8000",
    "https://127.0.0.1:8000",
    "https://localhost:8000",
}


def verify_trusted_origin(origin_or_request: Any, allowed_origins: Optional[Set[str]] = None) -> bool:
    """
    Validates Origin / Referer header on state-modifying requests.
    Accepts either a Request object or an origin string.
    """
    allowed = allowed_origins if allowed_origins is not None else DEFAULT_ALLOWED_ORIGINS
    from urllib.parse import urlparse

    if hasattr(origin_or_request, "headers"):
        origin = origin_or_request.headers.get("Origin") or origin_or_request.headers.get("Referer")
        if not origin:
            return True
        parsed = urlparse(origin)
        clean_origin = f"{parsed.scheme}://{parsed.netloc}".rstrip("/")
        if clean_origin not in allowed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Untrusted cross-origin request from: {clean_origin}"
            )
        return True
    else:
        if not origin_or_request:
            return True
        parsed = urlparse(str(origin_or_request))
        clean_origin = f"{parsed.scheme}://{parsed.netloc}".rstrip("/")
        return clean_origin in allowed
