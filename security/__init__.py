"""
Security and Control-Plane Hardening Module
"""
from security.secret_provider import (
    SecretProvider,
    KeyringSecretProvider,
    EncryptedFileSecretProvider,
    LocalMemorySecretProvider,
    get_secret_provider,
    set_secret_provider,
    resolve_credential,
    make_credential_ref,
    is_credential_ref,
    sanitize_settings_for_export,
    validate_settings_for_import,
    is_secret_key,
)
from security.rbac import (
    Role,
    UserSession,
    AuthManager,
    get_auth_manager,
    StructuredAuditLogger,
    get_audit_logger,
    RateLimiter,
    get_rate_limiter,
    require_role,
    verify_trusted_origin,
)

__all__ = [
    "SecretProvider",
    "KeyringSecretProvider",
    "EncryptedFileSecretProvider",
    "LocalMemorySecretProvider",
    "get_secret_provider",
    "set_secret_provider",
    "resolve_credential",
    "make_credential_ref",
    "is_credential_ref",
    "sanitize_settings_for_export",
    "validate_settings_for_import",
    "is_secret_key",
    "Role",
    "UserSession",
    "AuthManager",
    "get_auth_manager",
    "StructuredAuditLogger",
    "get_audit_logger",
    "RateLimiter",
    "get_rate_limiter",
    "require_role",
    "verify_trusted_origin",
]
