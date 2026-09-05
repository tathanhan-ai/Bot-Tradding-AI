"""
Secret Isolation & Credential Provider Protocol
Decouples exchange credentials (Binance / MEXC API keys and secrets) from SQLite database storage.
"""
from abc import ABC, abstractmethod
import base64
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
from typing import Any, Dict, Optional, Tuple


SERVICE_NAME = "binance_futures_quant_desk"

# Keys that must NEVER be exported in plaintext or accepted in unvalidated imports
FORBIDDEN_SECRET_PATTERNS = [
    re.compile(r"api_?key", re.IGNORECASE),
    re.compile(r"api_?secret", re.IGNORECASE),
    re.compile(r"secret", re.IGNORECASE),
    re.compile(r"password", re.IGNORECASE),
    re.compile(r"token", re.IGNORECASE),
    re.compile(r"private_?key", re.IGNORECASE),
    re.compile(r"credential", re.IGNORECASE),
    re.compile(r"authorization", re.IGNORECASE),
    re.compile(r"bearer", re.IGNORECASE),
]


class SecretProvider(ABC):
    """Abstract interface for secure secret retrieval and storage."""

    @abstractmethod
    def get_secret(self, key_id: str) -> Optional[str]:
        """Retrieve a secret by its identifier."""
        pass

    @abstractmethod
    def set_secret(self, key_id: str, value: str) -> None:
        """Store a secret under an identifier."""
        pass

    @abstractmethod
    def delete_secret(self, key_id: str) -> None:
        """Remove a secret by its identifier."""
        pass


class KeyringSecretProvider(SecretProvider):
    """Stores secrets in the OS Credential Locker / Keyring via the keyring library."""

    def __init__(self, service_name: str = SERVICE_NAME):
        self.service_name = service_name
        self._keyring = None
        try:
            import keyring
            self._keyring = keyring
        except ImportError:
            self._keyring = None

    @property
    def is_available(self) -> bool:
        if self._keyring is None:
            return False
        try:
            # Check backend is not fail/disabled
            backend = self._keyring.get_keyring()
            name = backend.__class__.__name__
            return "fail" not in name.lower() and "null" not in name.lower()
        except Exception:
            return False

    def get_secret(self, key_id: str) -> Optional[str]:
        if not self.is_available:
            return None
        try:
            return self._keyring.get_password(self.service_name, key_id)
        except Exception as e:
            print(f"[KeyringSecretProvider] Error getting secret {key_id}: {e}")
            return None

    def set_secret(self, key_id: str, value: str) -> None:
        if not self.is_available:
            raise RuntimeError("OS Keyring is not available")
        self._keyring.set_password(self.service_name, key_id, value)

    def delete_secret(self, key_id: str) -> None:
        if not self.is_available:
            return
        try:
            self._keyring.delete_password(self.service_name, key_id)
        except Exception:
            pass


class EncryptedFileSecretProvider(SecretProvider):
    """
    Fallback secret store using PBKDF2-HMAC encryption.
    Stores ciphertext at data/.keystore.enc with restrictive permissions.
    """

    def __init__(self, filepath: Optional[Path] = None, master_key: Optional[str] = None):
        if filepath is None:
            base_dir = Path(__file__).resolve().parent.parent / "data"
            base_dir.mkdir(parents=True, exist_ok=True)
            filepath = base_dir / ".keystore.enc"
        self.filepath = Path(filepath)
        self.master_key = master_key or os.environ.get("DESK_MASTER_KEY") or self._derive_machine_seed()
        self._cache: Dict[str, str] = {}
        self._load()

    def _derive_machine_seed(self) -> str:
        # Stable machine-tied seed
        components = [
            os.environ.get("COMPUTERNAME", "LOCAL"),
            os.environ.get("USERNAME", "DESK"),
            str(Path.home()),
        ]
        return hashlib.sha256("::".join(components).encode("utf-8")).hexdigest()

    def _xor_cipher(self, data: bytes, key: bytes) -> bytes:
        # Deterministic stream cipher with HMAC key expansion
        expanded_key = b""
        counter = 0
        while len(expanded_key) < len(data):
            h = hmac.new(key, f"expand:{counter}".encode("utf-8"), hashlib.sha256).digest()
            expanded_key += h
            counter += 1
        return bytes(a ^ b for a, b in zip(data, expanded_key[:len(data)]))

    def _load(self):
        if not self.filepath.exists():
            self._cache = {}
            return
        try:
            raw = self.filepath.read_bytes()
            if not raw:
                self._cache = {}
                return
            salt = raw[:16]
            mac = raw[16:48]
            ciphertext = raw[48:]

            key = hashlib.pbkdf2_hmac("sha256", self.master_key.encode("utf-8"), salt, 100_000)
            expected_mac = hmac.new(key, salt + ciphertext, hashlib.sha256).digest()
            if not hmac.compare_digest(mac, expected_mac):
                print("[EncryptedFileSecretProvider] Integrity check failed; resetting keystore")
                self._cache = {}
                return

            plaintext = self._xor_cipher(ciphertext, key)
            self._cache = json.loads(plaintext.decode("utf-8"))
        except Exception as e:
            print(f"[EncryptedFileSecretProvider] Error loading keystore: {e}")
            self._cache = {}

    def _save(self):
        try:
            salt = os.urandom(16)
            key = hashlib.pbkdf2_hmac("sha256", self.master_key.encode("utf-8"), salt, 100_000)
            plaintext = json.dumps(self._cache).encode("utf-8")
            ciphertext = self._xor_cipher(plaintext, key)
            mac = hmac.new(key, salt + ciphertext, hashlib.sha256).digest()

            self.filepath.parent.mkdir(parents=True, exist_ok=True)
            self.filepath.write_bytes(salt + mac + ciphertext)
            try:
                os.chmod(self.filepath, 0o600)
            except Exception:
                pass
        except Exception as e:
            print(f"[EncryptedFileSecretProvider] Error saving keystore: {e}")

    def get_secret(self, key_id: str) -> Optional[str]:
        return self._cache.get(key_id)

    def set_secret(self, key_id: str, value: str) -> None:
        self._cache[key_id] = str(value)
        self._save()

    def delete_secret(self, key_id: str) -> None:
        if key_id in self._cache:
            del self._cache[key_id]
            self._save()


class LocalMemorySecretProvider(SecretProvider):
    """In-memory secret provider for isolated unit tests."""

    def __init__(self):
        self._secrets: Dict[str, str] = {}

    def get_secret(self, key_id: str) -> Optional[str]:
        return self._secrets.get(key_id)

    def set_secret(self, key_id: str, value: str) -> None:
        self._secrets[key_id] = str(value)

    def delete_secret(self, key_id: str) -> None:
        self._secrets.pop(key_id, None)


# Default composite provider (Keyring preferred, Encrypted file fallback)
_GLOBAL_PROVIDER: Optional[SecretProvider] = None


def get_secret_provider() -> SecretProvider:
    global _GLOBAL_PROVIDER
    if _GLOBAL_PROVIDER is None:
        keyring_provider = KeyringSecretProvider()
        if keyring_provider.is_available:
            _GLOBAL_PROVIDER = keyring_provider
        else:
            _GLOBAL_PROVIDER = EncryptedFileSecretProvider()
    return _GLOBAL_PROVIDER


def set_secret_provider(provider: SecretProvider) -> None:
    global _GLOBAL_PROVIDER
    _GLOBAL_PROVIDER = provider


def make_credential_ref(service: str, key_id: str) -> str:
    """Returns a standardized reference string: keyring://service/key_id"""
    return f"keyring://{service}/{key_id}"


def is_credential_ref(val: Any) -> bool:
    return isinstance(val, str) and val.startswith("keyring://")


def resolve_credential(val: Any, provider: Optional[SecretProvider] = None) -> str:
    """
    If val is a credential ref (keyring://service/key_id), resolves it from provider.
    Otherwise returns val unchanged (for backwards compatibility).
    """
    if not is_credential_ref(val):
        return str(val or "")
    if provider is None:
        provider = get_secret_provider()
    # Format: keyring://service/key_id
    parts = val[len("keyring://"):].split("/", 1)
    if len(parts) == 2:
        key_id = parts[1]
    else:
        key_id = parts[0]
    secret = provider.get_secret(key_id)
    return secret or ""


# ---------------------------------------------------------------------------
# Export Sanitization & Import Validation Denylist
# ---------------------------------------------------------------------------

def is_secret_key(key: str) -> bool:
    return any(pat.search(key) for pat in FORBIDDEN_SECRET_PATTERNS)


def sanitize_settings_for_export(settings: Dict[str, Any], mask: bool = True) -> Dict[str, Any]:
    """
    Recursively scrubs any field that matches secret keywords or is a raw API key.
    If mask is True, masks secret keys with '***'; otherwise omits them.
    Replaces credential refs with opaque tokens.
    """
    sanitized = {}
    for k, v in settings.items():
        if is_secret_key(k):
            if mask:
                sanitized[k] = "***"
            continue
        if isinstance(v, dict):
            sanitized[k] = sanitize_settings_for_export(v, mask=mask)
        elif is_credential_ref(v):
            sanitized[k] = "[CREDENTIAL_REF_STORED_IN_KEYRING]"
        else:
            sanitized[k] = v
    return sanitized


def validate_settings_for_import(settings: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Scans an imported dictionary. Rejects if it contains raw unencrypted secrets.
    """
    for k, v in settings.items():
        if is_secret_key(k):
            if isinstance(v, str) and len(v.strip()) > 0:
                if not is_credential_ref(v) and v != "[CREDENTIAL_REF_STORED_IN_KEYRING]":
                    return False, f"Import payload contains forbidden raw secret field: '{k}'"
        if isinstance(v, dict):
            ok, msg = validate_settings_for_import(v)
            if not ok:
                return False, msg
    return True, "OK"
