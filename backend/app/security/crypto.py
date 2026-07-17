"""Key management + symmetric encryption for everything under data/.
See project-plan.md §3 — the threat model is a shared computer, theft, or
casual access, not network interception, so OS-keychain-gated key storage
with no daily password prompt is the practical default.
"""
import keyring
from cryptography.fernet import Fernet

SERVICE_NAME = "hearth"
KEY_ACCOUNT = "data_key"


def get_or_create_key() -> bytes:
    key = keyring.get_password(SERVICE_NAME, KEY_ACCOUNT)
    if key is None:
        key = Fernet.generate_key().decode()
        keyring.set_password(SERVICE_NAME, KEY_ACCOUNT, key)
    return key.encode()


def get_or_create_sqlcipher_key_hex() -> str:
    """SQLCipher wants a raw key, not a Fernet token — derive a separate
    32-byte hex key from the same keychain entry point so there's still only
    one secret a user could ever need to back up or rotate."""
    key = keyring.get_password(SERVICE_NAME, "sqlcipher_key")
    if key is None:
        key = Fernet.generate_key().decode()  # 32 url-safe base64 bytes, plenty of entropy
        keyring.set_password(SERVICE_NAME, "sqlcipher_key", key)
    return key.encode().hex()


class Crypto:
    """Lazy-initialized so importing this module never touches the OS
    keychain until encryption is actually needed."""

    def __init__(self):
        self._fernet: Fernet | None = None

    @property
    def fernet(self) -> Fernet:
        if self._fernet is None:
            self._fernet = Fernet(get_or_create_key())
        return self._fernet

    def encrypt(self, text: str) -> bytes:
        return self.fernet.encrypt(text.encode())

    def decrypt(self, token: bytes) -> str:
        return self.fernet.decrypt(token).decode()


_crypto = Crypto()


def encrypt(text: str) -> bytes:
    return _crypto.encrypt(text)


def decrypt(token: bytes) -> str:
    return _crypto.decrypt(token)


def encrypt_bytes(data: bytes) -> bytes:
    """Fernet-encrypt raw bytes (e.g. cached TTS WAV for identical replay)."""
    return _crypto.fernet.encrypt(data)


def decrypt_bytes(token: bytes) -> bytes:
    return _crypto.fernet.decrypt(token)
