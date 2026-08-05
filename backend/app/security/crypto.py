"""Key management + symmetric encryption for everything under data/.
See docs/project-plan.md §3 — the threat model is a shared computer, theft, or
casual access, not network interception, so OS-keychain-gated key storage
with no daily password prompt is the practical default.
"""
import secrets

import keyring
from cryptography.fernet import Fernet

SERVICE_NAME = "hearth"
KEY_ACCOUNT = "data_key"
SQLCIPHER_KEY_ACCOUNT = "sqlcipher_key"


def get_or_create_key() -> bytes:
    key = keyring.get_password(SERVICE_NAME, KEY_ACCOUNT)
    if key is None:
        key = Fernet.generate_key().decode()
        keyring.set_password(SERVICE_NAME, KEY_ACCOUNT, key)
    return key.encode()


def is_sqlcipher_raw_key(value: str) -> bool:
    """SQLCipher only treats a `PRAGMA key = "x'...'"` argument as a raw key
    when it is exactly 64 hex characters (32 bytes). Anything else is a
    passphrase, from which it derives the key via PBKDF2."""
    if len(value) != 64:
        return False
    try:
        bytes.fromhex(value)
    except ValueError:
        return False
    return True


def get_or_create_sqlcipher_key_hex() -> str:
    """The hex string handed to `PRAGMA key` — a separate secret from the
    Fernet data key, but stored behind the same keychain entry point so
    there is still only one place a user backs up or rotates.

    New installs get `secrets.token_hex(32)`, which is raw-key mode: 32
    bytes of entropy used directly as the key, no derivation.

    Installs created before that stored a Fernet key (a 44-character base64
    *string*) and passed `key.encode().hex()` — 88 hex characters. That is
    not a valid raw key, so SQLCipher silently treated it as a passphrase
    and ran PBKDF2 over it. Those databases are encrypted under the
    passphrase path and can only ever be opened that way, which is why the
    legacy value is preserved byte-for-byte here instead of being upgraded:
    "fixing" the length in place would change the derivation and make every
    existing profile.db undecryptable. Re-keying an existing database means
    `PRAGMA rekey` on an already-open connection, not a new key at open
    time — a deliberate migration, not something to do implicitly on boot.
    """
    key = keyring.get_password(SERVICE_NAME, SQLCIPHER_KEY_ACCOUNT)
    if key is None:
        raw_key_hex = secrets.token_hex(32)
        keyring.set_password(SERVICE_NAME, SQLCIPHER_KEY_ACCOUNT, raw_key_hex)
        return raw_key_hex
    if is_sqlcipher_raw_key(key):
        return key
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
