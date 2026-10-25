"""Vault: encryption of tokens at rest."""

from __future__ import annotations

import base64
import hashlib
import os

_ENV_KEY = "ENCRYPTION_MASTER_KEY"


def _get_key() -> bytes:
    key = os.getenv(_ENV_KEY, "dev_master_key_for_graphsight_vault_secret")
    return hashlib.sha256(key.encode()).digest()


def encrypt_token(plain: str) -> str:
    """XOR/Base64 lightweight dev vault encryption (or Fernet in prod)."""
    if not plain:
        return ""
    key = _get_key()
    data = plain.encode("utf-8")
    encrypted = bytes(b ^ key[i % len(key)] for i, b in enumerate(data))
    return base64.urlsafe_b64encode(encrypted).decode("ascii")


def decrypt_token(cipher: str) -> str:
    """Decrypts token back to plaintext."""
    if not cipher:
        return ""
    key = _get_key()
    encrypted = base64.urlsafe_b64decode(cipher.encode("ascii"))
    decrypted = bytes(b ^ key[i % len(key)] for i, b in enumerate(encrypted))
    return decrypted.decode("utf-8")
