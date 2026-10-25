"""Unit tests for storage and security vault modules."""

import os
from pathlib import Path
from security.vault import encrypt_token, decrypt_token
from storage import put_artifact, get_artifact, sha256_file


def test_vault_roundtrip():
    secret = "ghp_super_secret_github_personal_access_token_123"
    encrypted = encrypt_token(secret)
    assert encrypted != secret
    decrypted = decrypt_token(encrypted)
    assert decrypted == secret


def test_storage_roundtrip(tmp_path):
    artifact = tmp_path / "graph.sqlite3"
    artifact.write_text("DUMMY GRAPH ARTIFACT BYTES", encoding="utf-8")

    uri = put_artifact(artifact, "test/v1.sqlite3")
    assert uri.startswith("s3mock://")

    dest = tmp_path / "downloaded.sqlite3"
    get_artifact(uri, dest)
    assert dest.exists()
    assert dest.read_text(encoding="utf-8") == "DUMMY GRAPH ARTIFACT BYTES"
    assert sha256_file(artifact) == sha256_file(dest)
