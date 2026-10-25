"""Artifact storage: handles artifact URI resolution and local/S3 persistence."""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent
TENANT_DATA_DIR = Path(os.getenv("TENANT_DATA_DIR", _BACKEND_DIR / "tenant_data"))
S3_MOCK_DIR = TENANT_DATA_DIR / "s3_mock"
BUILD_DIR = TENANT_DATA_DIR / "build"
POD_CACHE_DIR = TENANT_DATA_DIR / "pods"

STORAGE_PROVIDER = os.getenv("TRACERAG_STORAGE_PROVIDER", "local").lower()
S3_BUCKET = os.getenv("TRACERAG_S3_BUCKET")
_MOCK_SCHEME = "s3mock://"
_S3_SCHEME = "s3://"


def _ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_dir(org_id: str) -> Path:
    return _ensure(BUILD_DIR / org_id)


def pod_artifact_path(pod_id: str, org_id: str, version: int) -> Path:
    return _ensure(POD_CACHE_DIR / pod_id / org_id) / f"v{version}.sqlite3"


def artifact_key(org_id: str, version: int) -> str:
    return f"artifacts/{org_id}/v{version}.sqlite3"


def put_artifact(local_path: Path | str, key: str) -> str:
    """Store local artifact and return its URI."""
    if STORAGE_PROVIDER == "s3":
        import boto3
        s3 = boto3.client("s3")
        s3.upload_file(str(local_path), S3_BUCKET, key)
        return f"{_S3_SCHEME}{S3_BUCKET}/{key}"
    dest = S3_MOCK_DIR / key
    _ensure(dest.parent)
    shutil.copyfile(local_path, dest)
    return f"{_MOCK_SCHEME}{key}"


def get_artifact(uri: str, dest_path: Path | str) -> Path:
    """Download artifact by URI."""
    dest = Path(dest_path)
    _ensure(dest.parent)

    if uri.startswith(_S3_SCHEME):
        import boto3
        bucket, _, key = uri[len(_S3_SCHEME):].partition("/")
        boto3.client("s3").download_file(bucket, key, str(dest))
        return dest

    if uri.startswith(_MOCK_SCHEME):
        src = S3_MOCK_DIR / uri[len(_MOCK_SCHEME):]
        if not src.exists():
            raise FileNotFoundError(f"Missing artifact: {src}")
        shutil.copyfile(src, dest)
        return dest

    raise ValueError(f"Unknown URI scheme: {uri}")


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
