"""Control-plane models for multi-tenant serving pod orchestration."""

from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from sqlmodel import Field, SQLModel


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return str(uuid4())


class LoadStatus:
    READY = "ready"
    PULLING = "pulling"
    FAILED = "failed"


class Organization(SQLModel, table=True):
    org_id: str = Field(primary_key=True)
    name: str
    desired_artifact_id: Optional[str] = None
    created_at: datetime = Field(default_factory=_utcnow)


class PodAssignment(SQLModel, table=True):
    id: str = Field(default_factory=_new_id, primary_key=True)
    pod_id: str = Field(index=True)
    org_id: str = Field(index=True)
    loaded_artifact_id: Optional[str] = None
    load_status: str = Field(default=LoadStatus.PULLING)
    updated_at: datetime = Field(default_factory=_utcnow)
