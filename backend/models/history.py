"""SQLModel tables for history store: User, ChatSession, and TraceLog."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

from sqlalchemy import Column, JSON
from sqlmodel import Field, Relationship, SQLModel


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return str(uuid4())


class User(SQLModel, table=True):
    id: str = Field(primary_key=True)
    email: str = Field(unique=True, index=True)
    created_at: datetime = Field(default_factory=_utcnow)

    sessions: List["ChatSession"] = Relationship(back_populates="user")


class ChatSession(SQLModel, table=True):
    id: str = Field(default_factory=_new_id, primary_key=True)
    user_id: str = Field(foreign_key="user.id", index=True)
    title: str = Field(default="New Chat")
    created_at: datetime = Field(default_factory=_utcnow)

    user: Optional[User] = Relationship(back_populates="sessions")
    traces: List["TraceLog"] = Relationship(back_populates="session")


class TraceLog(SQLModel, table=True):
    id: str = Field(default_factory=_new_id, primary_key=True)
    session_id: str = Field(foreign_key="chatsession.id", index=True)
    query: str
    execution_plan: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    graph_payload: Union[Dict[str, Any], List[Any]] = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=_utcnow)

    session: Optional[ChatSession] = Relationship(back_populates="traces")
