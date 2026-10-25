"""SQLModel database session dependency and schema initialization."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from pathlib import Path

from sqlmodel import Session, SQLModel, create_engine

logger = logging.getLogger("graphsight.database")

_engine = None


def get_engine():
    global _engine
    if _engine is None:
        db_url = os.getenv("APP_DATABASE_URL", "sqlite:///./history.sqlite3")
        connect_args = {"check_same_thread": False} if "sqlite" in db_url else {}
        _engine = create_engine(db_url, echo=False, connect_args=connect_args)
    return _engine


def init_db() -> None:
    """Create history tables if they don't exist."""
    from models import ChatSession, TraceLog, User  # noqa: F401
    SQLModel.metadata.create_all(get_engine())
    logger.info("History database schema ready.")


def get_session() -> Iterator[Session]:
    """FastAPI dependency providing database session."""
    with Session(get_engine()) as session:
        yield session
