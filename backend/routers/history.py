"""History endpoints for managing chats, sessions, and persisted traces."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import Session, select

from auth import get_current_user
from database import get_engine, get_session
from models import ChatSession, TraceLog, User

logger = logging.getLogger("graphsight.history")

router = APIRouter(prefix="/api", tags=["history"])


class SessionUpsert(BaseModel):
    user_id: Optional[str] = None
    email: Optional[str] = None
    title: str = "New Chat"
    session_id: Optional[str] = None


class SessionRead(BaseModel):
    id: str
    user_id: str
    title: str
    created_at: datetime


class TraceRead(BaseModel):
    id: str
    session_id: str
    query: str
    execution_plan: Dict[str, Any]
    graph_payload: Union[Dict[str, Any], List[Any]]
    created_at: datetime


def _ensure_user(db: Session, user_id: str, email: Optional[str]) -> User:
    user = db.get(User, user_id)
    if user is None:
        placeholder = f"{user_id}@graphsight.local"
        user = User(id=user_id, email=email or placeholder)
        db.add(user)
        db.commit()
        db.refresh(user)
    return user


def session_owner(session_id: str) -> Optional[str]:
    """Look up owner of a session."""
    with Session(get_engine()) as db:
        session = db.get(ChatSession, session_id)
        return session.user_id if session else None


def persist_trace(
    session_id: str,
    query: str,
    execution_plan: Dict[str, Any],
    graph_payload: Union[Dict[str, Any], List[Any]],
) -> str:
    """Stores trace execution record in database."""
    with Session(get_engine()) as db:
        trace = TraceLog(
            session_id=session_id,
            query=query,
            execution_plan=execution_plan,
            graph_payload=graph_payload,
        )
        db.add(trace)
        db.commit()
        db.refresh(trace)
        return trace.id


@router.post("/sessions", response_model=SessionRead)
def upsert_session(
    payload: SessionUpsert,
    user_id: str = Depends(get_current_user),
    db: Session = Depends(get_session),
) -> SessionRead:
    _ensure_user(db, user_id, payload.email)
    if payload.session_id:
        session = db.get(ChatSession, payload.session_id)
        if session is None or session.user_id != user_id:
            raise HTTPException(status_code=404, detail="Session not found")
        session.title = payload.title
    else:
        session = ChatSession(user_id=user_id, title=payload.title)
        db.add(session)

    db.commit()
    db.refresh(session)
    return SessionRead(
        id=session.id,
        user_id=session.user_id,
        title=session.title,
        created_at=session.created_at,
    )


@router.get("/sessions", response_model=List[SessionRead])
def list_sessions(
    user_id: str = Depends(get_current_user),
    db: Session = Depends(get_session),
) -> List[SessionRead]:
    statement = (
        select(ChatSession)
        .where(ChatSession.user_id == user_id)
        .order_by(ChatSession.created_at.desc())
    )
    results = db.exec(statement).all()
    return [
        SessionRead(
            id=s.id,
            user_id=s.user_id,
            title=s.title,
            created_at=s.created_at,
        )
        for s in results
    ]


@router.get("/sessions/{session_id}/traces", response_model=List[TraceRead])
def get_session_traces(
    session_id: str,
    user_id: str = Depends(get_current_user),
    db: Session = Depends(get_session),
) -> List[TraceRead]:
    session = db.get(ChatSession, session_id)
    if session is None or session.user_id != user_id:
        raise HTTPException(status_code=404, detail="Session not found")

    statement = (
        select(TraceLog)
        .where(TraceLog.session_id == session_id)
        .order_by(TraceLog.created_at.asc())
    )
    traces = db.exec(statement).all()
    return [
        TraceRead(
            id=t.id,
            session_id=t.session_id,
            query=t.query,
            execution_plan=t.execution_plan,
            graph_payload=t.graph_payload,
            created_at=t.created_at,
        )
        for t in traces
    ]
