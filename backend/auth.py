"""Authentication and tenancy helper module for Graphsight/TraceRAG."""

from __future__ import annotations

import logging
from contextvars import ContextVar
from typing import Optional

from fastapi import Header, HTTPException, Request, status

from tracerag import config

logger = logging.getLogger("graphsight.auth")

_current_org: ContextVar[Optional[str]] = ContextVar("current_org", default=None)
_current_user: ContextVar[Optional[str]] = ContextVar("current_user", default=None)


def current_org() -> Optional[str]:
    """Returns the request-scoped org_id from contextvar."""
    return _current_org.get() or config.DEFAULT_TENANT_ORG_ID


def get_current_user(authorization: Optional[str] = Header(None)) -> str:
    """Extracts or simulates the user ID. Falls back cleanly in local/dev mode."""
    if not authorization:
        return "dev-user"
    token = authorization.replace("Bearer ", "").strip()
    return token or "dev-user"


def get_current_tenant_org(
    x_org_id: Optional[str] = Header(None, alias="X-Org-Id"),
    authorization: Optional[str] = Header(None),
) -> str:
    """Resolves the tenant organization identity for multi-tenancy."""
    org = x_org_id or config.DEFAULT_TENANT_ORG_ID
    _current_org.set(org)
    return org
