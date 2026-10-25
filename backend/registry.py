"""In-memory tenant graph registry: maps org_id -> loaded Graph handle."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger("graphsight.registry")

HandleLoader = Callable[[str, str], "tuple[Any, Any]"]
_model_source: Any = None


def set_model_source(router: Any) -> None:
    global _model_source
    _model_source = router


def default_tenant_loader(org_id: str, file_path: str) -> "tuple[Any, Any]":
    """Opens a graph database and router for a specific tenant."""
    from tracerag.db import TraceDB
    from tracerag.router import TraceRouter

    db = TraceDB(file_path)
    router = TraceRouter(db)
    if _model_source is not None:
        router._embedder = getattr(_model_source, "_embedder", None)
    return router, db


@dataclass
class LoadedGraph:
    org_id: str
    db: Any
    router: Any
    artifact_id: Optional[str] = None
    path: Optional[str] = None


class TenantDatabaseRegistry:
    """Thread-safe registry mapping organization IDs to loaded graph instances."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tenants: Dict[str, LoadedGraph] = {}
        self._loader: Optional[HandleLoader] = None

    def set_handle_loader(self, loader: HandleLoader) -> None:
        self._loader = loader

    def bind(
        self,
        org_id: str,
        db: Any,
        router: Any,
        artifact_id: Optional[str] = None,
        path: Optional[str] = None,
    ) -> LoadedGraph:
        with self._lock:
            lg = LoadedGraph(
                org_id=org_id,
                db=db,
                router=router,
                artifact_id=artifact_id,
                path=path,
            )
            self._tenants[org_id] = lg
            return lg

    def get_tenant(self, org_id: str) -> Optional[LoadedGraph]:
        with self._lock:
            return self._tenants.get(org_id)

    def swap(self, org_id: str, artifact_id: str, file_path: str) -> LoadedGraph:
        """Hot-swaps a tenant's graph to a new artifact file."""
        router, db = (
            self._loader(org_id, file_path)
            if self._loader
            else (None, None)
        )
        return self.bind(
            org_id,
            db=db,
            router=router,
            artifact_id=artifact_id,
            path=file_path,
        )


REGISTRY = TenantDatabaseRegistry()
