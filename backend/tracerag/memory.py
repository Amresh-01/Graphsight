"""GraphMemory — Unified entry point for the embedded memory and hybrid retrieval engine.

Example usage:
    mem = GraphMemory("memory.sqlite3")
    mem.ingest_text("doc1", "PaymentService was refactored in PR #402 by Alice.")
    hits = mem.query("who worked on PaymentService?")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable, Optional

from . import config
from .curation import CurationEngine, IngestStats
from .db import TraceDB
from .extract import EntityExtractor
from .github_graph import GitHubGraphBuilder, GraphStats
from .router import RouterResponse, TraceRouter

logger = logging.getLogger(__name__)


class GraphMemory:
    """High-level facade unifying Graph Storage, Ingestion, and Hybrid Retrieval."""

    def __init__(self, path: str | Path = config.DB_PATH, embed_model: Optional[str] = None):
        self.path = Path(path)
        self.db = TraceDB(self.path)
        self.router = TraceRouter(self.db, embed_model or config.EMBED_MODEL)
        self._extractor = None
        self._curation = None

    def ingest_text(self, doc_id: str, text: str, source: Optional[str] = None) -> IngestStats:
        """Extracts entities from prose, resolves them, and builds co-occurrence edges."""
        if self._extractor is None:
            self._extractor = EntityExtractor()
        if self._curation is None:
            self._curation = CurationEngine(self.db, self.router.embed_model)

        entities = self._extractor.extract(text)
        return self._curation.ingest(doc_id, text, entities, source=source)

    def ingest_github(
        self,
        repo: str,
        pulls: Iterable[dict[str, Any]] = (),
        issues: Iterable[dict[str, Any]] = (),
        commits: Iterable[dict[str, Any]] = (),
    ) -> GraphStats:
        """Builds causal graph edges straight from structured GitHub payloads."""
        builder = GitHubGraphBuilder(self.db, self.router.embed_query, repo)
        return builder.build(pulls=pulls, issues=issues, commits=commits)

    def query(self, question: str, top_k: Optional[int] = None) -> RouterResponse:
        """Queries the memory graph using intent-based hybrid routing."""
        return self.router.route(question, top_k=top_k)

    def close(self) -> None:
        pass

    def __enter__(self) -> "GraphMemory":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
