"""Build the memory graph straight from structured GitHub payloads (PRs, issues, commits)."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, Optional

from . import config
from .db import TraceDB

logger = logging.getLogger(__name__)

_CLOSES_RE = re.compile(
    r"(?:fix(?:e[sd])?|close[sd]?|resolve[sd]?)\s+#(\d+)", re.IGNORECASE
)


@dataclass
class GraphStats:
    nodes: int = 0
    edges: int = 0
    by_relation: Dict[str, int] = field(default_factory=dict)

    def count(self, relation: str) -> None:
        self.edges += 1
        self.by_relation[relation] = self.by_relation.get(relation, 0) + 1


def parse_ts(value: Optional[str]) -> int:
    """Converts ISO-8601 ('2026-07-24T09:12:31Z') to Unix seconds."""
    if not value:
        return 0
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    except Exception:
        return 0


class GitHubGraphBuilder:
    """Constructs causal graph edges (AUTHORED, RESOLVES, TOUCHES) directly from GitHub payloads."""

    def __init__(self, db: TraceDB, encode_fn: Callable[[str], list[float]], repo: str) -> None:
        self.db = db
        self.encode = encode_fn
        self.repo = repo

    def build(
        self,
        pulls: Iterable[Dict[str, Any]] = (),
        issues: Iterable[Dict[str, Any]] = (),
        commits: Iterable[Dict[str, Any]] = (),
    ) -> GraphStats:
        stats = GraphStats()

        # Ingest Pull Requests
        for pr in pulls:
            pr_num = pr.get("number")
            pr_id = f"pr_{pr_num}"
            title = pr.get("title", "")
            body = pr.get("body", "") or ""
            author = (pr.get("user") or {}).get("login", "unknown")
            ts = parse_ts(pr.get("created_at") or pr.get("merged_at"))

            # Embed and upsert PR node
            vec = self.encode(f"PR #{pr_num}: {title} {body[:200]}")
            self.db.upsert_node(
                node_id=pr_id,
                label=f"PR #{pr_num}: {title}",
                node_type="PR",
                embedding=vec,
                ts=ts,
            )
            stats.nodes += 1

            # Author relationship
            author_id = f"user_{author.lower()}"
            self.db.upsert_node(
                node_id=author_id,
                label=author,
                node_type="Person",
                embedding=self.encode(f"Developer {author}"),
                ts=ts,
            )
            stats.nodes += 1
            self.db.upsert_relation(
                from_id=author_id,
                to_id=pr_id,
                relation=config.RELATION_AUTHORED,
                ts=ts,
            )
            stats.count(config.RELATION_AUTHORED)

            # Resolves Issue relationship
            for match in _CLOSES_RE.finditer(body):
                issue_id = f"issue_{match.group(1)}"
                self.db.upsert_relation(
                    from_id=pr_id,
                    to_id=issue_id,
                    relation=config.RELATION_RESOLVES,
                    ts=ts,
                )
                stats.count(config.RELATION_RESOLVES)

        return stats
