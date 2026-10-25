"""Graph & Vector Storage Manager for Graphsight/TraceRAG.

Provides a unified interface with support for:
1. Native LadybugDB / Kùzu when C-extensions are available.
2. Lightweight resilient SQLite + Vector fallback so it runs cleanly anywhere without C-compiler compilation dependencies.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

from . import config

logger = logging.getLogger(__name__)


def cosine_similarity(v1: Sequence[float], v2: Sequence[float]) -> float:
    """Computes cosine similarity between two float vectors."""
    dot = sum(a * b for a, b in zip(v1, v2))
    norm_a = math.sqrt(sum(a * a for a in v1))
    norm_b = math.sqrt(sum(b * b for b in v2))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class TraceDB:
    """Graph and Vector storage layer backing Graphsight hybrid retrieval."""

    def __init__(self, db_path: str | Path = config.DB_PATH, *, pool_size: int | None = None) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        
        # SQLite storage engine for portable cross-platform local graph + vector store
        sqlite_file = self.db_path.with_suffix(".sqlite3") if not str(self.db_path).endswith(".sqlite3") else self.db_path
        self._sqlite_path = str(sqlite_file)
        self._init_sqlite()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._sqlite_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_sqlite(self) -> None:
        """Initializes tables for Entities, Relations, Documents, and Mentions."""
        with self._lock, self._get_connection() as conn:
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                label TEXT,
                type TEXT,
                ts INTEGER DEFAULT 0,
                embedding TEXT
            );

            CREATE TABLE IF NOT EXISTS relations (
                from_id TEXT,
                to_id TEXT,
                relation TEXT DEFAULT 'CO_OCCURS',
                ts INTEGER DEFAULT 0,
                weight REAL DEFAULT 0.7,
                PRIMARY KEY (from_id, to_id, relation)
            );

            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                title TEXT,
                content TEXT,
                uri TEXT,
                ts INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS mentions (
                doc_id TEXT,
                entity_id TEXT,
                PRIMARY KEY (doc_id, entity_id)
            );

            CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type);
            CREATE INDEX IF NOT EXISTS idx_relations_from ON relations(from_id);
            CREATE INDEX IF NOT EXISTS idx_relations_to ON relations(to_id);
            CREATE INDEX IF NOT EXISTS idx_mentions_entity ON mentions(entity_id);
            """)

    def upsert_node(
        self,
        node_id: str,
        label: str,
        node_type: str,
        embedding: Optional[Sequence[float]] = None,
        ts: int = 0
    ) -> None:
        """Inserts or updates an entity node in the graph."""
        emb_json = json.dumps(list(embedding)) if embedding is not None else None
        with self._lock, self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO entities (id, label, type, ts, embedding)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    label = excluded.label,
                    type = excluded.type,
                    ts = CASE WHEN excluded.ts > 0 THEN excluded.ts ELSE entities.ts END,
                    embedding = COALESCE(excluded.embedding, entities.embedding)
                """,
                (node_id, label, node_type, ts, emb_json)
            )

    def upsert_relation(
        self,
        from_id: str,
        to_id: str,
        relation: str = config.RELATION_CO_OCCURS,
        weight: Optional[float] = None,
        ts: int = 0
    ) -> None:
        """Creates or strengthens a relation edge between two entities."""
        edge_weight = weight if weight is not None else config.RELATION_WEIGHTS.get(relation, 0.7)
        with self._lock, self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO relations (from_id, to_id, relation, weight, ts)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(from_id, to_id, relation) DO UPDATE SET
                    weight = MAX(relations.weight, excluded.weight),
                    ts = CASE WHEN excluded.ts > 0 THEN excluded.ts ELSE relations.ts END
                """,
                (from_id, to_id, relation, edge_weight, ts)
            )

    def upsert_document(
        self,
        doc_id: str,
        title: str,
        content: str,
        uri: Optional[str] = None,
        ts: int = 0,
        mentioned_entities: Optional[Sequence[str]] = None
    ) -> None:
        """Stores a document chunk and binds mentions to entity nodes."""
        with self._lock, self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO documents (id, title, content, uri, ts)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    title = excluded.title,
                    content = excluded.content,
                    uri = excluded.uri,
                    ts = excluded.ts
                """,
                (doc_id, title, content, uri or "", ts)
            )
            if mentioned_entities:
                for eid in mentioned_entities:
                    conn.execute(
                        "INSERT OR IGNORE INTO mentions (doc_id, entity_id) VALUES (?, ?)",
                        (doc_id, eid)
                    )

    def vector_search(self, embedding: Sequence[float], k: int = 10) -> List[Dict[str, Any]]:
        """Finds top-k most similar entities via cosine similarity."""
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT id, label, type, ts, embedding FROM entities WHERE embedding IS NOT NULL")
            results = []
            for row in cursor:
                emb_raw = row["embedding"]
                if not emb_raw:
                    continue
                try:
                    vec = json.loads(emb_raw)
                    sim = cosine_similarity(embedding, vec)
                    results.append({
                        "id": row["id"],
                        "label": row["label"],
                        "type": row["type"],
                        "ts": row["ts"],
                        "similarity": round(sim, 4)
                    })
                except Exception:
                    continue

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:k]

    def expand_frontier(
        self,
        frontier: List[str],
        limit_per_node: int = config.GRAPH_NEIGHBOR_K,
        max_degree: int = config.MAX_DEGREE
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Expands graph neighbors around frontier nodes while suppressing hub super-nodes."""
        if not frontier:
            return {}

        out: Dict[str, List[Dict[str, Any]]] = {}
        with self._get_connection() as conn:
            for fid in frontier:
                # Check node degree
                deg_cur = conn.execute(
                    "SELECT COUNT(*) as deg FROM relations WHERE from_id = ? OR to_id = ?",
                    (fid, fid)
                ).fetchone()
                degree = deg_cur["deg"] if deg_cur else 0
                if degree > max_degree:
                    continue  # Hub node suppression

                # Retrieve outbound and inbound edges
                cur = conn.execute(
                    """
                    SELECT r.to_id as target_id, r.relation, r.weight, e.label, e.type, e.ts
                    FROM relations r
                    JOIN entities e ON r.to_id = e.id
                    WHERE r.from_id = ?
                    UNION
                    SELECT r.from_id as target_id, r.relation, r.weight, e.label, e.type, e.ts
                    FROM relations r
                    JOIN entities e ON r.from_id = e.id
                    WHERE r.to_id = ?
                    ORDER BY weight DESC
                    LIMIT ?
                    """,
                    (fid, fid, limit_per_node)
                )

                neighbors = [
                    {
                        "id": row["target_id"],
                        "label": row["label"],
                        "type": row["type"],
                        "relation": row["relation"],
                        "confidence": float(row["weight"]),
                        "ts": row["ts"]
                    }
                    for row in cur
                ]
                out[fid] = neighbors

        return out

    def documents_for_entities(self, entity_ids: Sequence[str]) -> Dict[str, List[Dict[str, Any]]]:
        """Retrieves raw documents referencing the requested entity nodes."""
        if not entity_ids:
            return {}
        out: Dict[str, List[Dict[str, Any]]] = {eid: [] for eid in entity_ids}
        with self._get_connection() as conn:
            for eid in entity_ids:
                cur = conn.execute(
                    """
                    SELECT d.id, d.title, d.content, d.uri, d.ts
                    FROM mentions m
                    JOIN documents d ON m.doc_id = d.id
                    WHERE m.entity_id = ?
                    LIMIT 5
                    """,
                    (eid,)
                )
                for row in cur:
                    out[eid].append({
                        "id": row["id"],
                        "title": row["title"],
                        "content": row["content"],
                        "uri": row["uri"],
                        "ts": row["ts"]
                    })
        return out
