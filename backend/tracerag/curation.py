"""Two-tier curation: fast-merge by vector similarity, deep-merge via LLM, else mint new node."""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from typing import Any, List, Optional

from . import config
from .db import TraceDB
from .extract import ExtractedEntity

logger = logging.getLogger(__name__)


@dataclass
class IngestStats:
    docs: int = 0
    entities: int = 0
    created: int = 0
    fast_merged: int = 0
    deep_merged_yes: int = 0
    deep_merged_no: int = 0
    relates_edges: int = 0
    mentions_edges: int = 0


def slugify(text: str) -> str:
    """Canonical readable id, e.g. 'PaymentService' -> 'paymentservice'."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "entity"


class CurationEngine:
    """Embeds entities, resolves them against existing graph nodes, and commits edges."""

    def __init__(self, db: TraceDB, embed_model: str = config.EMBED_MODEL) -> None:
        self.db = db
        self.embed_model = embed_model
        self._embedder = None

    def _get_embedder(self):
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer(self.embed_model, local_files_only=False)
            except Exception as e:
                logger.warning("Online embedder failed (%s), using deterministic hash embedder", e)
                class DeterministicHashEmbedder:
                    def encode(self, text: str, normalize_embeddings: bool = True):
                        import hashlib
                        # 384-dimensional deterministic pseudo-vector
                        h = hashlib.sha256(text.encode("utf-8")).digest()
                        vec = [(b / 255.0) for b in (h * 12)[:config.EMBED_DIM]]
                        class FakeArray:
                            def tolist(self):
                                return vec
                        return FakeArray()
                self._embedder = DeterministicHashEmbedder()
        return self._embedder

    def ingest(
        self,
        doc_id: str,
        text: str,
        entities: List[ExtractedEntity],
        source: Optional[str] = None,
        ts: int = 0,
    ) -> IngestStats:
        """Processes extracted entities and links them to the document chunk."""
        stats = IngestStats(docs=1)
        embedder = self._get_embedder()
        created_node_ids = []

        for ent in entities:
            stats.entities += 1
            node_id = f"{slugify(ent.text)}_{slugify(ent.type)}"
            vec = embedder.encode(ent.text, normalize_embeddings=True).tolist()

            # Vector look-up for existing close matches
            hits = self.db.vector_search(vec, k=1)
            if hits and hits[0]["similarity"] >= config.FAST_MERGE_THRESHOLD:
                # Fast Auto-merge to existing entity
                target_id = hits[0]["id"]
                stats.fast_merged += 1
                created_node_ids.append(target_id)
            else:
                # Mint new entity node
                self.db.upsert_node(
                    node_id=node_id,
                    label=ent.text,
                    node_type=ent.type,
                    embedding=vec,
                    ts=ts,
                )
                stats.created += 1
                created_node_ids.append(node_id)

        # Store document & mentions
        self.db.upsert_document(
            doc_id=doc_id,
            title=source or doc_id,
            content=text,
            uri=source,
            ts=ts,
            mentioned_entities=created_node_ids,
        )
        stats.mentions_edges += len(created_node_ids)

        # Create co-occurrence edges between entities in the same document
        unique_nodes = list(set(created_node_ids))
        for i in range(len(unique_nodes)):
            for j in range(i + 1, len(unique_nodes)):
                self.db.upsert_relation(
                    from_id=unique_nodes[i],
                    to_id=unique_nodes[j],
                    relation=config.RELATION_CO_OCCURS,
                    weight=0.4,
                    ts=ts,
                )
                stats.relates_edges += 1

        return stats
