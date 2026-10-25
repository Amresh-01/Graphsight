"""Intent-based hybrid router: vector recall + multi-hop graph traversal."""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from . import config
from .db import TraceDB
from .recency import age_days, decay_factor

logger = logging.getLogger(__name__)

_EMBED_EXECUTOR = ThreadPoolExecutor(
    max_workers=config.EMBED_WORKERS, thread_name_prefix="embed"
)


def shutdown_embed_executor(wait: bool = True) -> None:
    _EMBED_EXECUTOR.shutdown(wait=wait)


@dataclass
class RoutedNode:
    id: str
    label: str | None
    type: str | None
    score_total: float
    score_vector: float
    score_graph: float
    recency: float = 1.0
    age_days: float | None = None
    documents: list[dict] = field(default_factory=list)


@dataclass
class RouterResponse:
    query: str
    results: list[RoutedNode]
    trace_log: dict


class TraceRouter:
    """Hybrid retriever: balances semantic vector recall and graph traversal."""

    def __init__(self, db: TraceDB, embed_model: str = config.EMBED_MODEL) -> None:
        self.db = db
        self.embed_model = embed_model
        self._embedder = None

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.embed_model)
        return self._embedder

    def embed_query(self, query: str) -> List[float]:
        """Encodes query string into dense vector embedding."""
        embedder = self._get_embedder()
        truncated = query[:config.MAX_QUERY_CHARS]
        vec = embedder.encode(truncated, normalize_embeddings=True)
        return vec.tolist()

    def classify_intent(self, query: str) -> Tuple[float, float, str]:
        """Categorizes query intent to establish vector vs. graph weights."""
        q = query.lower()
        if any(marker in q for marker in config.RELATIONAL_QUERY_MARKERS):
            w = config.ROUTER_WEIGHTS_RELATIONAL
            return w["vector"], w["graph"], "relational"

        if any(marker in q for marker in config.SEMANTIC_QUERY_MARKERS):
            w = config.ROUTER_WEIGHTS_CONCEPTUAL
            return w["vector"], w["graph"], "semantic"

        # Default balanced/semantic
        w = config.ROUTER_WEIGHTS_CONCEPTUAL
        return w["vector"], w["graph"], "semantic"

    def route(self, query: str, top_k: Optional[int] = None) -> RouterResponse:
        """Executes full hybrid routing: vector search + BFS graph expansion + intent fusion."""
        start_t = time.time()
        w_vec, w_graph, intent = self.classify_intent(query)
        meta: Dict[str, Dict[str, Any]] = {}

        # 1. Vector Search
        query_emb = self.embed_query(query)
        v_hits = self.db.vector_search(query_emb, k=config.TOP_K_VECTOR)
        s_vec: Dict[str, float] = {}
        for h in v_hits:
            nid = h["id"]
            s_vec[nid] = h["similarity"]
            meta[nid] = {"label": h["label"], "type": h["type"], "ts": h["ts"]}

        # 2. Graph Expansion from Seeds
        seed_candidates = sorted(v_hits, key=lambda x: x["similarity"], reverse=True)
        seeds = [
            h["id"] for h in seed_candidates[:config.GRAPH_SEED_TOP_N]
            if h["similarity"] >= config.GRAPH_SEED_MIN_SIM
        ]

        s_graph: Dict[str, float] = {seed: 1.0 for seed in seeds}
        frontier = list(seeds)
        hops_log = []

        for _ in range(config.GRAPH_MAX_HOPS):
            if not frontier:
                break
            expanded = self.db.expand_frontier(
                frontier,
                limit_per_node=config.GRAPH_NEIGHBOR_K,
                max_degree=config.MAX_DEGREE
            )
            next_frontier = []
            for src_id, neighbors in expanded.items():
                parent_score = s_graph.get(src_id, 1.0)
                for nb in neighbors:
                    target_id = nb["id"]
                    edge_weight = nb["confidence"]
                    path_score = parent_score * edge_weight

                    hops_log.append({
                        "from": src_id,
                        "to": target_id,
                        "relation": nb.get("relation"),
                        "weight": edge_weight
                    })

                    meta.setdefault(target_id, {
                        "label": nb["label"],
                        "type": nb["type"],
                        "ts": nb.get("ts", 0)
                    })

                    if path_score > s_graph.get(target_id, 0.0):
                        s_graph[target_id] = path_score
                        next_frontier.append(target_id)

            frontier = next_frontier

        # 3. Fusion & Recency Decay
        all_ids = set(s_vec.keys()) | set(s_graph.keys())
        now = time.time()
        candidates: List[RoutedNode] = []

        for nid in all_ids:
            score_v = s_vec.get(nid, 0.0)
            score_g = s_graph.get(nid, 0.0)
            combined = (w_vec * score_v) + (w_graph * score_g)

            node_meta = meta.get(nid, {})
            node_ts = node_meta.get("ts")
            node_type = node_meta.get("type")

            r_decay = decay_factor(node_ts, node_type=node_type, now=now)
            final_score = combined * r_decay

            candidates.append(RoutedNode(
                id=nid,
                label=node_meta.get("label"),
                type=node_type,
                score_total=round(final_score, 4),
                score_vector=round(score_v, 4),
                score_graph=round(score_g, 4),
                recency=round(r_decay, 3),
                age_days=round(age_days(node_ts, now) or 0.0, 1) if node_ts else None,
            ))

        candidates.sort(key=lambda x: x.score_total, reverse=True)
        final_results = candidates[:(top_k or config.TOP_K_VECTOR)]

        # 4. Attach Documents
        docs_map = self.db.documents_for_entities([r.id for r in final_results])
        for r in final_results:
            r.documents = docs_map.get(r.id, [])

        duration = time.time() - start_t
        trace_log = {
            "query": query,
            "intent": intent,
            "weights": {"vector": w_vec, "graph": w_graph},
            "seeds": seeds,
            "hops": hops_log,
            "evaluated_nodes": len(all_ids),
            "latency_ms": round(duration * 1000, 2),
        }

        return RouterResponse(query=query, results=final_results, trace_log=trace_log)
