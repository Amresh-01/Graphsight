"""Model Context Protocol (MCP) server for Graphsight/TraceRAG.

Exposes typed, semantic graph retrieval tools to AI coding agents (Cursor, Claude Desktop):
- trace_impact: Blast-radius and causal dependency trace around an entity
- search_context: Semantic & relational hybrid search across graph memory
- find_entity: Disambiguation and entity lookup with exact and vector matching
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from tracerag import config
from tracerag.db import TraceDB
from tracerag.router import TraceRouter

logger = logging.getLogger("graphsight.mcp")

try:
    from mcp.server.fastmcp import FastMCP
    _MCP_AVAILABLE = True
except Exception:
    FastMCP = None
    _MCP_AVAILABLE = False


def is_available() -> bool:
    return _MCP_AVAILABLE


@dataclass
class Tenant:
    router: TraceRouter
    db: TraceDB
    graph_id: Optional[str] = None


TenantProvider = Callable[[str], Tenant]
_provider: Optional[TenantProvider] = None


def set_tenant_provider(provider: TenantProvider) -> None:
    global _provider
    _provider = provider


def get_current_tenant() -> Tenant:
    if _provider is None:
        raise RuntimeError("MCP tenant provider not configured")
    import auth
    org_id = auth.current_org() or config.DEFAULT_TENANT_ORG_ID
    return _provider(org_id)


def resolve_entity(tenant: Tenant, name: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Resolves an entity name via exact matching or vector similarity."""
    hits = tenant.db.vector_search(tenant.router.embed_query(name), k=limit)
    return [
        {
            "id": h["id"],
            "label": h["label"],
            "type": h["type"],
            "similarity": h["similarity"],
        }
        for h in hits
    ]


def build_app():
    """Builds the FastMCP ASGI application exposing semantic tools."""
    if not _MCP_AVAILABLE:
        raise RuntimeError("MCP SDK not installed")

    mcp = FastMCP("Graphsight Context Intelligence")

    @mcp.tool()
    def find_entity(name: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Lookup entities in the knowledge graph by name or concept."""
        tenant = get_current_tenant()
        return resolve_entity(tenant, name, limit=limit)

    @mcp.tool()
    def search_context(query: str, top_k: int = 5) -> Dict[str, Any]:
        """Perform intent-weighted hybrid search across code entities and documents."""
        tenant = get_current_tenant()
        resp = tenant.router.route(query, top_k=top_k)
        return {
            "query": query,
            "intent": resp.trace_log.get("intent"),
            "hits": [
                {
                    "id": r.id,
                    "label": r.label,
                    "type": r.type,
                    "score": r.score_total,
                    "documents": r.documents,
                }
                for r in resp.results
            ],
        }

    @mcp.tool()
    def trace_impact(entity_name: str, max_hops: int = 2) -> Dict[str, Any]:
        """Compute the causal blast-radius and relational graph neighborhood of an entity."""
        tenant = get_current_tenant()
        candidates = resolve_entity(tenant, entity_name, limit=1)
        if not candidates:
            return {"error": f"Entity not found: {entity_name}"}

        seed_id = candidates[0]["id"]
        expanded = tenant.db.expand_frontier([seed_id], limit_per_node=10, max_degree=20)
        return {
            "entity": candidates[0],
            "neighborhood": expanded.get(seed_id, []),
        }

    return mcp
