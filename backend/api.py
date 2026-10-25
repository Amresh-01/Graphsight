"""FastAPI REST API server for Graphsight / TraceRAG.

Exposes:
- POST /api/trace: Hybrid routing trace with vector/graph scores and execution plan
- POST /api/query: Direct query returning ranked entities and grounded prompt context
- POST /api/ingest: Ingest documents and extract entities into the knowledge graph
- POST /api/ingest/github: Ingest GitHub pull requests, issues, and commits
- GET /api/health: Service readiness and health probe
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from tracerag import config
from tracerag.memory import GraphMemory
from tracerag.router import shutdown_embed_executor
from database import init_db
from routers import history

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("graphsight.api")

_state: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize shared GraphMemory & History DB
    mem = GraphMemory(config.DB_PATH)
    _state["mem"] = mem
    try:
        init_db()
        logger.info("History database initialized.")
    except Exception as e:
        logger.warning("History DB initialization failed: %s", e)
    yield
    # Shutdown: Cleanly stop background thread executors
    try:
        shutdown_embed_executor(wait=True)
    except Exception as e:
        logger.warning("Shutdown error: %s", e)
    logger.info("Graphsight API shutdown complete.")


app = FastAPI(
    title="Graphsight / TraceRAG API",
    version="0.1.0",
    description="Context intelligence & GraphRAG retrieval engine for AI agents",
    lifespan=lifespan,
)

app.include_router(history.router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request & Response Contracts
class TraceRequest(BaseModel):
    query: str
    top_k: Optional[int] = None


class IngestTextRequest(BaseModel):
    doc_id: str
    text: str
    source: Optional[str] = None


class GitHubIngestRequest(BaseModel):
    repo: str
    pulls: List[Dict[str, Any]] = []
    issues: List[Dict[str, Any]] = []
    commits: List[Dict[str, Any]] = []


@app.get("/api/health")
async def health() -> Dict[str, str]:
    return {"status": "ok", "service": "graphsight-api"}


@app.post("/api/trace")
async def trace(req: TraceRequest) -> Dict[str, Any]:
    """Executes intent-based hybrid routing and returns the full trace DAG."""
    mem: GraphMemory = _state.get("mem")
    if not mem:
        raise HTTPException(status_code=503, detail="Graph engine uninitialized")

    resp = await asyncio.to_thread(mem.query, req.query, req.top_k)
    payload = {
        "query": resp.query,
        "results": [asdict(r) for r in resp.results],
        "trace_log": resp.trace_log,
    }
    return payload


@app.post("/api/query")
async def query_endpoint(req: TraceRequest) -> Dict[str, Any]:
    """Returns top ranked entities and assembled prompt context."""
    mem: GraphMemory = _state.get("mem")
    if not mem:
        raise HTTPException(status_code=503, detail="Graph engine uninitialized")

    resp = await asyncio.to_thread(mem.query, req.query, req.top_k)
    return {
        "query": req.query,
        "hits": [asdict(r) for r in resp.results],
        "intent": resp.trace_log.get("intent"),
        "latency_ms": resp.trace_log.get("latency_ms"),
    }


@app.post("/api/ingest")
async def ingest_text(req: IngestTextRequest) -> Dict[str, Any]:
    """Ingests a text document, extracts entities, and builds graph edges."""
    mem: GraphMemory = _state.get("mem")
    if not mem:
        raise HTTPException(status_code=503, detail="Graph engine uninitialized")

    stats = await asyncio.to_thread(mem.ingest_text, req.doc_id, req.text, req.source)
    return {
        "status": "success",
        "doc_id": req.doc_id,
        "docs": stats.docs,
        "entities": stats.entities,
        "created": stats.created,
        "fast_merged": stats.fast_merged,
        "relates_edges": stats.relates_edges,
        "mentions_edges": stats.mentions_edges,
    }


@app.post("/api/ingest/github")
async def ingest_github(req: GitHubIngestRequest) -> Dict[str, Any]:
    """Ingests structured GitHub PRs, issues, and commits into causal graph."""
    mem: GraphMemory = _state.get("mem")
    if not mem:
        raise HTTPException(status_code=503, detail="Graph engine uninitialized")

    stats = await asyncio.to_thread(
        mem.ingest_github,
        repo=req.repo,
        pulls=req.pulls,
        issues=req.issues,
        commits=req.commits,
    )
    return {
        "status": "success",
        "repo": req.repo,
        "nodes": stats.nodes,
        "edges": stats.edges,
        "by_relation": stats.by_relation,
    }
