"""Studio mapper: maps an AgentTrace onto the Studio's TraceState schema."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Set

from .schema import AgentTrace

_KIND_TO_ENTITY = {
    "pr": "PR",
    "pull_request": "PR",
    "pullrequest": "PR",
    "service": "Service",
    "person": "Person",
    "user": "Person",
    "author": "Person",
    "document": "Document",
    "doc": "Document",
    "chunk": "Document",
    "commit": "Commit",
    "repo": "Repo",
    "repository": "Repo",
    "ticket": "Ticket",
    "issue": "Ticket",
    "tool": "Tool",
}


def to_tracestate(trace: AgentTrace) -> Dict[str, Any]:
    """Converts an AgentTrace into a Studio-renderable TraceState dictionary."""
    has_usage = trace.answer is not None and any(
        item.answer_overlap is not None for r in trace.retrievals for item in r.items
    )
    used_threshold = 0.15

    nodes: List[Dict[str, Any]] = []
    seen_nodes: Set[str] = set()

    for retrieval in trace.retrievals:
        for item in retrieval.items:
            if item.id in seen_nodes:
                continue
            seen_nodes.add(item.id)

            if has_usage:
                used = (item.answer_overlap or 0.0) >= used_threshold
                subtitle = (
                    f"overlaps the answer ({item.answer_overlap:.2f}) · via {trace.framework}"
                    if used
                    else f"no lexical trace in answer · via {trace.framework}"
                )
            else:
                used = True
                subtitle = f"retrieved via {retrieval.arm} arm"

            nodes.append({
                "id": item.id,
                "label": item.label,
                "type": _KIND_TO_ENTITY.get(str(item.metadata.get("type", "document")).lower(), "Document"),
                "active": used,
                "position": {"x": 0, "y": 0},
                "similarity": item.score,
                "score": item.score,
                "meta": {
                    "subtitle": subtitle,
                    "snippet": (item.content or "")[:600] or None,
                    "answerOverlap": item.answer_overlap,
                },
            })

    steps = [
        {
            "id": r.span_id,
            "index": idx,
            "title": f"Retrieval: {r.query[:40]}",
            "detail": f"{r.arm} arm · {len(r.items)} items",
            "status": "complete",
            "badge": "retriever",
            "arm": r.arm,
        }
        for idx, r in enumerate(trace.retrievals)
    ]

    return {
        "id": trace.trace_id or f"trace_{int(time.time() * 1000)}",
        "query": trace.query,
        "answer": trace.answer,
        "computedAt": trace.timestamp.isoformat(),
        "framework": trace.framework,
        "weights": {"vector": 1.0, "graph": 0.0, "intent": "conceptual"},
        "confidence": {"score": 1.0, "uncertainty": 0.0, "rationale": "Direct Agent Trace"},
        "steps": steps,
        "metrics": {
            "queryTimeSec": 0.0,
            "nodesEvaluated": len(nodes),
        },
        "graph": {"nodes": nodes, "edges": []},
        "context": "\n\n".join(item.content for r in trace.retrievals for item in r.items if item.content)[:8000] or None,
    }
