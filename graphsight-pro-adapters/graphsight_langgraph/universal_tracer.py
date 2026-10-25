"""Universal / CrewAI manual run tracer to instrument any custom multi-agent workflow."""
from __future__ import annotations

import time
from typing import Any, Optional
import uuid

from ._text import compute_smart_overlap, tokens as _tokens
from .schema import AgentTrace, Retrieval, RetrievedItem, Span, TraceEdge


class UniversalAgentTracer:
    """Explicit instrumentation tracer for CrewAI, AutoGen, or custom AI agent loops."""

    def __init__(self, framework: str = "crewai") -> None:
        self.framework = framework
        self._t0 = time.perf_counter()
        self._spans: dict[str, Span] = {}
        self._order: list[str] = []
        self._retrievals: list[Retrieval] = []
        self._active_span: Optional[str] = None
        self._first_query: Optional[str] = None

    def _now_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000.0

    def start_span(self, name: str, kind: str = "node", parent_id: Optional[str] = None) -> str:
        span_id = str(uuid.uuid4())
        span = Span(
            id=span_id,
            name=name,
            kind=kind,  # type: ignore[arg-type]
            parent_id=parent_id or self._active_span,
            start_ms=self._now_ms(),
        )
        self._spans[span_id] = span
        self._order.append(span_id)
        self._active_span = span_id
        return span_id

    def end_span(self, span_id: Optional[str] = None, status: str = "ok") -> None:
        sid = span_id or self._active_span
        if sid and sid in self._spans:
            self._spans[sid].end_ms = self._now_ms()
            self._spans[sid].status = status  # type: ignore[assignment]
            if self._active_span == sid:
                self._active_span = self._spans[sid].parent_id

    def record_retrieval(
        self,
        query: str,
        items: list[dict[str, Any]],
        edges: Optional[list[dict[str, Any]]] = None,
        span_id: Optional[str] = None,
    ) -> None:
        if not self._first_query and query:
            self._first_query = query

        sid = span_id or self._active_span or self.start_span("retrieval", kind="retriever")
        retrieved_items = []
        for it in items:
            retrieved_items.append(
                RetrievedItem(
                    id=str(it.get("id") or uuid.uuid4()),
                    label=str(it.get("label") or it.get("title") or "item"),
                    kind=str(it.get("kind") or "document"),
                    score=float(it["score"]) if it.get("score") is not None else None,
                    content=it.get("content") or it.get("text"),
                    source_uri=it.get("source") or it.get("url"),
                    metadata=it.get("metadata", {}),
                )
            )

        trace_edges = []
        for e in edges or []:
            trace_edges.append(
                TraceEdge(
                    source=str(e["source"]),
                    target=str(e["target"]),
                    relation=e.get("relation"),
                    weight=e.get("weight"),
                )
            )

        self._retrievals.append(
            Retrieval(
                span_id=sid,
                query=query,
                arm="graph" if trace_edges else "vector",
                items=retrieved_items,
                edges=trace_edges,
            )
        )

    def finish(self, query: Optional[str] = None, answer: Optional[str] = None) -> AgentTrace:
        spans = [self._spans[sid] for sid in self._order]
        for span in spans:
            if span.end_ms is None:
                span.end_ms = self._now_ms()
                span.status = "ok"

        latency = max((s.end_ms or 0.0) for s in spans) if spans else None
        if answer:
            for r in self._retrievals:
                for it in r.items:
                    it.answer_overlap = compute_smart_overlap(answer, it.content)

        return AgentTrace(
            query=query or self._first_query or "",
            framework=self.framework,
            spans=spans,
            retrievals=self._retrievals,
            answer=answer,
            latency_ms=latency,
        )
