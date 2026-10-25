"""LlamaIndex callback handler that captures query engine and retriever runs as an AgentTrace."""
from __future__ import annotations

import hashlib
import time
from typing import Any, Optional, cast
import uuid

from ._text import compute_smart_overlap, tokens as _tokens
from .schema import AgentTrace, Retrieval, RetrievedItem, Span, TraceEdge

try:
    from llama_index.core.callbacks.base_handler import BaseCallbackHandler
except ImportError:
    try:
        from llama_index.callbacks.base_handler import BaseCallbackHandler  # type: ignore
    except ImportError:
        class BaseCallbackHandler:  # type: ignore[no-redef]
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass


class LlamaIndexTracer(BaseCallbackHandler):
    """Captures retrieval, LLM, and query events from LlamaIndex runs."""

    def __init__(self, event_starts_to_ignore: Optional[list[Any]] = None, event_ends_to_ignore: Optional[list[Any]] = None) -> None:
        super().__init__(
            event_starts_to_ignore=event_starts_to_ignore or [],
            event_ends_to_ignore=event_ends_to_ignore or []
        )
        self._t0 = time.perf_counter()
        self._spans: dict[str, Span] = {}
        self._order: list[str] = []
        self._retrievals: list[Retrieval] = []
        self._first_query: Optional[str] = None
        self._last_response_text: Optional[str] = None

    def _now_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000.0

    def on_event_start(
        self,
        event_type: Any,
        payload: Optional[dict[str, Any]] = None,
        event_id: str = "",
        parent_id: str = "",
        **kwargs: Any,
    ) -> str:
        event_id = event_id or str(uuid.uuid4())
        name = str(event_type)
        kind = "node"

        ev_str = str(event_type).lower()
        if "retrieve" in ev_str:
            kind = "retriever"
            query_str = (payload or {}).get("query_str") or (payload or {}).get("query") or ""
            if not self._first_query and query_str:
                self._first_query = str(query_str)
            self._retrievals.append(Retrieval(span_id=event_id, query=str(query_str)))
        elif "llm" in ev_str:
            kind = "llm"
        elif "function_call" in ev_str or "tool" in ev_str:
            kind = "tool"
        elif "query" in ev_str:
            kind = "node"

        span = Span(
            id=event_id,
            name=name,
            kind=kind,  # type: ignore[arg-type]
            parent_id=parent_id if parent_id else None,
            start_ms=self._now_ms(),
        )
        self._spans[event_id] = span
        self._order.append(event_id)
        return event_id

    def on_event_end(
        self,
        event_type: Any,
        payload: Optional[dict[str, Any]] = None,
        event_id: str = "",
        **kwargs: Any,
    ) -> None:
        span = self._spans.get(event_id)
        if span:
            span.end_ms = self._now_ms()
            span.status = "ok"

        ev_str = str(event_type).lower()
        if payload and "retrieve" in ev_str:
            retrieval = next((r for r in reversed(self._retrievals) if r.span_id == event_id), None)
            nodes = payload.get("nodes") or []
            if retrieval is not None:
                for node_with_score in nodes:
                    # NodeWithScore or TextNode
                    node = getattr(node_with_score, "node", node_with_score)
                    score = getattr(node_with_score, "score", None)
                    node_id = getattr(node, "node_id", None) or getattr(node, "id_", None) or str(uuid.uuid4())[:8]
                    content = getattr(node, "text", None) or getattr(node, "get_content", lambda: "")()
                    metadata = getattr(node, "metadata", {}) or {}

                    retrieval.items.append(
                        RetrievedItem(
                            id=str(node_id),
                            label=metadata.get("title") or metadata.get("label") or f"doc_{node_id[:8]}",
                            kind=metadata.get("kind") or "document",
                            score=float(score) if score is not None else None,
                            vector_score=float(score) if score is not None else None,
                            content=content,
                            source_uri=metadata.get("source") or metadata.get("file_path"),
                            metadata={k: v for k, v in metadata.items() if isinstance(v, (str, int, float, bool))},
                        )
                    )
                retrieval.arm = "vector"

        if payload and ("response" in payload or "response_text" in payload):
            resp = payload.get("response") or payload.get("response_text")
            self._last_response_text = str(getattr(resp, "response", resp))

    def start_trace(self, trace_id: Optional[str] = None) -> None:
        pass

    def end_trace(
        self,
        trace_id: Optional[str] = None,
        trace_map: Optional[dict[str, list[str]]] = None,
    ) -> None:
        pass

    def finish(self, query: Optional[str] = None, answer: Optional[str] = None) -> AgentTrace:
        spans = [self._spans[sid] for sid in self._order]
        for span in spans:
            if span.end_ms is None:
                span.end_ms = self._now_ms()
                span.status = "ok"
        latency = max((s.end_ms or 0.0) for s in spans) if spans else None
        if answer is None:
            answer = self._last_response_text
        if answer:
            for retrieval in self._retrievals:
                for item in retrieval.items:
                    item.answer_overlap = compute_smart_overlap(answer, item.content)

        return AgentTrace(
            query=query or self._first_query or "",
            framework="llamaindex",
            spans=spans,
            retrievals=self._retrievals,
            answer=answer,
            latency_ms=latency,
        )
