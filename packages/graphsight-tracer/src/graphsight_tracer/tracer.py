"""LangGraph and LangChain callback tracer for Graphsight.

Intercepts node execution boundaries, retriever documents, and LLM responses,
automatically calculating answer overlap scores against the final generation.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, List, Optional
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document

from ._text import compute_answer_overlap, tokenize
from .schema import AgentTrace, RetrievalSpan, RetrievedItem

_SCORE_KEYS = ("score", "relevance_score", "similarity", "_score", "vector_score")
_NOISE_NAMES = {"LangGraph", "RunnableSequence", "RunnableCallable", "ChannelWrite", "__start__", "__end__"}


def _doc_id(doc: Document) -> str:
    for key in ("id", "node_id", "doc_id"):
        if doc.metadata.get(key):
            return str(doc.metadata[key])
    if getattr(doc, "id", None):
        return str(doc.id)
    return "doc_" + hashlib.sha1(doc.page_content.encode("utf-8")).hexdigest()[:10]


def _doc_score(doc: Document) -> Optional[float]:
    for key in _SCORE_KEYS:
        val = doc.metadata.get(key)
        if isinstance(val, (int, float)):
            return float(val)
    return None


class LangGraphTracer(BaseCallbackHandler):
    """Callback handler capturing LangGraph and LangChain agent runs as an AgentTrace."""

    def __init__(self) -> None:
        super().__init__()
        self._t0 = time.perf_counter()
        self._spans: Dict[str, Dict[str, Any]] = {}
        self._order: List[str] = []
        self._retrieval_spans: List[RetrievalSpan] = []
        self._first_query: Optional[str] = None
        self._last_llm_text: Optional[str] = None

    def _now_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000.0

    def _open_span(
        self,
        run_id: UUID,
        parent_run_id: Optional[UUID],
        name: str,
        kind: str,
    ) -> Dict[str, Any]:
        span = {
            "id": str(run_id),
            "name": name,
            "kind": kind,
            "parent_id": str(parent_run_id) if parent_run_id else None,
            "start_ms": self._now_ms(),
            "status": "running",
        }
        self._spans[span["id"]] = span
        self._order.append(span["id"])
        return span

    def _close_span(self, run_id: UUID, status: str = "ok") -> None:
        span = self._spans.get(str(run_id))
        if span is not None:
            span["end_ms"] = self._now_ms()
            span["status"] = status

    @staticmethod
    def _resolve_name(
        serialized: Optional[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]],
        kwargs: Dict[str, Any],
        fallback: str,
    ) -> str:
        return (
            kwargs.get("name")
            or (serialized or {}).get("name")
            or (metadata or {}).get("langgraph_node")
            or fallback
        )

    # Chain events (nodes in LangGraph)
    def on_chain_start(
        self,
        serialized: Optional[Dict[str, Any]],
        inputs: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        name = self._resolve_name(serialized, metadata, kwargs, "chain")
        node = (metadata or {}).get("langgraph_node")
        if name in _NOISE_NAMES or (node is not None and name != node):
            return
        self._open_span(run_id, parent_run_id, name, "node")

    def on_chain_end(self, outputs: Any, *, run_id: UUID, **kwargs: Any) -> None:
        self._close_span(run_id)

    def on_chain_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._close_span(run_id, status="error")

    # Retriever events
    def on_retriever_start(
        self,
        serialized: Optional[Dict[str, Any]],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        if self._first_query is None:
            self._first_query = query
        name = self._resolve_name(serialized, metadata, kwargs, "retriever")
        span = self._open_span(run_id, parent_run_id, name, "retriever")
        self._retrieval_spans.append(
            RetrievalSpan(span_id=span["id"], query=query, arm="vector", items=[])
        )

    def on_retriever_end(
        self, documents: List[Document], *, run_id: UUID, **kwargs: Any
    ) -> None:
        self._close_span(run_id)
        current = next(
            (r for r in reversed(self._retrieval_spans) if r.span_id == str(run_id)), None
        )
        if current is None:
            return

        has_edges = False
        for doc in documents:
            item_id = _doc_id(doc)
            edges = doc.metadata.get("edges") or []
            if edges:
                has_edges = True

            current.items.append(
                RetrievedItem(
                    id=item_id,
                    label=str(doc.metadata.get("label") or doc.metadata.get("title") or item_id),
                    content=doc.page_content,
                    score=_doc_score(doc),
                    metadata={
                        k: v for k, v in doc.metadata.items()
                        if isinstance(v, (str, int, float, bool))
                    },
                )
            )

        current.arm = "graph" if has_edges else "vector"

    def on_retriever_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._close_span(run_id, status="error")

    # LLM events
    def on_llm_start(
        self,
        serialized: Optional[Dict[str, Any]],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        self._open_span(
            run_id, parent_run_id, self._resolve_name(serialized, metadata, kwargs, "llm"), "llm"
        )

    def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> None:
        self._close_span(run_id)
        try:
            self._last_llm_text = response.generations[-1][-1].text or self._last_llm_text
        except (AttributeError, IndexError, TypeError):
            pass

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._close_span(run_id, status="error")

    # Tool events
    def on_tool_start(
        self,
        serialized: Optional[Dict[str, Any]],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        self._open_span(
            run_id, parent_run_id, self._resolve_name(serialized, metadata, kwargs, "tool"), "tool"
        )

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        self._close_span(run_id)

    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._close_span(run_id, status="error")

    def finish(self, query: Optional[str] = None, answer: Optional[str] = None) -> AgentTrace:
        """Assembles the completed AgentTrace and calculates lexical overlap against answer."""
        target_answer = answer or self._last_llm_text
        target_query = query or self._first_query or ""

        # Score retrieved items against answer
        if target_answer:
            for span in self._retrieval_spans:
                for item in span.items:
                    item.answer_overlap = compute_answer_overlap(item.content, target_answer)

        trace = AgentTrace(
            query=target_query,
            answer=target_answer,
            framework="langgraph",
            retrievals=self._retrieval_spans,
            metadata={"total_spans": len(self._spans)},
        )
        return trace
