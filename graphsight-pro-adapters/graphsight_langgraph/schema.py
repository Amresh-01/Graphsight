"""AgentTrace: the framework-neutral trace contract. Plain dataclasses,
zero deps, JSON-ready via asdict."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Literal, Optional

SCHEMA_VERSION = "0.2"

SpanKind = Literal["node", "retriever", "llm", "tool", "agent"]
Arm = Literal["vector", "graph", "hybrid", "unknown"]


@dataclass
class Span:
    """One unit of agent execution (a node, a retriever call, tool execution, etc.)."""

    id: str
    name: str
    kind: SpanKind
    parent_id: Optional[str] = None
    start_ms: float = 0.0
    end_ms: Optional[float] = None
    status: Literal["ok", "error", "running"] = "running"


@dataclass
class RetrievedItem:
    """One retrieved chunk/entity with its provenance."""

    id: str
    label: str
    kind: str = "document"
    score: Optional[float] = None
    vector_score: Optional[float] = None
    graph_score: Optional[float] = None
    content: Optional[str] = None
    source_uri: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    answer_overlap: Optional[float] = None


@dataclass
class TraceEdge:
    """Relational structure between retrieved items."""

    source: str
    target: str
    relation: Optional[str] = None
    weight: Optional[float] = None


@dataclass
class Retrieval:
    """One retrieval event inside the run."""

    span_id: str
    query: str
    arm: Arm = "unknown"
    items: list[RetrievedItem] = field(default_factory=list)
    edges: list[TraceEdge] = field(default_factory=list)


@dataclass
class AgentTrace:
    """The full trace of one agent run."""

    query: str
    framework: str = "langgraph"
    schema_version: str = SCHEMA_VERSION
    spans: list[Span] = field(default_factory=list)
    retrievals: list[Retrieval] = field(default_factory=list)
    answer: Optional[str] = None
    latency_ms: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
