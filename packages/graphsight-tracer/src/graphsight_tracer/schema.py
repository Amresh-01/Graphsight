"""Core schema defining Graphsight trace contracts."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


def _slugify(text: str, max_length: int = 40) -> str:
    """Generate a clean slug for file naming from a query string."""
    cleaned = re.sub(r"[^\w\s-]", "", text.lower())
    slug = re.sub(r"[-\s]+", "-", cleaned).strip("-")
    return slug[:max_length] or "trace"


class RetrievedItem(BaseModel):
    """Represents a single document or chunk retrieved during execution."""
    id: str
    label: str
    content: str
    score: Optional[float] = None
    answer_overlap: Optional[float] = Field(
        default=None,
        description="Lexical overlap score against final answer (0.0 to 1.0). None if answer is unavailable."
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def is_used(self) -> Optional[bool]:
        """Returns True if the item shows meaningful lexical trace in the final answer."""
        if self.answer_overlap is None:
            return None
        return self.answer_overlap > 0.15


class RetrievalSpan(BaseModel):
    """Represents a discrete retrieval step (e.g., vector search, graph walk)."""
    span_id: str
    query: str
    arm: str = Field(default="vector", description="'vector', 'graph', 'keyword', or 'hybrid'")
    items: List[RetrievedItem] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    duration_ms: Optional[float] = None

    @property
    def total_retrieved(self) -> int:
        return len(self.items)

    @property
    def used_count(self) -> int:
        return sum(1 for item in self.items if item.is_used is True)

    @property
    def ignored_count(self) -> int:
        return sum(1 for item in self.items if item.is_used is False)


class AgentTrace(BaseModel):
    """Represents an entire traced execution run of an agent or RAG pipeline."""
    trace_id: Optional[str] = None
    query: str
    answer: Optional[str] = None
    framework: str = Field(default="custom", description="'custom', 'langgraph', 'langchain', or 'llamaindex'")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    retrievals: List[RetrievalSpan] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        """Calculates high-level metrics for the entire run."""
        total_items = sum(span.total_retrieved for span in self.retrievals)
        used_items = sum(span.used_count for span in self.retrievals)
        ignored_items = sum(span.ignored_count for span in self.retrievals)

        utilization_rate = (used_items / total_items) if total_items > 0 else 0.0

        return {
            "query": self.query,
            "has_answer": bool(self.answer),
            "total_retrieval_spans": len(self.retrievals),
            "total_items_retrieved": total_items,
            "items_used": used_items,
            "items_ignored": ignored_items,
            "retrieval_utilization_rate": round(utilization_rate, 3)
        }

    def save(self, output_dir: str | Path = ".graphsight") -> Path:
        """Saves trace as a JSON artifact in the target directory."""
        dir_path = Path(output_dir)
        dir_path.mkdir(parents=True, exist_ok=True)

        timestamp_str = self.timestamp.strftime("%Y%m%dT%H%M%S")
        slug = _slugify(self.query)
        filename = f"{timestamp_str}_{slug}.json"
        filepath = dir_path / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(mode="json"), f, indent=2)

        return filepath
