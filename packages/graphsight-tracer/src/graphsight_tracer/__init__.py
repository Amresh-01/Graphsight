"""Graphsight Tracer - Lightweight trace contract and lexical overlap evaluator."""

__version__ = "0.1.0"

from ._text import compute_answer_overlap, tokenize
from .capture import capture, save_trace, score_trace
from .mapper import to_tracestate
from .schema import AgentTrace, RetrievalSpan, RetrievedItem
from .tracer import LangGraphTracer

__all__ = [
    "AgentTrace",
    "RetrievalSpan",
    "RetrievedItem",
    "LangGraphTracer",
    "tokenize",
    "compute_answer_overlap",
    "save_trace",
    "score_trace",
    "capture",
    "to_tracestate",
]
