"""Graphsight Tracer - Lightweight trace contract and lexical overlap evaluator."""

__version__ = "0.1.0"

from ._text import compute_answer_overlap, tokenize
from .capture import capture, save_trace, score_trace
from .schema import AgentTrace, RetrievalSpan, RetrievedItem

__all__ = [
    "AgentTrace",
    "RetrievalSpan",
    "RetrievedItem",
    "tokenize",
    "compute_answer_overlap",
    "save_trace",
    "score_trace",
    "capture",
]
