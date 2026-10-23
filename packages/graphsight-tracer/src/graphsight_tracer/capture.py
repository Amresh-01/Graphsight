"""Local run history: save and capture traces to .graphsight/ directory."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

from ._text import compute_answer_overlap
from .schema import AgentTrace


def save_trace(trace: AgentTrace, output_dir: Union[str, Path, None] = None) -> Path:
    """Writes an AgentTrace to local directory in portable JSON format.

    Default target directory is $GRAPHSIGHT_DIR or ./.graphsight.
    """
    target = Path(output_dir or os.environ.get("GRAPHSIGHT_DIR") or ".graphsight")
    return trace.save(output_dir=target)


def score_trace(trace: AgentTrace, answer: Optional[str] = None) -> AgentTrace:
    """Applies zero-cost lexical overlap scoring to all retrieved items across all spans."""
    target_answer = answer or trace.answer
    if not target_answer:
        return trace

    trace.answer = target_answer

    for span in trace.retrievals:
        for item in span.items:
            item.answer_overlap = compute_answer_overlap(
                chunk_text=item.content,
                answer_text=target_answer
            )

    return trace


def capture(
    trace: AgentTrace,
    answer: Optional[str] = None,
    output_dir: Union[str, Path, None] = None
) -> Path:
    """Scores retrieved items against answer and writes trace file in one call."""
    scored_trace = score_trace(trace, answer=answer)
    return save_trace(scored_trace, output_dir=output_dir)
