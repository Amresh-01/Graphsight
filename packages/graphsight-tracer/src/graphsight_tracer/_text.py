"""Shared lexical processing and overlap scoring for Graphsight.

Zero-cost, zero-LLM heuristic: compares retrieved document tokens against the final answer.
"""
from __future__ import annotations

import re
from typing import Optional, Set

# Base stopwords that carry little semantic significance in technical and natural answers
STOPWORDS: frozenset[str] = frozenset({
    "the", "and", "for", "that", "this", "with", "from", "was", "were", "are",
    "has", "have", "not", "but", "its", "also", "into", "over", "after", "is",
    "been", "being", "then", "them", "they", "will", "would", "could", "should",
    "about", "these", "those", "their", "there", "what", "when", "where", "which"
})

# Words with 3+ characters starting with alphanumeric, allowing common code symbols (_#-.)
WORD_PATTERN: re.Pattern = re.compile(r"[a-z0-9][a-z0-9_#-]{2,}")


def tokenize(
    text: str,
    stop: frozenset[str] = STOPWORDS,
    pattern: re.Pattern = WORD_PATTERN
) -> Set[str]:
    """Extracts cleaned, lowercased meaningful tokens excluding stopwords."""
    if not text:
        return set()
    return {w for w in pattern.findall(text.lower()) if w not in stop}


def compute_answer_overlap(
    chunk_text: str,
    answer_text: Optional[str],
    stop: frozenset[str] = STOPWORDS
) -> Optional[float]:
    """Computes the lexical overlap between a retrieved chunk and the final generated answer.

    Returns:
        float between 0.0 and 1.0 representing token overlap.
        None if the answer is missing/empty or chunk has no valid tokens.

    Formula:
        intersection(chunk_tokens, answer_tokens) / min(len(chunk_tokens), len(answer_tokens))
    """
    if not answer_text or not chunk_text:
        return None

    c_tokens = tokenize(chunk_text, stop=stop)
    a_tokens = tokenize(answer_text, stop=stop)

    if not c_tokens or not a_tokens:
        return 0.0

    common_tokens = c_tokens & a_tokens
    denominator = min(len(c_tokens), len(a_tokens))

    if denominator == 0:
        return 0.0

    score = len(common_tokens) / denominator
    return round(min(score, 1.0), 3)
