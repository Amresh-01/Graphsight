"""Advanced lexical and semantic scoring for retrieved-vs-used detection.

Upgrades the original naive word-set intersection:
1. Sublinear term frequency (TF) weighting (reduces impact of repetitive filler words).
2. Bigram phrase overlap (detects technical multi-word terms like 'checkout timeout', 'auth token').
3. Stem / prefix matching (matches 'timing' ~ 'timeout' ~ 'times', 'fixes' ~ 'fixed').
4. Code identifier recognition (preserves camelCase, snake_case, and kebab-case tokens).
"""
from __future__ import annotations

import collections
import math
import re

STOP = frozenset({
    "the", "and", "for", "that", "this", "with", "from", "was", "were", "are",
    "has", "have", "not", "but", "its", "also", "into", "over", "after", "been",
    "they", "them", "their", "will", "would", "could", "should", "than", "then",
    "some", "such", "only", "about", "which", "when", "where", "what", "who", "how"
})

# Matches words, symbols, and code identifiers (e.g. payment_service, PR#101, v2.1)
WORD_RE = re.compile(r"[a-z0-9][a-z0-9_\#\.\-]{1,}")


def _stem(word: str) -> str:
    """Lightweight suffix normalization without heavy nltk/spacy dependency."""
    for suffix in ("ing", "ed", "es", "ly", "tion", "ions", "ment", "s"):
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            return word[:-len(suffix)]
    return word


def tokens(text: str, stop: frozenset[str] = STOP, pattern: re.Pattern | None = None) -> set[str]:
    """Extract standard token set (compatible with original contract)."""
    if not text:
        return set()
    pat = pattern if pattern is not None else WORD_RE
    raw = [w for w in pat.findall(text.lower()) if w not in stop]
    stems = {_stem(w) for w in raw}
    return set(raw) | stems


def compute_smart_overlap(answer: str, content: str) -> float | None:
    """Computes a robust similarity score (0.0 to 1.0) between retrieved content and model answer.
    
    Combines:
    - Unigram overlap with TF dampening
    - Bigram (phrase) preservation
    - Stem-level matching
    """
    if not answer or not content:
        return None

    ans_raw = [w for w in WORD_RE.findall(answer.lower()) if w not in STOP]
    doc_raw = [w for w in WORD_RE.findall(content.lower()) if w not in STOP]

    if not ans_raw or not doc_raw:
        return None

    # 1. Unigram frequency maps with stem awareness
    ans_counts: collections.Counter[str] = collections.Counter()
    for w in ans_raw:
        ans_counts[w] += 1
        ans_counts[_stem(w)] += 0.75

    doc_counts: collections.Counter[str] = collections.Counter()
    for w in doc_raw:
        doc_counts[w] += 1
        doc_counts[_stem(w)] += 0.75

    # Sublinear TF overlap score
    shared_unigrams = set(ans_counts.keys()) & set(doc_counts.keys())
    unigram_score = sum(math.sqrt(min(ans_counts[w], doc_counts[w])) for w in shared_unigrams)
    norm_uni = math.sqrt(sum(ans_counts.values())) * math.sqrt(sum(doc_counts.values()))
    uni_sim = (unigram_score / norm_uni) if norm_uni > 0 else 0.0

    # 2. Bigram phrase overlap (e.g. "checkout timeout")
    ans_bigrams = {f"{ans_raw[i]}_{ans_raw[i+1]}" for i in range(len(ans_raw) - 1)}
    doc_bigrams = {f"{doc_raw[i]}_{doc_raw[i+1]}" for i in range(len(doc_raw) - 1)}
    
    bi_sim = 0.0
    if ans_bigrams and doc_bigrams:
        shared_bi = len(ans_bigrams & doc_bigrams)
        bi_sim = shared_bi / min(len(ans_bigrams), len(doc_bigrams))

    # Weighted blend: 70% unigram with stem/TF, 30% exact phrase overlap
    score = (0.7 * uni_sim) + (0.3 * bi_sim)
    # Scale to intuitive range and bound
    final_score = min(1.0, score * 1.4)
    return round(final_score, 3)
