"""Tests for Day 2 lexical overlap engine and trace capture."""

from pathlib import Path
from graphsight_tracer._text import compute_answer_overlap, tokenize
from graphsight_tracer.capture import capture, score_trace
from graphsight_tracer.schema import AgentTrace, RetrievalSpan, RetrievedItem


def test_tokenize():
    text = "The quick brown fox jumps over PR #101 with webhook_error."
    tokens = tokenize(text)
    # Stopwords ("the", "over", "with") must be excluded
    assert "the" not in tokens
    assert "over" not in tokens
    assert "with" not in tokens
    # Key tokens must be preserved
    assert "quick" in tokens
    assert "brown" in tokens
    assert "fox" in tokens
    assert "webhook_error" in tokens


def test_compute_answer_overlap():
    chunk_used = "The payment webhook timed out because the database pool connection was exhausted."
    chunk_ignored = "User profile pictures should be compressed before uploading to S3."
    answer = "The issue was caused by database connection pool exhaustion in the payment webhook."

    score_used = compute_answer_overlap(chunk_used, answer)
    score_ignored = compute_answer_overlap(chunk_ignored, answer)

    assert score_used is not None
    assert score_ignored is not None
    assert score_used > 0.40  # High overlap
    assert score_ignored == 0.0  # Zero overlap


def test_capture_and_scoring_pipeline(tmp_path: Path):
    trace = AgentTrace(query="Why is checkout slow?")
    span = RetrievalSpan(span_id="retrieve", query=trace.query)

    span.items.append(RetrievedItem(
        id="c1",
        label="db.py",
        content="Checkout latency increases due to unindexed SQL queries."
    ))
    span.items.append(RetrievedItem(
        id="c2",
        label="auth.py",
        content="JWT auth tokens are refreshed every 15 minutes."
    ))
    trace.retrievals.append(span)

    # Capture with answer
    answer = "Checkout is slow because SQL queries lack indexing."
    trace_path = capture(trace, answer=answer, output_dir=tmp_path)

    assert trace_path.exists()
    assert trace.retrievals[0].items[0].is_used is True
    assert trace.retrievals[0].items[1].is_used is False
