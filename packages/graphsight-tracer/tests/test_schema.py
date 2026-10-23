"""Tests for graphsight-tracer schema."""

from pathlib import Path
from graphsight_tracer.schema import AgentTrace, RetrievalSpan, RetrievedItem


def test_retrieved_item_used_status():
    item_used = RetrievedItem(
        id="chunk_1",
        label="doc1.txt",
        content="Database connection timed out",
        score=0.9,
        answer_overlap=0.45
    )
    assert item_used.is_used is True

    item_ignored = RetrievedItem(
        id="chunk_2",
        label="doc2.txt",
        content="Frontend animation styling",
        score=0.85,
        answer_overlap=0.02
    )
    assert item_ignored.is_used is False


def test_agent_trace_summary_and_save(tmp_path: Path):
    trace = AgentTrace(
        query="Why is payment failing?",
        answer="Missing database connection pool caused payment gateway timeout.",
        framework="custom"
    )

    span = RetrievalSpan(span_id="span_1", query=trace.query, arm="vector")
    span.items.append(RetrievedItem(id="c1", label="db.py", content="DB pool config", answer_overlap=0.6))
    span.items.append(RetrievedItem(id="c2", label="auth.py", content="Auth token", answer_overlap=0.01))

    trace.retrievals.append(span)

    summary = trace.summary()
    assert summary["total_items_retrieved"] == 2
    assert summary["items_used"] == 1
    assert summary["items_ignored"] == 1
    assert summary["retrieval_utilization_rate"] == 0.5

    filepath = trace.save(output_dir=tmp_path)
    assert filepath.exists()
    assert filepath.suffix == ".json"
