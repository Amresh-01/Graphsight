"""Integration test for full GraphMemory ingestion and retrieval flow."""

import pytest
from tracerag.memory import GraphMemory


def test_full_memory_flow(tmp_path):
    db_path = tmp_path / "integration.sqlite3"
    
    with GraphMemory(db_path) as mem:
        # 1. Ingest text document containing entities
        stats = mem.ingest_text(
            doc_id="doc_pr_auth",
            text="PaymentService handles checkout payments and was resolved by PR #104.",
            source="pr-notes.md",
        )
        assert stats.docs == 1
        assert stats.entities >= 2

        # 2. Ingest structured GitHub PR
        gh_stats = mem.ingest_github(
            repo="acme/platform",
            pulls=[{
                "number": 104,
                "title": "Fix card tokenization bug",
                "body": "Fixes #882 in PaymentService",
                "user": {"login": "alice"},
                "created_at": "2026-07-20T10:00:00Z"
            }]
        )
        assert gh_stats.nodes >= 2

        # 3. Query the unified memory
        resp = mem.query("who touched PaymentService?")
        assert resp.query == "who touched PaymentService?"
        assert resp.trace_log["intent"] in ("relational", "semantic")
        assert len(resp.results) > 0
