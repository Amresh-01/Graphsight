"""Tests for TraceRAG storage and hybrid routing engine."""

import time
import pytest
from tracerag.config import RELATION_AUTHORED, RELATION_RESOLVES
from tracerag.db import TraceDB
from tracerag.recency import decay_factor, age_days
from tracerag.router import TraceRouter


@pytest.fixture
def test_db(tmp_path):
    db_file = tmp_path / "test_store.sqlite3"
    return TraceDB(db_path=db_file)


def test_recency_decay():
    now = time.time()
    one_month_ago = now - (30 * 86400)
    
    # Ticket has half_life = 30 days
    decay = decay_factor(one_month_ago, node_type="Ticket", now=now)
    assert 0.45 <= decay <= 0.55

    # No timestamp -> factor is 1.0
    assert decay_factor(None, node_type="Ticket", now=now) == 1.0


def test_graph_storage_and_vector_search(test_db):
    test_db.upsert_node(
        node_id="auth_service",
        label="Authentication Service",
        node_type="Service",
        embedding=[0.1, 0.9, 0.0],
        ts=int(time.time()),
    )
    test_db.upsert_node(
        node_id="pr_101",
        label="PR #101: Fix OAuth token exchange",
        node_type="PR",
        embedding=[0.2, 0.8, 0.0],
        ts=int(time.time()),
    )
    test_db.upsert_relation(
        from_id="pr_101",
        to_id="auth_service",
        relation=RELATION_RESOLVES,
        weight=0.9,
    )

    hits = test_db.vector_search([0.15, 0.85, 0.0], k=2)
    assert len(hits) == 2
    assert hits[0]["id"] in ("auth_service", "pr_101")

    # Expand neighbors
    expanded = test_db.expand_frontier(["pr_101"])
    assert "pr_101" in expanded
    assert len(expanded["pr_101"]) == 1
    assert expanded["pr_101"][0]["id"] == "auth_service"


def test_hybrid_router_intent(test_db):
    router = TraceRouter(db=test_db)
    
    # Relational markers -> should classify relational
    _, _, intent_rel = router.classify_intent("Which PR resolved the issue in authentication?")
    assert intent_rel == "relational"

    # Semantic markers -> should classify semantic
    _, _, intent_sem = router.classify_intent("Explain the architecture of the system")
    assert intent_sem == "semantic"
