"""Tests for FastAPI backend endpoints."""

import pytest
from fastapi.testclient import TestClient
from api import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_check(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "graphsight-api"}


def test_ingest_and_query_flow(client):
    # 1. Ingest text document
    ingest_res = client.post(
        "/api/ingest",
        json={
            "doc_id": "test_doc_auth",
            "text": "AuthManager validates JWT tokens and handles user sessions.",
            "source": "auth-guide.md",
        },
    )
    assert ingest_res.status_code == 200
    data = ingest_res.json()
    assert data["status"] == "success"
    assert data["docs"] == 1

    # 2. Query trace
    trace_res = client.post(
        "/api/trace",
        json={"query": "how does AuthManager work?", "top_k": 5},
    )
    assert trace_res.status_code == 200
    trace_data = trace_res.json()
    assert "query" in trace_data
    assert "results" in trace_data
    assert "trace_log" in trace_data
