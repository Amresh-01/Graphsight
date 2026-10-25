"""Unit tests for chat sessions and history tracking."""

import pytest
from fastapi.testclient import TestClient
from api import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_session_lifecycle_and_history(client):
    # 1. Create a session
    res = client.post("/api/sessions", json={"title": "Session on Payments"})
    assert res.status_code == 200
    session_data = res.json()
    assert session_data["title"] == "Session on Payments"
    session_id = session_data["id"]

    # 2. List sessions
    list_res = client.get("/api/sessions")
    assert list_res.status_code == 200
    sessions = list_res.json()
    assert any(s["id"] == session_id for s in sessions)

    # 3. Check traces for this session
    traces_res = client.get(f"/api/sessions/{session_id}/traces")
    assert traces_res.status_code == 200
    assert isinstance(traces_res.json(), list)
