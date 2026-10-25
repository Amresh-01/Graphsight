"""Tests for graphsight-server HTTP endpoints and normalization."""

import json
import threading
from http.client import HTTPConnection
from pathlib import Path

import pytest
from graphsight_server.server import (
    TRACE_ROUTE,
    RUNS_ROUTE,
    make_server,
    normalize_trace_for_studio,
)


@pytest.fixture
def sample_trace_data():
    return {
        "trace_id": "test_123",
        "query": "What are vector indexes?",
        "answer": "Vector indexes allow fast nearest neighbor search.",
        "framework": "langchain",
        "timestamp": "2026-10-25T12:00:00Z",
        "retrievals": [
            {
                "span_id": "span_1",
                "query": "vector indexing",
                "arm": "vector",
                "items": [
                    {
                        "id": "doc_1",
                        "label": "Vector Index Guide",
                        "content": "Vector indexes enable approximate nearest neighbor search over dense embeddings.",
                        "score": 0.89,
                        "answer_overlap": 0.35,
                    },
                    {
                        "id": "doc_2",
                        "label": "Graph Database Intro",
                        "content": "Graph databases store entities and relationships.",
                        "score": 0.45,
                        "answer_overlap": 0.02,
                    },
                ],
            }
        ],
    }


def test_normalize_trace_for_studio(sample_trace_data):
    normalized = normalize_trace_for_studio(sample_trace_data)
    assert normalized["query"] == "What are vector indexes?"
    assert len(normalized["graph"]["nodes"]) == 2

    node1 = normalized["graph"]["nodes"][0]
    assert node1["id"] == "doc_1"
    assert node1["active"] is True
    assert "overlaps the answer" in node1["meta"]["subtitle"]

    node2 = normalized["graph"]["nodes"][1]
    assert node2["id"] == "doc_2"
    assert node2["active"] is False
    assert "no lexical trace in answer" in node2["meta"]["subtitle"]


def test_server_trace_and_runs_endpoints(tmp_path, sample_trace_data):
    trace_file = tmp_path / "test_run.json"
    trace_file.write_text(json.dumps(sample_trace_data), encoding="utf-8")

    server = make_server(port=4631, trace_path=trace_file, history_dir=tmp_path)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    try:
        conn = HTTPConnection("127.0.0.1", 4631, timeout=5)

        # 1. Fetch trace
        conn.request("GET", TRACE_ROUTE)
        res = conn.getresponse()
        assert res.status == 200
        body = json.loads(res.read().decode("utf-8"))
        assert body["query"] == "What are vector indexes?"
        assert len(body["graph"]["nodes"]) == 2

        # 2. Fetch runs
        conn.request("GET", RUNS_ROUTE)
        res_runs = conn.getresponse()
        assert res_runs.status == 200
        runs_body = json.loads(res_runs.read().decode("utf-8"))
        assert len(runs_body) == 1
        assert runs_body[0]["file"] == "test_run.json"

        # 3. Fetch single run via /__run__/
        conn.request("GET", "/__run__/test_run.json")
        res_single = conn.getresponse()
        assert res_single.status == 200

        # 4. Check SPA fallback on unknown route
        conn.request("GET", "/random/spa/route")
        res_spa = conn.getresponse()
        assert res_spa.status == 200
        assert "html" in res_spa.getheader("Content-Type", "")

    finally:
        server.shutdown()
        server.server_close()
