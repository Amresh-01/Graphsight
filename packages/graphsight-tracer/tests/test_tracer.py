"""Unit tests for LangGraphTracer and to_tracestate mapper."""

import uuid
from langchain_core.documents import Document
from graphsight_tracer.mapper import to_tracestate
from graphsight_tracer.tracer import LangGraphTracer


def test_tracer_lifecycle_and_overlap():
    tracer = LangGraphTracer()

    # 1. Chain node start and retriever call
    node_id = uuid.uuid4()
    tracer.on_chain_start(
        serialized={"name": "retrieve_node"},
        inputs={"query": "how to authenticate?"},
        run_id=node_id,
        metadata={"langgraph_node": "retrieve_node"},
    )

    retriever_id = uuid.uuid4()
    tracer.on_retriever_start(
        serialized={"name": "vector_retriever"},
        query="how to authenticate?",
        run_id=retriever_id,
        parent_run_id=node_id,
    )

    # Return 2 documents: 1 used in answer, 1 unused
    docs = [
        Document(
            page_content="Use JWT tokens passed in Authorization header Bearer token.",
            metadata={"id": "doc_auth_jwt", "title": "JWT Auth Guide"},
        ),
        Document(
            page_content="Unrelated database backup procedures and disaster recovery.",
            metadata={"id": "doc_db_backup", "title": "Database Backups"},
        ),
    ]
    tracer.on_retriever_end(documents=docs, run_id=retriever_id)
    tracer.on_chain_end(outputs={}, run_id=node_id)

    # 2. Finish trace with final generated answer
    answer = "Authenticate your requests by including a JWT token in the Authorization Bearer header."
    trace = tracer.finish(query="how to authenticate?", answer=answer)

    assert trace.query == "how to authenticate?"
    assert len(trace.retrievals) == 1
    items = trace.retrievals[0].items
    assert len(items) == 2

    jwt_item = next(i for i in items if i.id == "doc_auth_jwt")
    backup_item = next(i for i in items if i.id == "doc_db_backup")

    assert jwt_item.answer_overlap is not None
    assert jwt_item.answer_overlap > 0.20
    assert jwt_item.is_used is True

    assert backup_item.answer_overlap is not None
    assert backup_item.answer_overlap < 0.10
    assert backup_item.is_used is False

    # 3. Test mapper to Studio TraceState
    studio_state = to_tracestate(trace)
    assert studio_state["query"] == "how to authenticate?"
    nodes = studio_state["graph"]["nodes"]
    assert len(nodes) == 2

    active_node = next(n for n in nodes if n["id"] == "doc_auth_jwt")
    dimmed_node = next(n for n in nodes if n["id"] == "doc_db_backup")
    assert active_node["active"] is True
    assert dimmed_node["active"] is False
