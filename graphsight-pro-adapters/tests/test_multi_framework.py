"""Unit test for LlamaIndexTracer and UniversalAgentTracer."""
from graphsight_langgraph import LlamaIndexTracer, UniversalAgentTracer


def test_universal_agent_tracer():
    tracer = UniversalAgentTracer(framework="crewai")
    
    # 1. Start agent execution
    sid = tracer.start_span("researcher_agent", kind="agent")
    
    # 2. Record retrieval
    tracer.record_retrieval(
        query="What are the key advantages of Graphsight Pro?",
        items=[
            {
                "id": "doc_1",
                "title": "Graphsight Pro Features",
                "score": 0.95,
                "text": "Graphsight Pro provides zero-dependency tracing, multi-framework support, and interactive graphs.",
            },
            {
                "id": "doc_2",
                "title": "Old Release Notes",
                "score": 0.42,
                "text": "Legacy release notes with outdated details.",
            }
        ],
        edges=[
            {"source": "doc_1", "target": "doc_2", "relation": "EXTENDS"}
        ],
        span_id=sid
    )
    
    tracer.end_span(sid)
    
    # 3. Finish with final answer
    trace = tracer.finish(
        answer="The key advantages of Graphsight Pro include zero-dependency tracing and multi-framework support."
    )
    
    assert trace.framework == "crewai"
    assert len(trace.spans) == 1
    assert len(trace.retrievals) == 1
    
    r = trace.retrievals[0]
    assert r.arm == "graph"
    assert len(r.items) == 2
    
    # Used vs unused lexical heuristic
    item1 = r.items[0]
    assert item1.answer_overlap is not None and item1.answer_overlap > 0.2
    
    item2 = r.items[1]
    assert item2.answer_overlap is not None and item2.answer_overlap < 0.2


def test_llamaindex_tracer_simulation():
    tracer = LlamaIndexTracer()
    
    ev_id = tracer.on_event_start(
        "retrieve",
        payload={"query_str": "Explain retrieval graph tracing"}
    )
    
    class DummyNode:
        node_id = "node_42"
        text = "Retrieval graph tracing maps documents to relationships and answers."
        metadata = {"title": "Architecture Overview"}

    class DummyScoredNode:
        node = DummyNode()
        score = 0.88

    tracer.on_event_end(
        "retrieve",
        payload={"nodes": [DummyScoredNode()]},
        event_id=ev_id
    )
    
    trace = tracer.finish(answer="Retrieval graph tracing helps observe agents.")
    assert trace.framework == "llamaindex"
    assert len(trace.retrievals) == 1
    assert trace.retrievals[0].items[0].id == "node_42"
    assert trace.retrievals[0].items[0].score == 0.88
