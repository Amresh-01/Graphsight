"""graphsight-pro-adapters: capture LangGraph, LlamaIndex, CrewAI, and custom agent runs as Graphsight traces.

    # 1. LangGraph:
    from graphsight_langgraph import LangGraphTracer, capture
    tracer = LangGraphTracer()
    result = graph.invoke(inputs, config={"callbacks": [tracer]})
    capture(tracer, query="...", answer=result.get("answer"))

    # 2. LlamaIndex:
    from graphsight_langgraph import LlamaIndexTracer
    llama_tracer = LlamaIndexTracer()
    # pass into Settings.callback_manager

    # 3. Universal / CrewAI:
    from graphsight_langgraph import UniversalAgentTracer
    agent_tracer = UniversalAgentTracer(framework="crewai")

Then browse every run with the viewer: `graphsight-pro .graphsight/`
"""
from .capture import capture, save_trace
from .llamaindex_tracer import LlamaIndexTracer
from .mapper import to_tracestate
from .schema import (
    SCHEMA_VERSION,
    AgentTrace,
    Retrieval,
    RetrievedItem,
    Span,
    TraceEdge,
)
from .tracer import LangGraphTracer
from .universal_tracer import UniversalAgentTracer

__all__ = [
    "SCHEMA_VERSION",
    "AgentTrace",
    "LangGraphTracer",
    "LlamaIndexTracer",
    "UniversalAgentTracer",
    "Retrieval",
    "RetrievedItem",
    "Span",
    "TraceEdge",
    "capture",
    "save_trace",
    "to_tracestate",
]
