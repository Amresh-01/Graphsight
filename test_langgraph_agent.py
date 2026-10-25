"""Real-life integration test:
Simulating a LangGraph AI Agent debugging an authentication outage.
"""
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from graphsight_langgraph import LangGraphTracer, capture

# 1. Real knowledge corpus
DATABASE_DOCUMENTS = [
    Document(
        page_content="PR #204 migrates JWT secret keys to AWS Secrets Manager with dynamic rotation every 12 hours.",
        metadata={
            "id": "PR_204",
            "title": "PR #204: Secrets Manager Migration",
            "type": "pull_request",
            "score": 0.96,
            "edges": [
                {"source": "PR_204", "target": "AUTH_SVC", "relation": "TOUCHES"},
                {"source": "PR_204", "target": "INCIDENT_91", "relation": "RESOLVES"}
            ]
        }
    ),
    Document(
        page_content="auth-service handles OAuth2 handshake, session validation, and JWT verification for internal services.",
        metadata={
            "id": "AUTH_SVC",
            "title": "auth-service",
            "type": "service",
            "score": 0.88,
        }
    ),
    Document(
        page_content="Weekly team lunch scheduled for Friday at 1 PM in cafeteria.",
        metadata={
            "id": "DOC_IRRELEVANT",
            "title": "Team Announcements",
            "type": "document",
            "score": 0.52,
        }
    )
]

# 2. Retriever implementation
class DummyRetriever(BaseRetriever):
    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun):
        return DATABASE_DOCUMENTS

# 3. LangGraph Agent definition
class AgentState(TypedDict):
    question: str
    documents: list
    answer: str

def build_real_agent():
    retriever = DummyRetriever()

    def retrieve_node(state: AgentState, config):
        docs = retriever.invoke(state["question"], config=config)
        return {"documents": docs}

    def generate_answer_node(state: AgentState, config):
        # AI models answer based on PR_204 and AUTH_SVC, ignoring DOC_IRRELEVANT
        return {
            "answer": "The outage is caused by PR #204, which migrated JWT secret keys to AWS Secrets Manager with dynamic rotation, affecting auth-service."
        }

    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate_answer", generate_answer_node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate_answer")
    graph.add_edge("generate_answer", END)
    return graph.compile()

# 4. Run real execution with Graphsight Pro Tracer
if __name__ == "__main__":
    print("--- Running Real LangGraph Agent with Graphsight Pro ---")
    agent = build_real_agent()
    tracer = LangGraphTracer()

    question = "Why are internal services failing JWT verification today?"
    print(f"User Query: {question}")
    
    # Run agent with callback
    result = agent.invoke({"question": question}, config={"callbacks": [tracer]})
    
    # Capture trace
    trace_file = capture(tracer, query=question, answer=result["answer"])
    
    print("\n[RESULT]")
    print("AI Answer:", result["answer"])
    print(f"\nTrace successfully captured to: {trace_file}")
    print("\nVerification check:")
    trace = tracer.finish(answer=result["answer"])
    for r in trace.retrievals:
        for item in r.items:
            status = "USED (Highlighted)" if (item.answer_overlap and item.answer_overlap > 0.2) else "UNUSED (Dimmed)"
            print(f" -> {item.label} (Score: {item.score}) => Overlap: {item.answer_overlap} [{status}]")
