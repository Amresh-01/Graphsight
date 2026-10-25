"""Simulate a real-life Agent retrieval run using Graphsight Pro."""
from graphsight_langgraph import UniversalAgentTracer, capture

def run_agent_simulation():
    print("1. Initializing Graphsight Pro Universal Tracer...")
    tracer = UniversalAgentTracer(framework="crewai")

    # Step 1: Agent starts investigating
    print("2. Agent is executing steps...")
    sid = tracer.start_span("auth_debugger_agent", kind="agent")

    # Step 2: Agent performs retrieval (pulling 3 documents)
    query = "Why did payment token authorization fail for customer #883?"
    print(f"3. Query retrieved 3 documents for: '{query}'")

    tracer.record_retrieval(
        query=query,
        items=[
            {
                "id": "PR_104",
                "title": "PR #104: Stripe Webhook Migration",
                "score": 0.94,
                "text": "PR #104 migrated payment token authentication to the v3 headers which changed signature requirements.",
            },
            {
                "id": "DOC_99",
                "title": "README: Local Dev Environment",
                "score": 0.81,
                "text": "To set up redis and local docker containers for payment mock service, run docker compose up.",
            },
            {
                "id": "ISSUE_42",
                "title": "Issue #42: Customer #883 Expired Card",
                "score": 0.45,
                "text": "Report of intermittent auth token expiration on retry attempts.",
            }
        ],
        edges=[
            {"source": "PR_104", "target": "ISSUE_42", "relation": "RESOLVES"}
        ],
        span_id=sid
    )

    tracer.end_span(sid)

    # Step 3: LLM generates the final answer (using PR_104, ignoring DOC_99)
    answer = "The payment authorization failed due to PR #104, which migrated payment token authentication to v3 headers and introduced new signature requirements."
    print("4. LLM generated final answer based on evidence.")

    # Step 4: Finish & automatically save trace to .graphsight/
    trace_path = capture(tracer, query=query, answer=answer)
    print(f"\nSUCCESS! Real trace generated and saved to: {trace_path}")
    print("\nTo view the interactive graph in your browser right now, run:")
    print(f"graphsight-pro {trace_path}")

if __name__ == "__main__":
    run_agent_simulation()
