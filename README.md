# Graphsight

**See what your agent retrieved, and what it actually used.**

Graphsight is a local-first observability and context intelligence engine for AI agents and RAG (Retrieval-Augmented Generation) applications.

---

## ⚡ The Core Problem

When an agent retrieves 10-15 documents or code chunks to answer a query, LLMs typically formulate the final answer using only 2 or 3 items. The remaining retrieved documents:
- **Inflate token costs** by clogging the context window.
- **Increase response latency**.
- **Introduce hallucination risks** through irrelevant context.

Traditional RAG observability tools only record *what was retrieved*. **Graphsight reveals the delta between what was retrieved and what was actually utilized in the answer.**

---

## 🏗️ Architecture & Development Roadmap

- [x] **Day 1: Core Trace Contracts & Data Models**
  - Define `RetrievedItem`, `RetrievalSpan`, and `AgentTrace` data models.
  - Serialization to portable `.graphsight/` JSON format.
- [ ] **Day 2: Zero-Cost Lexical Overlap Engine**
  - Fast token intersection and lexical trace heuristic (zero LLM overhead).
- [ ] **Day 3: Custom RAG Pipeline & Tracer Client**
  - Capture traces from vanilla Python retrieval loops.
- [ ] **Day 4: LangGraph & LangChain Callback Tracer**
  - Automatic callback integration into agent state graphs.
- [ ] **Day 5: Local Zero-Dependency Trace Viewer Server**
  - Python stdlib HTTP server for local inspection (`127.0.0.1:4630`).
- [ ] **Day 6: Interactive Trace Studio UI**
  - Visual DAG and itemized *Used vs. Ignored* chunk inspection.
- [ ] **Day 7: Run Diffing & Benchmarks**
  - Trace comparison across prompt and chunking iterations.

---

## 🚀 Quickstart (Day 1)

```bash
# Clone the repository
git clone https://github.com/Amresh-01/Graphsight.git
cd Graphsight

# Install package in editable mode
pip install -e .
```

### Creating Your First Trace

```python
from graphsight.models import AgentTrace, RetrievalSpan, RetrievedItem

trace = AgentTrace(
    query="Why is the payment webhook timing out?",
    answer="The payment webhook timed out because of missing database connection pooling.",
    framework="custom"
)

span = RetrievalSpan(span_id="vector_search", query=trace.query, arm="vector")
span.items.append(RetrievedItem(
    id="chunk_1",
    label="webhook_handler.ts",
    content="The payment webhook handles Stripe events and queries DB pool directly.",
    score=0.92,
    answer_overlap=0.65
))

trace.retrievals.append(span)
saved_path = trace.save()
print(f"Trace saved to {saved_path}")
```

---

## 📜 License
MIT License
