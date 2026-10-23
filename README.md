# Graphsight (Monorepo)

**See what your agent retrieved, and what it actually used.**

Graphsight is a local-first observability and context intelligence engine for AI agents and RAG (Retrieval-Augmented Generation) applications.

---

## 📦 Monorepo Architecture

This repository is split into modular, focused packages:

| Package | Path | Description |
|---|---|---|
| [`graphsight-tracer`](packages/graphsight-tracer) | `packages/graphsight-tracer` | Core trace schema, zero-cost lexical overlap engine, and agent callbacks. |
| [`graphsight-server`](packages/graphsight-server) | `packages/graphsight-server` | Zero-dependency local server and CLI (`graphsight .graphsight/`). |
| [`graphsight-ui`](packages/graphsight-ui) | `packages/graphsight-ui` | Developer Studio UI (React + Vite + Tailwind) for trace inspection. |

---

## ⚡ The Core Problem

When an agent retrieves 10-15 documents or code chunks to answer a query, LLMs typically formulate the final answer using only 2 or 3 items. The remaining retrieved documents:
- **Inflate token costs** by clogging the context window.
- **Increase response latency**.
- **Introduce hallucination risks** through irrelevant context.

Traditional RAG observability tools only record *what was retrieved*. **Graphsight reveals the delta between what was retrieved and what was actually utilized in the answer.**

---

## 🏗️ Step-by-Step Monorepo Roadmap

- [x] **Day 1: Monorepo Setup & Core Trace Contracts**
  - Modular `packages/` monorepo structure.
  - Define `RetrievedItem`, `RetrievalSpan`, and `AgentTrace` data models in `graphsight-tracer`.
  - Serialization to portable `.graphsight/` JSON format.
- [ ] **Day 2: Zero-Cost Lexical Overlap Engine (`_text.py`)**
  - Fast token intersection and lexical trace heuristic (zero LLM overhead).
- [ ] **Day 3: Custom RAG Pipeline & Tracer Client (`capture.py`)**
  - Capture traces from vanilla Python retrieval loops.
- [ ] **Day 4: LangGraph & LangChain Callback Tracer (`tracer.py`)**
  - Automatic callback integration into agent state graphs.
- [ ] **Day 5: Local Zero-Dependency Trace Viewer Server (`server.py`)**
  - Python stdlib HTTP server for local inspection (`127.0.0.1:4630`).
- [ ] **Day 6: CLI Tool (`cli.py`)**
  - Terminal launcher command (`graphsight .graphsight/`).
- [ ] **Day 7: Interactive Trace Studio UI (`graphsight-ui`)**
  - Visual DAG and itemized *Used vs. Ignored* chunk inspection.

---

## 🚀 Quickstart

### 1. Install Tracer Package
```bash
cd packages/graphsight-tracer
pip install -e .
```

### 2. Creating Your First Trace
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
