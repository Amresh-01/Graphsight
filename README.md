# Graphsight 🔍⚡

<p align="center">
  <a href="https://github.com/Amresh-01/Graphsight"><img src="https://img.shields.io/badge/Graphsight-v0.1.0-6366f1?style=for-the-badge&logo=target&logoColor=white" alt="Graphsight Version" /></a>
  <a href="https://github.com/Amresh-01/Graphsight/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-emerald?style=for-the-badge" alt="License" /></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python Version" /></a>
  <a href="https://www.typescriptlang.org/"><img src="https://img.shields.io/badge/TypeScript-React_18-3178C6?style=for-the-badge&logo=typescript&logoColor=white" alt="TypeScript" /></a>
  <a href="http://graphsight.amreshdev.me/"><img src="https://img.shields.io/badge/Live_Demo-graphsight.amreshdev.me-8B5CF6?style=for-the-badge&logo=vercel&logoColor=white" alt="Live Demo" /></a>
</p>

<h3 align="center">
  See what your AI agent retrieved &mdash; and what it <i>actually</i> used.
</h3>

<p align="center">
  <b>Graphsight</b> is a local-first observability, trace visualization, and context intelligence engine for AI agents (LangGraph, LlamaIndex, CrewAI) and GraphRAG pipelines.
</p>

<p align="center">
  <a href="#-the-core-problem">The Problem</a> •
  <a href="#-retrieved-vs-used-the-missing-signal">Retrieved vs Used</a> •
  <a href="#-features--highlights">Features</a> •
  <a href="#-monorepo-architecture">Architecture</a> •
  <a href="#-quickstart">Quickstart</a> •
  <a href="#-framework-adapters">Integrations</a> •
  <a href="#-cli-reference">CLI Reference</a> •
  <a href="#-live-cloud-studio--api">Cloud Studio</a>
</p>

---

## ⚡ The Core Problem

When an agent retrieves 10–15 documents, code files, or memory chunks to answer a prompt, **LLMs typically formulate the final response using only 2 or 3 items**. 

The remaining 80% retrieved chunks:
- 💸 **Inflate token costs** by clogging the LLM context window.
- ⏳ **Increase response latency** (slower prompt processing and time-to-first-token).
- ⚠️ **Dilute reasoning & induce hallucinations** with distractor context.

> **Traditional RAG observability tools only record *what was retrieved*.**  
> **Graphsight reveals the delta between what was retrieved and what was *actually utilized* in the generated answer.**

```
┌─────────────────────────────────────────────────────────────┐
│ 15 Documents Retrieved (Vector / Hybrid Search)             │
│ [Doc 1] [Doc 2] [Doc 3] [Doc 4] [Doc 5] ... [Doc 15]        │
└──────────────────────────┬──────────────────────────────────┘
                           │ Agent Reasoning Loop
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ Final LLM Answer Formulated                                 │
└──────────────────────────┬──────────────────────────────────┘
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
    🟢 2-3 USED ITEMS           🔴 12-13 IGNORED ITEMS
    Grounded the answer;        Wasted tokens, increased latency,
    verified lexical overlap    potential distractor / noise
```

---

## 🎯 Retrieved vs. Used: The Missing Signal

Graphsight runs a **zero-cost lexical overlap heuristic engine** (`answer_overlap`) between each retrieved document and the final synthesized response (at zero LLM call overhead).

In the interactive canvas, each node is visually surfaced:
- 🟢 **Highlighted / Green Accent**: The evidence directly fed into the final output.
- ⚪ **Dimmed / Unused**: High retrieval score, but ignored by the model.

### Pinpoint the 2 Classic Retrieval Failures at a Glance:
1. **Right document retrieved, but ignored by the model** → A dimmed node with a high retrieval score (`score: 0.95`, `overlap: 0.0`). Your retriever worked; your system prompt, context ordering, or model attention failed.
2. **Wrong document trusted / Hallucination** → A low-relevance document that heavily influenced the answer.

---

## ✨ Features & Highlights

- 🕸️ **Interactive Graph & DAG Studio**: Renders nodes as typed entities (`PullRequest`, `Service`, `Ticket`, `Document`, `Person`, `Repo`, `Commit`) with relational dependency edges (`AUTHORED`, `RESOLVES`, `TOUCHES`).
- ⚡ **Zero-Dependency Local Viewer (`graphsight-pro`)**: Uses Python stdlib HTTP server & browser launcher. Runs completely on your machine—**zero telemetry, nothing leaves your localhost**.
- 🔌 **Universal Agent Support**: First-class drop-in callback handlers for **LangGraph**, **LlamaIndex**, **CrewAI**, and raw Python retrieval loops.
- ⏱️ **Execution Timeline**: Step-by-step breakdown of agent execution spans, retriever calls, latency per span, and vector vs. graph retrieval arms.
- 🗂️ **Zero-Database Run History**: Point `graphsight-pro .graphsight/` at a folder to browse historical runs, inspect regressions, and compare runs side-by-side.
- 🌐 **Full-Stack Studio & REST API**: Includes a modern React 18 + Vite + Tailwind CSS studio and a FastAPI backend with GraphMemory, vector routing, and SQLite storage.
- 🐙 **Instant GitHub Tracing (`graphsight-pro-github-trace`)**: Trace any public GitHub repo in 60 seconds with no API keys required.

---

## 📦 Monorepo Architecture

```
Graphsight/
├── graphsight-pro/             # Zero-dependency local trace viewer & CLI launcher (PyPI: graphsight-pro)
├── graphsight-pro-adapters/    # Universal multi-framework tracer (LangGraph, LlamaIndex, CrewAI)
├── frontend/                   # Interactive Trace Studio (React, Vite, Tailwind CSS, Canvas Graph)
├── backend/                    # FastAPI server, GraphMemory, vector router, and SQLite persistence
├── test_real_trace.py          # Standalone simulation demonstrating retrieval vs. used scoring
├── test_langgraph_agent.py     # End-to-end LangGraph agent callback trace example
└── test_real_github_live.py    # Live GitHub multi-hop graph generation test
```

| Component | Path | Description |
|---|---|---|
| **`graphsight-pro`** | [`graphsight-pro/`](graphsight-pro/) | Zero-dependency local viewer CLI (`graphsight-pro .graphsight/`). |
| **`graphsight-pro-adapters`** | [`graphsight-pro-adapters/`](graphsight-pro-adapters/) | Multi-framework agent tracers (`LangGraphTracer`, `LlamaIndexTracer`, `UniversalAgentTracer`). |
| **Studio UI** | [`frontend/`](frontend/) | High-performance React + TypeScript graph explorer and trace inspector. |
| **Backend Engine** | [`backend/`](backend/) | FastAPI GraphRAG context intelligence service, memory indexer, and MCP server. |

---

## 🚀 Quickstart

### 1. Install via pip

```bash
# Install the zero-dependency local viewer
pip install graphsight-pro

# Install framework adapters (LangGraph, LlamaIndex, CrewAI)
pip install graphsight-pro-langgraph
```

*(Or install from local source in editable mode):*
```bash
pip install -e ./graphsight-pro
pip install -e ./graphsight-pro-adapters
```

---

### 2. 60-Second Demo: Trace a Public GitHub Repo

No setup, no accounts, and no tokens required for public repositories:

```bash
# 1. Fetch recent PRs/commits and create a trace graph for a repo
graphsight-pro-github-trace langchain-ai/langgraph "who fixed streaming bugs?"

# 2. Launch the zero-dependency local visualizer
graphsight-pro graphsight_out/trace_state.json
```
Your default browser will pop open automatically at `http://127.0.0.1:4630` showing the interactive entity-relationship graph!

---

## 🧩 Framework Adapters

### 🦜 LangGraph Integration

Attach `LangGraphTracer` directly into your agent's config callbacks:

```python
from langgraph.graph import StateGraph
from graphsight_langgraph import LangGraphTracer, capture

# 1. Create tracer instance
tracer = LangGraphTracer()

# 2. Invoke your agent graph with tracer callback
result = agent_graph.invoke(
    {"question": "Why is the webhook signature failing?"},
    config={"callbacks": [tracer]}
)

# 3. Capture and score retrieved items against the final answer
trace_path = capture(
    tracer, 
    query="Why is the webhook signature failing?", 
    answer=result["answer"]
)

print(f"Trace saved to {trace_path}")
# Open trace in browser:
# $ graphsight-pro .graphsight/
```

---

### 🦙 LlamaIndex Integration

```python
from llama_index.core import Settings, VectorStoreIndex
from graphsight_langgraph import LlamaIndexTracer, capture

tracer = LlamaIndexTracer()
Settings.callback_manager.add_handler(tracer)

query_engine = index.as_query_engine()
response = query_engine.query("Explain customer auth lifecycle.")

trace_path = capture(tracer, query="Explain customer auth lifecycle.", answer=str(response))
```

---

### 🤖 CrewAI / Universal Custom Pipeline

For custom pipelines, vanilla loops, or CrewAI agents:

```python
from graphsight_langgraph import UniversalAgentTracer, capture

tracer = UniversalAgentTracer(framework="crewai")

# Start an execution span
sid = tracer.start_span("debugger_agent", kind="agent")

# Record retrieval items & relational graph edges
tracer.record_retrieval(
    query="Why did payment authorization fail?",
    items=[
        {
            "id": "PR_104", 
            "title": "PR #104: Stripe Webhook Migration", 
            "score": 0.94,
            "text": "Migrated auth to v3 headers and changed signature requirements."
        },
        {
            "id": "DOC_99", 
            "title": "Local Docker Setup", 
            "score": 0.81,
            "text": "Run docker compose up to start redis mock."
        }
    ],
    edges=[{"source": "PR_104", "target": "AUTH_SERVICE", "relation": "TOUCHES"}],
    span_id=sid
)

tracer.end_span(sid)

# Final answer generated by LLM
answer = "The failure was caused by PR #104 which modified signature headers in v3."

# Automatically computes lexical overlap and marks PR_104 as USED, DOC_99 as UNUSED
trace_file = capture(tracer, query="Why did payment authorization fail?", answer=answer)
```

---

## 💻 CLI Reference

`graphsight-pro` comes with a stdlib-only local HTTP web server and command line runner:

```bash
graphsight-pro [path] [--port PORT] [--no-browser]
```

| Command / Option | Description |
|---|---|
| `graphsight-pro .graphsight/` | Opens the **Run History Browser** containing all captured agent sessions. |
| `graphsight-pro path/to/trace.json` | Directly opens a single trace file in the interactive studio. |
| `--port 4630` | Specify custom local port (default: `4630`). |
| `--no-browser` | Launch the server in headless mode without automatically popping open the browser. |

---

## 🖥️ Running Full-Stack Studio Locally

If you wish to run the full Developer Studio frontend and FastAPI backend:

### Backend (FastAPI + GraphMemory)
```bash
cd backend
pip install -r requirements.txt
uvicorn api:app --reload --port 8000
```

### Frontend (React + Vite + Tailwind CSS)
```bash
cd frontend
npm install
npm run dev
```
Open [http://localhost:5173](http://localhost:5173) in your browser.

---

## 📊 Trace Schema (v0.2)

Graphsight traces are portable, standalone JSON files that require no external database:

```json
{
  "schema_version": "0.2",
  "trace_id": "tr_9d42fe81",
  "timestamp": "2026-10-26T15:45:00Z",
  "query": "Why did payment token authorization fail?",
  "answer": "The payment authorization failed due to PR #104...",
  "framework": "langgraph",
  "retrievals": [
    {
      "span_id": "retriever_node",
      "query": "Why did payment token authorization fail?",
      "items": [
        {
          "id": "PR_104",
          "label": "PR #104: Stripe Webhook Migration",
          "content": "PR #104 migrated payment token authentication to v3 headers...",
          "score": 0.94,
          "answer_overlap": 0.78
        },
        {
          "id": "DOC_99",
          "label": "README: Local Dev Environment",
          "content": "To set up redis and local docker containers...",
          "score": 0.81,
          "answer_overlap": 0.04
        }
      ],
      "edges": [
        { "source": "PR_104", "target": "AUTH_SERVICE", "relation": "TOUCHES" }
      ]
    }
  ]
}
```

---

## 🤝 Contributing & License

Contributions are always welcome! Feel free to open an issue or submit a pull request on GitHub.

Distributed under the **MIT License**. See [LICENSE](LICENSE) for more information.

---

<p align="center">
  <b>Built with ❤️ by <a href="https://github.com/Amresh-01">Amresh Chaurasiya</a></b>
  <br />
  <sub>Explore the live web viewer at <a href="http://graphsight.amreshdev.me/">graphsight.amreshdev.me</a></sub>
</p>
