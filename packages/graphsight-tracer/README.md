# graphsight-tracer

Lightweight trace contract and lexical overlap evaluator for AI agents and RAG pipelines.

## Installation

```bash
pip install graphsight-tracer
```

## Features

- Framework-agnostic `AgentTrace`, `RetrievalSpan`, and `RetrievedItem` contracts.
- Zero-cost lexical overlap scoring against agent answers.
- Serializes runs to portable `.graphsight/` JSON format.
