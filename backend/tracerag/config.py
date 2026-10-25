"""Central configuration for TraceRAG hybrid retrieval engine."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

DB_PATH = Path(os.getenv("TRACERAG_DB_PATH", PROJECT_ROOT / "memory.lbug"))
DATASETS_DIR = Path(os.getenv("TRACERAG_DATASETS_DIR", PROJECT_ROOT / "datasets"))
RESULTS_CSV = Path(os.getenv("TRACERAG_RESULTS_CSV", PROJECT_ROOT / "results.csv"))

# Multi-tenancy & Pod identity
MULTI_TENANCY_ENABLED = os.getenv("MULTI_TENANCY_ENABLED", "0") in ("1", "true", "True")
DEFAULT_TENANT_ORG_ID = os.getenv("DEFAULT_TENANT_ORG_ID", "org_local_dev")
POD_ID = os.getenv("POD_ID", "pod-local-dev")

EMBED_MODEL = os.getenv("TRACERAG_EMBED_MODEL", "all-MiniLM-L6-v2")
EMBED_DIM = 384  # 384-dimensional dense vectors
EMBED_WORKERS = int(os.getenv("TRACERAG_EMBED_WORKERS", "4"))

DB_POOL_SIZE = int(os.getenv("TRACERAG_DB_POOL_SIZE", "10"))
DB_POOL_TIMEOUT = float(os.getenv("TRACERAG_DB_POOL_TIMEOUT", "15"))

MAX_QUERY_CHARS = int(os.getenv("TRACERAG_MAX_QUERY_CHARS", "2000"))
QUERY_EXTRACT_CACHE = int(os.getenv("TRACERAG_QUERY_EXTRACT_CACHE", "512"))

# CORS settings
CORS_ORIGINS = [
    o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()
] or ["*"]

# LLM Providers for intent classification and entity adjudication
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "anthropic/claude-3-haiku")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# Two-tier entity curation thresholds (cosine similarity)
FAST_MERGE_THRESHOLD = float(os.getenv("TRACERAG_FAST_MERGE", "0.92"))
DEEP_MERGE_THRESHOLD = float(os.getenv("TRACERAG_DEEP_MERGE", "0.85"))
GREY_ZONE_LOW = DEEP_MERGE_THRESHOLD
CURATION_TOP_K = int(os.getenv("TRACERAG_CURATION_TOP_K", "5"))

# Hybrid router weights (vector + graph = 1.0)
ROUTER_WEIGHTS_CONCEPTUAL = {"vector": 0.80, "graph": 0.20}
ROUTER_WEIGHTS_RELATIONAL = {"vector": 0.15, "graph": 0.85}

RELATIONAL_QUERY_MARKERS = (
    "who", "whom", "whose", "which", "what caused", "caused by", "because of",
    "related to", "connected to", "depends on", "owns", "owned by", "between",
    "path from", "linked to", "responsible for", "which pr", "which ticket",
)

SEMANTIC_QUERY_MARKERS = (
    "explain", "architecture", "overview", "summary", "summarise", "summarize",
    "describe", "what is", "how does", "concept", "definition", "purpose of",
)

TOP_K_VECTOR = int(os.getenv("TRACERAG_TOP_K_VECTOR", "10"))
TOP_K_GRAPH = int(os.getenv("TRACERAG_TOP_K_GRAPH", "10"))

GRAPH_SEED_TOP_N = int(os.getenv("TRACERAG_GRAPH_SEED_TOP_N", "3"))
GRAPH_SEED_MIN_SIM = float(os.getenv("TRACERAG_GRAPH_SEED_MIN_SIM", "0.35"))
GRAPH_NEIGHBOR_K = int(os.getenv("TRACERAG_GRAPH_NEIGHBOR_K", "5"))
GRAPH_MAX_HOPS = int(os.getenv("TRACERAG_GRAPH_MAX_HOPS", "2"))
MAX_DEGREE = int(os.getenv("TRACERAG_MAX_DEGREE", "10"))  # skip hub nodes above this degree

# Entity Extraction Settings
GLINER_MODEL = os.getenv("TRACERAG_GLINER_MODEL", "urchade/gliner_medium-v2.1")
ENTITY_LABELS = ["Person", "Service", "Library", "Ticket", "PR", "Team", "Tool"]
MIN_ENTITY_CHARS = int(os.getenv("TRACERAG_MIN_ENTITY_CHARS", "3"))
GLINER_THRESHOLD = float(os.getenv("TRACERAG_GLINER_THRESHOLD", "0.55"))
GLINER_WINDOW_WORDS = int(os.getenv("TRACERAG_GLINER_WINDOW_WORDS", "300"))
GLINER_WINDOW_OVERLAP = int(os.getenv("TRACERAG_GLINER_WINDOW_OVERLAP", "50"))

# Recency score decay
RECENCY_ENABLED = os.getenv("TRACERAG_RECENCY_ENABLED", "1") not in ("0", "false", "False")
RECENCY_FLOOR = float(os.getenv("TRACERAG_RECENCY_FLOOR", "0.20"))
DEFAULT_HALF_LIFE_DAYS = float(os.getenv("TRACERAG_DEFAULT_HALF_LIFE_DAYS", "180"))
RECENCY_HALF_LIFE_DAYS = {
    "Ticket": 30.0,
    "PR": 45.0,
    "Commit": 60.0,
    "Document": 180.0,
    "Service": 365.0,
    "Library": 365.0,
    "Person": 365.0,
}

# Graph Schema Table Names
NODE_TABLE = "Entity"
REL_TABLE = "RELATES_TO"
DOC_TABLE = "Document"
MENTIONS_TABLE = "MENTIONS"
VECTOR_INDEX = "idx_entity_embedding"

# Relation types & edge weights
RELATION_AUTHORED = "AUTHORED"
RELATION_REVIEWED = "REVIEWED"
RELATION_RESOLVES = "RESOLVES"
RELATION_TOUCHES = "TOUCHES"
RELATION_PART_OF = "PART_OF"
RELATION_REPORTED = "REPORTED"
RELATION_CO_OCCURS = "CO_OCCURS"

RELATION_WEIGHTS = {
    RELATION_AUTHORED: 0.95,
    RELATION_REVIEWED: 0.85,
    RELATION_RESOLVES: 0.90,
    RELATION_TOUCHES: 0.70,
    RELATION_PART_OF: 0.80,
    RELATION_REPORTED: 0.75,
    RELATION_CO_OCCURS: 0.40,
}
