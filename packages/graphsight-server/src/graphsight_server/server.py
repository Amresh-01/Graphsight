"""Stdlib HTTP server for Graphsight local trace viewer studio.

Serves:
- Bundled UI SPA with fallback to index.html
- /__trace__.json for a single trace file
- /__runs__.json for history directory runs listing
- /__run__/<filename> for retrieving individual trace runs
"""

from __future__ import annotations

import json
import os
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import unquote

DIST_DIR = Path(__file__).parent / "dist"
TRACE_ROUTE = "/__trace__.json"
RUNS_ROUTE = "/__runs__.json"
RUN_PREFIX = "/__run__/"
_MAX_RUNS = 200


def normalize_trace_for_studio(data: Dict[str, Any]) -> Dict[str, Any]:
    """Ensures trace payload conforms to Studio expectations (TraceState schema).

    Supports both:
    1. Direct TraceState schema (with 'graph', 'steps', etc.)
    2. AgentTrace schema from graphsight-tracer (with 'retrievals', 'query', 'answer')
    """
    if "graph" in data and "nodes" in (data.get("graph") or {}):
        return data

    # Convert graphsight-tracer AgentTrace to Studio format
    nodes: List[Dict[str, Any]] = []
    seen_nodes = set()
    retrievals = data.get("retrievals") or []
    answer = data.get("answer")

    has_answer = bool(answer)
    used_threshold = 0.15

    for r in retrievals:
        arm = r.get("arm", "vector")
        for item in r.get("items") or []:
            item_id = str(item.get("id"))
            if item_id in seen_nodes:
                continue
            seen_nodes.add(item_id)

            overlap = item.get("answer_overlap")
            if has_answer and overlap is not None:
                used = overlap >= used_threshold
                subtitle = (
                    f"overlaps the answer ({overlap:.2f})"
                    if used
                    else "no lexical trace in answer"
                )
            else:
                used = True
                subtitle = f"retrieved via {arm} arm"

            nodes.append({
                "id": item_id,
                "label": item.get("label", item_id),
                "type": item.get("metadata", {}).get("type", "Document"),
                "active": used,
                "position": {"x": 0, "y": 0},
                "similarity": item.get("score"),
                "score": item.get("score"),
                "meta": {
                    "subtitle": subtitle,
                    "snippet": (item.get("content") or "")[:600] or None,
                    "answerOverlap": overlap,
                }
            })

    steps = [
        {
            "id": f"span_{idx}",
            "index": idx,
            "title": f"Retrieval: {r.get('query', 'Query')}",
            "detail": f"{r.get('arm', 'vector')} arm · {len(r.get('items', []))} items",
            "status": "complete",
            "badge": "retriever",
            "arm": r.get("arm", "vector"),
            "durationMs": r.get("duration_ms"),
        }
        for idx, r in enumerate(retrievals)
    ]

    return {
        "id": data.get("trace_id", "trace_local"),
        "query": data.get("query", ""),
        "answer": answer,
        "computedAt": data.get("timestamp"),
        "framework": data.get("framework", "custom"),
        "weights": {"vector": 1.0, "graph": 0.0, "intent": "conceptual"},
        "confidence": {"score": 1.0, "uncertainty": 0.0, "rationale": "Direct local trace"},
        "steps": steps,
        "metrics": {
            "queryTimeSec": 0.0,
            "nodesEvaluated": len(nodes),
            "totalRetrieved": sum(len(r.get("items", [])) for r in retrievals),
            "itemsUsed": sum(1 for n in nodes if n.get("active")),
        },
        "graph": {"nodes": nodes, "edges": []},
        "context": "\n\n".join(
            item.get("content", "")
            for r in retrievals
            for item in r.get("items", [])
            if item.get("content")
        )[:8000] or None,
    }


class StudioHandler(SimpleHTTPRequestHandler):
    """Zero-dependency HTTP handler for studio UI and trace endpoints."""

    trace_path: Optional[Path] = None
    history_dir: Optional[Path] = None

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0].split("#", 1)[0]

        if path == TRACE_ROUTE:
            self._serve_file(self.trace_path, "No trace file provided to graphsight")
            return
        if path == RUNS_ROUTE:
            self._serve_runs()
            return
        if path.startswith(RUN_PREFIX):
            self._serve_run(unquote(path[len(RUN_PREFIX):]))
            return

        # SPA fallback: if file does not exist, serve index.html
        if not os.path.isfile(self.translate_path(path)):
            self.path = "/index.html"
        super().do_GET()

    def _send_json(self, data: Any) -> None:
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _serve_file(self, path: Optional[Path], missing_message: str) -> None:
        if path is None or not path.is_file():
            self.send_error(404, missing_message)
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            normalized = normalize_trace_for_studio(raw)
            self._send_json(normalized)
        except Exception as e:
            self.send_error(500, f"Error reading trace file: {e}")

    def _serve_runs(self) -> None:
        if self.history_dir is None or not self.history_dir.is_dir():
            self.send_error(404, "No history directory provided to graphsight")
            return

        runs = []
        files = sorted(
            self.history_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[:_MAX_RUNS]

        for f in files:
            entry = {"file": f.name}
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                entry["query"] = data.get("query")
                entry["computedAt"] = data.get("computedAt") or data.get("timestamp")
                retrievals = data.get("retrievals") or []
                if "graph" in data:
                    entry["nodes"] = len((data.get("graph") or {}).get("nodes") or [])
                else:
                    entry["nodes"] = sum(len(r.get("items", [])) for r in retrievals)
                runs.append(entry)
            except (OSError, json.JSONDecodeError):
                continue

        self._send_json(runs)

    def _serve_run(self, name: str) -> None:
        # Prevent path traversal attacks
        if (
            self.history_dir is None
            or Path(name).name != name
            or not name.endswith(".json")
        ):
            self.send_error(404, "Invalid or unknown run file")
            return
        self._serve_file(self.history_dir / name, "Run file not found")

    def log_message(self, format: str, *args: Any) -> None:
        # Keep terminal log clean
        pass


def make_server(
    port: int = 4630,
    trace_path: Optional[Path] = None,
    history_dir: Optional[Path] = None,
    dist_dir: Optional[Path] = None,
) -> ThreadingHTTPServer:
    """Creates a configured ThreadingHTTPServer serving graphsight UI and trace APIs."""
    static_dir = dist_dir or DIST_DIR
    if not (static_dir / "index.html").is_file():
        # Fallback minimal index.html if dist not bundled yet
        static_dir.mkdir(parents=True, exist_ok=True)
        fallback_html = static_dir / "index.html"
        if not fallback_html.exists():
            fallback_html.write_text(
                "<!DOCTYPE html><html><head><title>Graphsight Local Viewer</title></head>"
                "<body><h1>Graphsight Studio</h1><p>Trace server running. Use /__trace__.json or /__runs__.json</p></body></html>",
                encoding="utf-8",
            )

    handler = partial(
        type(
            "BoundStudioHandler",
            (StudioHandler,),
            {"trace_path": trace_path, "history_dir": history_dir},
        ),
        directory=str(static_dir),
    )
    return ThreadingHTTPServer(("127.0.0.1", port), handler)
