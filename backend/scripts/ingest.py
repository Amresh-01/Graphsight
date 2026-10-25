"""Headless document ingestion CLI script for Graphsight/TraceRAG.

Usage:
    python scripts/ingest.py --datasets ./datasets --reset
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tracerag import config
from tracerag.curation import CurationEngine, IngestStats
from tracerag.db import TraceDB
from tracerag.extract import EntityExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("graphsight.ingest")


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str
    meta: dict


def load_documents(datasets_dir: Path) -> Iterator[Document]:
    if not datasets_dir.exists():
        logger.warning("Datasets directory does not exist: %s", datasets_dir)
        return

    for path in sorted(datasets_dir.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        rel = path.relative_to(datasets_dir).as_posix()
        try:
            if suffix in (".md", ".markdown", ".txt"):
                text = path.read_text(encoding="utf-8", errors="replace")
                yield Document(doc_id=rel, text=text, meta={"source": rel})
            elif suffix == ".json":
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    for idx, item in enumerate(data):
                        text = item.get("text") or item.get("content") or json.dumps(item)
                        yield Document(doc_id=f"{rel}#{idx}", text=text, meta={"source": rel})
                elif isinstance(data, dict):
                    text = data.get("text") or data.get("content") or json.dumps(data)
                    yield Document(doc_id=rel, text=text, meta={"source": rel})
        except Exception as e:
            logger.warning("Failed to load %s: %s", rel, e)


def run_ingest(datasets_dir: Path, db_path: Path, reset: bool = False) -> IngestStats:
    if reset and db_path.exists():
        logger.info("Resetting existing database at %s", db_path)
        db_path.unlink()

    db = TraceDB(db_path)
    extractor = EntityExtractor()
    curation = CurationEngine(db)

    total_stats = IngestStats()
    docs = list(load_documents(datasets_dir))
    logger.info("Found %d documents to ingest from %s", len(docs), datasets_dir)

    for doc in docs:
        entities = extractor.extract(doc.text)
        stats = curation.ingest(
            doc_id=doc.doc_id,
            text=doc.text,
            entities=entities,
            source=doc.meta.get("source"),
        )
        total_stats.docs += stats.docs
        total_stats.entities += stats.entities
        total_stats.created += stats.created
        total_stats.fast_merged += stats.fast_merged
        total_stats.relates_edges += stats.relates_edges
        total_stats.mentions_edges += stats.mentions_edges

    logger.info("Ingestion finished! Docs: %d, Nodes Created: %d, Fast Merged: %d, Edges: %d",
                total_stats.docs, total_stats.created, total_stats.fast_merged, total_stats.relates_edges)
    return total_stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest text and markdown documents into Graphsight")
    parser.add_argument("--datasets", type=Path, default=config.DATASETS_DIR, help="Path to datasets folder")
    parser.add_argument("--db", type=Path, default=config.DB_PATH, help="Path to database store")
    parser.add_argument("--reset", action="store_true", help="Reset database before ingest")
    args = parser.parse_args()

    run_ingest(args.datasets, args.db, reset=args.reset)


if __name__ == "__main__":
    main()
