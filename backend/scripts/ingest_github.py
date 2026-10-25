"""Headless GitHub PR & Issue ingestion CLI script.

Usage:
    python scripts/ingest_github.py --repo pallets/click --issues 20 --reset
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tracerag import config
from tracerag.db import TraceDB
from tracerag.github_graph import GitHubGraphBuilder
from tracerag.router import TraceRouter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("graphsight.ingest_github")


def fetch_github_pulls(repo: str, limit: int = 30) -> list[dict]:
    token = os.getenv("GITHUB_TOKEN")
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    url = f"https://api.github.com/repos/{repo}/pulls?state=all&per_page={min(limit, 100)}"
    try:
        res = requests.get(url, headers=headers, timeout=10)
        if res.status_code == 200:
            return res.json()
        logger.warning("GitHub API returned status %s: %s", res.status_code, res.text)
    except Exception as e:
        logger.warning("Failed to fetch GitHub PRs: %s", e)
    return []


def run_github_ingest(repo: str, db_path: Path, limit: int = 30, reset: bool = False):
    if reset and db_path.exists():
        logger.info("Resetting existing database at %s", db_path)
        db_path.unlink()

    db = TraceDB(db_path)
    router = TraceRouter(db)
    builder = GitHubGraphBuilder(db, router.embed_query, repo)

    logger.info("Fetching pulls for repo: %s (limit: %d)", repo, limit)
    pulls = fetch_github_pulls(repo, limit=limit)
    stats = builder.build(pulls=pulls)

    logger.info("GitHub Ingest completed for %s: %d nodes, %d edges created.", repo, stats.nodes, stats.edges)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest GitHub PRs and issues into Graphsight")
    parser.add_argument("--repo", type=str, required=True, help="GitHub repository name (e.g. pallets/click)")
    parser.add_argument("--limit", type=int, default=30, help="Number of PRs to fetch")
    parser.add_argument("--db", type=Path, default=config.DB_PATH, help="Path to database store")
    parser.add_argument("--reset", action="store_true", help="Reset database before ingest")
    args = parser.parse_args()

    run_github_ingest(args.repo, args.db, limit=args.limit, reset=args.reset)


if __name__ == "__main__":
    main()
