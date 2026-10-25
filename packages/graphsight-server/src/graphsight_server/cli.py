"""Graphsight CLI: Inspect agent retrieval traces in your local browser."""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path
from typing import Optional

from .server import RUNS_ROUTE, TRACE_ROUTE, make_server


def main(argv: Optional[list[str]] = None) -> None:
    """CLI entrypoint: graphsight [path] [--port 4630] [--no-browser]"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except (AttributeError, OSError):
            pass

    parser = argparse.ArgumentParser(
        prog="graphsight",
        description="Open a trace JSON file or a history directory (e.g. .graphsight/) in your browser.",
    )
    parser.add_argument(
        "trace",
        nargs="?",
        type=Path,
        help="Path to a trace JSON file or history directory (default: current directory or .graphsight/)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=4630,
        help="Port to serve on (default: 4630)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not automatically launch a browser window",
    )

    args = parser.parse_args(argv)

    trace_path: Optional[Path] = None
    history_dir: Optional[Path] = None

    if args.trace is not None:
        target = args.trace.resolve()
        if target.is_dir():
            history_dir = target
        elif target.is_file():
            trace_path = target
        else:
            parser.error(f"Target path does not exist: {target}")
    else:
        # Default behavior: look for .graphsight/ in cwd, otherwise use cwd
        default_dir = Path(".graphsight").resolve()
        if default_dir.is_dir():
            history_dir = default_dir
        else:
            history_dir = Path.cwd().resolve()

    server = make_server(port=args.port, trace_path=trace_path, history_dir=history_dir)
    url = f"http://127.0.0.1:{args.port}"
    if trace_path is not None:
        url += f"?src={TRACE_ROUTE}"
    elif history_dir is not None:
        url += f"?runs={RUNS_ROUTE}"

    print(f"\n⚡ Graphsight Trace Viewer running at: {url}")
    if trace_path is not None:
        print(f"   Traced file: {trace_path}")
    if history_dir is not None:
        print(f"   History dir: {history_dir}")
    print("   Press Ctrl+C to stop.\n")

    if not args.no_browser:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Graphsight server... goodbye!")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
