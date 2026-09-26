"""Reset MedSignal stores (vector / graph / SQL).

Examples:
    python scripts/reset_stores.py --all --force
    python scripts/reset_stores.py --vectors --sql
    python scripts/reset_stores.py --graph --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Reset MedSignal data stores.")
    parser.add_argument("--all", action="store_true", help="Reset every store.")
    parser.add_argument("--vectors", action="store_true", help="Reset the Qdrant vector store.")
    parser.add_argument("--graph", action="store_true", help="Reset the knowledge graph.")
    parser.add_argument("--sql", action="store_true", help="Reset the SQLite store.")
    parser.add_argument("--force", action="store_true", help="Skip the confirmation prompt.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Reset the requested stores; returns a process exit code."""
    args = _parse_args(argv)
    targets = {
        "vectors": args.all or args.vectors,
        "graph": args.all or args.graph,
        "sql": args.all or args.sql,
    }
    if not any(targets.values()):
        print("Nothing to do. Pass --all or one of --vectors/--graph/--sql.")
        return 1

    selected = [name for name, on in targets.items() if on]
    if not args.force:
        answer = input(f"Reset these stores permanently: {', '.join(selected)}? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            print("Aborted.")
            return 1

    if targets["vectors"]:
        from src.storage.vector_store import QdrantVectorStore

        QdrantVectorStore().reset()
        logger.info("Vector store reset.")
    if targets["graph"]:
        from src.storage.graph_store import get_graph_store

        get_graph_store().reset()
        logger.info("Knowledge graph reset.")
    if targets["sql"]:
        from src.storage.sql_store import SQLStore

        SQLStore().reset()
        logger.info("SQL store reset.")

    print(f"Reset complete: {', '.join(selected)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
