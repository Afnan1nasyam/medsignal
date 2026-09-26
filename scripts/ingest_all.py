"""CLI entry point for running the MedSignal ingestion pipeline.

Loads the four public data sources from ``data/sample`` (or a supplied
directory), builds the SQL store, knowledge graph, and Qdrant vector
collections, and prints a progress display plus a summary table.

Examples:
    python scripts/ingest_all.py                       # ingest all sample data
    python scripts/ingest_all.py --reset               # wipe stores first
    python scripts/ingest_all.py --source faers        # one source only
    python scripts/ingest_all.py --enrich              # LLM-enrich PubMed (laptop)
    python scripts/ingest_all.py --data-dir data/raw   # a different base directory

Note: embedding requires the sentence-transformers model and the full run needs
the Groq API for ``--enrich``; both are intended for the personal laptop.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make ``src`` importable when run as ``python scripts/ingest_all.py``.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger  # noqa: E402

from src.config import settings  # noqa: E402
from src.ingestion.pipeline import IngestionPipeline  # noqa: E402

# Rich is preferred for display but optional; fall back to plain printing.
try:
    from rich.console import Console
    from rich.table import Table

    _CONSOLE: "Console | None" = Console()
except Exception:  # noqa: BLE001 - rich is an optional nicety
    _CONSOLE = None

# source key -> (subdirectory name, pipeline method name)
_SOURCES: dict[str, tuple[str, str]] = {
    "faers": ("faers", "ingest_faers"),
    "pubmed": ("pubmed", "ingest_pubmed"),
    "trials": ("clinical_trials", "ingest_clinical_trials"),
    "labels": ("drug_labels", "ingest_drug_labels"),
}


def _print(message: str) -> None:
    """Print via rich if available, else the standard print."""
    if _CONSOLE is not None:
        _CONSOLE.print(message)
    else:
        print(message)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Ingest MedSignal data into the SQL, graph, and vector stores.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Base directory containing the four source subdirectories "
        "(default: <DATA_DIR>/sample).",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Wipe all stores before ingesting.",
    )
    parser.add_argument(
        "--enrich",
        action="store_true",
        help="LLM-enrich PubMed articles (requires the Groq API).",
    )
    parser.add_argument(
        "--source",
        choices=[*_SOURCES.keys(), "all"],
        default="all",
        help="Which source to ingest (default: all).",
    )
    return parser.parse_args(argv)


def _run_single(pipeline: IngestionPipeline, source: str, base: Path, enrich: bool) -> dict[str, int]:
    """Ingest a single source and return a ``{source: count}`` dict."""
    subdir, method_name = _SOURCES[source]
    method = getattr(pipeline, method_name)
    data_dir = base / subdir

    def _call() -> int:
        if source == "pubmed":
            return method(data_dir, enrich=enrich)
        return method(data_dir)

    if _CONSOLE is not None:
        with _CONSOLE.status(f"[bold cyan]Ingesting {source}...", spinner="dots"):
            count = _call()
    else:
        _print(f"Ingesting {source}...")
        count = _call()
    return {source: count}


def _render_summary(counts: dict[str, int], elapsed: float, stats: dict) -> None:
    """Render the per-source counts and store statistics."""
    if _CONSOLE is not None:
        table = Table(title="Ingestion Summary", show_lines=False)
        table.add_column("Source", style="cyan", no_wrap=True)
        table.add_column("Records ingested", justify="right", style="green")
        for source, count in counts.items():
            table.add_row(source, str(count))
        table.add_row("[bold]TOTAL[/bold]", f"[bold]{sum(counts.values())}[/bold]")
        _CONSOLE.print(table)

        store = Table(title="Store Statistics", show_lines=False)
        store.add_column("Store", style="cyan")
        store.add_column("Detail", style="white")
        sql_stats = stats.get("sql", {})
        store.add_row("SQL tables", ", ".join(f"{k}={v}" for k, v in sql_stats.items()))
        vector_stats = stats.get("vector", {})
        store.add_row("Vector collections", ", ".join(f"{k}={v}" for k, v in vector_stats.items()))
        graph_stats = stats.get("graph", {})
        store.add_row(
            "Graph",
            f"nodes={graph_stats.get('total_nodes', 0)}, edges={graph_stats.get('total_edges', 0)}",
        )
        _CONSOLE.print(store)
        _CONSOLE.print(f"[dim]Completed in {elapsed:.2f}s[/dim]")
    else:
        print("\n=== Ingestion Summary ===")
        for source, count in counts.items():
            print(f"  {source:8s}: {count}")
        print(f"  {'TOTAL':8s}: {sum(counts.values())}")
        print("\n=== Store Statistics ===")
        print(f"  SQL:    {stats.get('sql', {})}")
        print(f"  Vector: {stats.get('vector', {})}")
        graph_stats = stats.get("graph", {})
        print(
            f"  Graph:  nodes={graph_stats.get('total_nodes', 0)}, "
            f"edges={graph_stats.get('total_edges', 0)}"
        )
        print(f"\nCompleted in {elapsed:.2f}s")


def main(argv: list[str] | None = None) -> int:
    """Run ingestion per the parsed CLI arguments; return a process exit code."""
    args = _parse_args(argv)
    base = Path(args.data_dir) if args.data_dir else Path(settings.DATA_DIR) / "sample"

    if not base.exists():
        _print(f"[red]Data directory not found: {base}[/red]" if _CONSOLE else f"Data directory not found: {base}")
        return 1

    _print(f"[bold]MedSignal ingestion[/bold] — base directory: {base}" if _CONSOLE else f"MedSignal ingestion — base directory: {base}")

    pipeline = IngestionPipeline(use_extractor=args.enrich)

    if args.reset:
        _print("Resetting all stores...")
        pipeline.reset_stores()

    start = time.perf_counter()
    if args.source == "all":
        summary = pipeline.ingest_all(data_dir=base, enrich=args.enrich)
        counts = summary["counts"]
        elapsed = summary["total_time_seconds"]
        stats = summary["store_stats"]
    else:
        counts = _run_single(pipeline, args.source, base, args.enrich)
        # Persist the graph for single-source runs too (NetworkX backend only).
        if hasattr(pipeline.graph, "save"):
            pipeline.graph.save()
        elapsed = round(time.perf_counter() - start, 2)
        stats = pipeline.get_stats()

    _render_summary(counts, elapsed, stats)
    logger.info("Ingestion CLI finished: {}", counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
