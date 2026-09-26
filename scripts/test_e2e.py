# Requires GROQ_API_KEY. Run on personal laptop.
"""End-to-end smoke test: ingest sample data, then run 5 agent queries.

This exercises the whole stack (ingestion → stores → agent → synthesis) and so
needs the embedding model and the Groq API. It is NOT runnable on the dev
machine (proxy blocks model downloads and Groq); run it on the personal laptop
after setting GROQ_API_KEY.

Run:
    python scripts/test_e2e.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

from src.config import settings

_QUERIES = [
    "What are the common side effects of metformin?",
    "Is there evidence linking metformin to lactic acidosis in elderly patients?",
    "What drug interactions should I watch for with warfarin?",
    "Compare the GI safety of ibuprofen vs aspirin",
    "Does amiodarone affect thyroid function?",
]


def main() -> int:
    """Ingest sample data, run the smoke queries, and print a summary."""
    from src.agents.graph import run_agent
    from src.ingestion.pipeline import IngestionPipeline

    logger.info("Ingesting sample data...")
    pipeline = IngestionPipeline()
    summary = pipeline.ingest_all(data_dir=Path(settings.DATA_DIR) / "sample")
    logger.info("Ingestion summary: {}", summary["counts"])

    print("\n" + "=" * 70)
    print("Running end-to-end smoke queries")
    print("=" * 70)
    for i, query in enumerate(_QUERIES, start=1):
        result = run_agent(query)
        print(f"\n[{i}/{len(_QUERIES)}] {query}")
        print(f"    grade: {result.evidence_grade.value} ({result.evidence_grade.label})")
        print(f"    sources: {', '.join(s.value for s in result.sources_consulted) or 'none'}")
        print(f"    citations: {len(result.citations)} | iterations: {result.iterations_used}")
        print(f"    answer: {result.answer[:200]}...")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
