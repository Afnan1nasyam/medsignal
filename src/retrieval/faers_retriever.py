"""FAERS retrieval — semantic vector search fused with structured SQL lookups.

Combines Qdrant similarity search over FAERS report narratives with exact
SQL filtering by drug/reaction, deduplicating by ``report_id`` so a report
retrieved by both paths appears once.
"""

from __future__ import annotations

from collections import Counter

from loguru import logger

from src.models.enums import DataSource
from src.models.schemas import RetrievalResult
from src.storage.sql_store import SQLStore
from src.storage.vector_store import QdrantVectorStore
from src.utils.embedding import BiomedEmbeddingModel

_TOP_EVENTS = 10


def _row_to_narrative(row: dict) -> str:
    """Render a SQL FAERS row into a compact narrative for a chunk."""
    drugs = ", ".join(
        d.get("name", "") for d in (row.get("drugs") or []) if d.get("name")
    ) or "none recorded"
    reactions = ", ".join(
        r.get("preferred_term", "")
        for r in (row.get("reactions") or [])
        if r.get("preferred_term")
    ) or "none recorded"
    serious = "serious" if row.get("serious") else "non-serious"
    return (
        f"FAERS report {row.get('report_id', '')}: drugs [{drugs}]; "
        f"reactions [{reactions}]; {serious}; "
        f"reporter {row.get('reporter_type', 'unknown')}."
    )


class FaersRetriever:
    """Retrieves FAERS evidence via combined vector + SQL search."""

    def __init__(
        self,
        vector_store: QdrantVectorStore,
        sql_store: SQLStore,
        embedding_model: BiomedEmbeddingModel,
    ) -> None:
        """Store the shared vector store, SQL store, and embedding model."""
        self.vector_store = vector_store
        self.sql = sql_store
        self.embedding = embedding_model

    def retrieve(
        self, query: str, drug_name: str | None = None, top_k: int = 10
    ) -> RetrievalResult:
        """Retrieve FAERS reports relevant to ``query`` (optionally drug-scoped).

        Args:
            query: Natural-language query.
            drug_name: Optional drug to additionally filter via SQL.
            top_k: Max vector results to fetch.

        Returns:
            A :class:`RetrievalResult` with deduplicated report chunks.
        """
        chunks: list[dict] = []
        seen: set[str] = set()

        # -- semantic vector search --
        try:
            embedding = self.embedding.embed_text(query)
            for hit in self.vector_store.search(
                DataSource.FAERS, embedding, top_k=top_k
            ):
                record_id = str(hit.get("metadata", {}).get("record_id", ""))
                if record_id and record_id in seen:
                    continue
                if record_id:
                    seen.add(record_id)
                chunks.append(
                    {
                        "text": hit.get("text", ""),
                        "metadata": hit.get("metadata", {}),
                        "score": hit.get("score", 0.0),
                        "origin": "vector",
                    }
                )
        except Exception as exc:  # noqa: BLE001 - fall back to SQL-only
            logger.error("FAERS vector search failed: {}", exc)

        # -- structured SQL lookup (drug-scoped) --
        if drug_name:
            try:
                counts = self.get_drug_event_counts(drug_name)
                for row in self.sql.search_faers(drug_name=drug_name):
                    record_id = str(row.get("report_id", ""))
                    if record_id and record_id in seen:
                        continue
                    if record_id:
                        seen.add(record_id)
                    chunks.append(
                        {
                            "text": _row_to_narrative(row),
                            "metadata": {
                                "record_id": record_id,
                                "source": DataSource.FAERS.value,
                                "serious": row.get("serious"),
                                "reporter_type": row.get("reporter_type"),
                                "drug_event_counts": counts.get("event_counts", {}),
                            },
                            "score": 0.0,
                            "origin": "sql",
                        }
                    )
            except Exception as exc:  # noqa: BLE001 - never crash retrieval
                logger.error("FAERS SQL search failed for {}: {}", drug_name, exc)

        logger.info("FAERS retrieval returned {} chunks", len(chunks))
        return RetrievalResult(
            chunks=chunks,
            source=DataSource.FAERS,
            query=query,
            result_count=len(chunks),
        )

    def get_drug_event_counts(self, drug_name: str) -> dict:
        """Aggregate FAERS reaction counts for a drug from the SQL store.

        Args:
            drug_name: Drug to aggregate reactions for.

        Returns:
            A dict with the total report count, a ``{term: count}`` map, and the
            top reactions by frequency.
        """
        try:
            rows = self.sql.search_faers(drug_name=drug_name)
        except Exception as exc:  # noqa: BLE001
            logger.error("FAERS count aggregation failed for {}: {}", drug_name, exc)
            return {"drug": drug_name, "total_reports": 0, "event_counts": {}, "top_events": []}

        term_counts: Counter[str] = Counter()
        for row in rows:
            for reaction in row.get("reactions") or []:
                term = reaction.get("preferred_term")
                if term:
                    term_counts[term] += 1

        top_events = [
            {"reaction": term, "count": count}
            for term, count in term_counts.most_common(_TOP_EVENTS)
        ]
        return {
            "drug": drug_name,
            "total_reports": len(rows),
            "event_counts": dict(term_counts),
            "top_events": top_events,
        }
