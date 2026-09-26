"""Clinical-trials retrieval — vector search fused with SQL filters.

Combines Qdrant similarity search over trial-summary chunks with SQL filtering
by intervention drug / condition, deduplicating by ``nct_id``.
"""

from __future__ import annotations

from loguru import logger

from src.models.enums import DataSource
from src.models.schemas import RetrievalResult
from src.storage.sql_store import SQLStore
from src.storage.vector_store import QdrantVectorStore
from src.utils.embedding import BiomedEmbeddingModel


def _row_to_text(row: dict) -> str:
    """Render a SQL trial row into a summary chunk."""
    conditions = ", ".join(row.get("conditions") or []) or "none listed"
    interventions = ", ".join(
        iv.get("name", "") for iv in (row.get("interventions") or []) if iv.get("name")
    ) or "none listed"
    ae_terms = ", ".join(
        ae.get("term", "") for ae in (row.get("adverse_events") or []) if ae.get("term")
    )
    text = (
        f"Trial {row.get('nct_id', '')}: {row.get('title', '')}. "
        f"Phase {row.get('phase', 'NA')}, status {row.get('status', 'unknown')}. "
        f"Conditions: {conditions}. Interventions: {interventions}."
    )
    if ae_terms:
        text += f" Reported adverse events: {ae_terms}."
    return text


class TrialsRetriever:
    """Retrieves clinical-trial evidence via combined vector + SQL search."""

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
        self,
        query: str,
        drug: str | None = None,
        condition: str | None = None,
        top_k: int = 10,
    ) -> RetrievalResult:
        """Retrieve clinical trials relevant to ``query`` (optionally scoped).

        Args:
            query: Natural-language query.
            drug: Optional intervention-drug SQL filter.
            condition: Optional condition SQL filter.
            top_k: Max vector results to fetch.

        Returns:
            A :class:`RetrievalResult` with deduplicated trial chunks whose
            metadata carries ``nct_id``, ``phase`` and ``status``.
        """
        chunks: list[dict] = []
        seen: set[str] = set()

        try:
            embedding = self.embedding.embed_text(query)
            for hit in self.vector_store.search(
                DataSource.CLINICAL_TRIALS, embedding, top_k=top_k
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
            logger.error("Trials vector search failed: {}", exc)

        if drug or condition:
            try:
                for row in self.sql.search_trials(drug=drug, condition=condition):
                    record_id = str(row.get("nct_id", ""))
                    if record_id and record_id in seen:
                        continue
                    if record_id:
                        seen.add(record_id)
                    chunks.append(
                        {
                            "text": _row_to_text(row),
                            "metadata": {
                                "record_id": record_id,
                                "nct_id": record_id,
                                "source": DataSource.CLINICAL_TRIALS.value,
                                "title": row.get("title"),
                                "phase": row.get("phase"),
                                "status": row.get("status"),
                                "conditions": row.get("conditions"),
                            },
                            "score": 0.0,
                            "origin": "sql",
                        }
                    )
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Trials SQL search failed (drug={}, condition={}): {}",
                    drug,
                    condition,
                    exc,
                )

        logger.info("Trials retrieval returned {} chunks", len(chunks))
        return RetrievalResult(
            chunks=chunks,
            source=DataSource.CLINICAL_TRIALS,
            query=query,
            result_count=len(chunks),
        )
