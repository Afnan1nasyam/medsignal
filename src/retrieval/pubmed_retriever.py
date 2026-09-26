"""PubMed retrieval — semantic vector search fused with SQL entity filters.

Combines Qdrant similarity search over abstract chunks with SQL filtering by
mentioned drug / adverse event, deduplicating by ``pmid``.
"""

from __future__ import annotations

from loguru import logger

from src.models.enums import DataSource
from src.models.schemas import RetrievalResult
from src.storage.sql_store import SQLStore
from src.storage.vector_store import QdrantVectorStore
from src.utils.embedding import BiomedEmbeddingModel

_ABSTRACT_SNIPPET = 600


def _row_to_text(row: dict) -> str:
    """Render a SQL PubMed row into a title + abstract snippet chunk."""
    abstract = (row.get("abstract") or "").strip()
    if len(abstract) > _ABSTRACT_SNIPPET:
        abstract = abstract[:_ABSTRACT_SNIPPET].rstrip() + "..."
    return f"[{row.get('title', '')}] {abstract}".strip()


class PubMedRetriever:
    """Retrieves PubMed evidence via combined vector + SQL search."""

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
        ae: str | None = None,
        top_k: int = 10,
    ) -> RetrievalResult:
        """Retrieve PubMed articles relevant to ``query`` (optionally scoped).

        Args:
            query: Natural-language query.
            drug: Optional mentioned-drug SQL filter.
            ae: Optional mentioned-adverse-event SQL filter.
            top_k: Max vector results to fetch.

        Returns:
            A :class:`RetrievalResult` with deduplicated article chunks whose
            metadata carries ``pmid`` and ``study_type``.
        """
        chunks: list[dict] = []
        seen: set[str] = set()

        try:
            embedding = self.embedding.embed_text(query)
            for hit in self.vector_store.search(
                DataSource.PUBMED, embedding, top_k=top_k
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
            logger.error("PubMed vector search failed: {}", exc)

        if drug or ae:
            try:
                for row in self.sql.search_pubmed(drug=drug, adverse_event=ae):
                    record_id = str(row.get("pmid", ""))
                    if record_id and record_id in seen:
                        continue
                    if record_id:
                        seen.add(record_id)
                    chunks.append(
                        {
                            "text": _row_to_text(row),
                            "metadata": {
                                "record_id": record_id,
                                "pmid": record_id,
                                "source": DataSource.PUBMED.value,
                                "title": row.get("title"),
                                "journal": row.get("journal"),
                                "study_type": row.get("study_type"),
                                "drugs_mentioned": row.get("drugs_mentioned"),
                                "adverse_events_mentioned": row.get("adverse_events"),
                            },
                            "score": 0.0,
                            "origin": "sql",
                        }
                    )
            except Exception as exc:  # noqa: BLE001
                logger.error("PubMed SQL search failed (drug={}, ae={}): {}", drug, ae, exc)

        logger.info("PubMed retrieval returned {} chunks", len(chunks))
        return RetrievalResult(
            chunks=chunks,
            source=DataSource.PUBMED,
            query=query,
            result_count=len(chunks),
        )
