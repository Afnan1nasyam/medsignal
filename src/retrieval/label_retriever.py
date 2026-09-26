"""Drug-label retrieval — vector search plus full-label section lookup.

Combines Qdrant similarity search over label-section chunks with a direct SQL
fetch of the full structured label, exposing individual sections (warnings,
interactions, adverse reactions, etc.) for the agent to cite verbatim.
"""

from __future__ import annotations

from loguru import logger

from src.models.enums import DataSource
from src.models.schemas import RetrievalResult
from src.storage.sql_store import SQLStore
from src.storage.vector_store import QdrantVectorStore
from src.utils.embedding import BiomedEmbeddingModel

# Section name (and accepted synonyms) -> the SQL row key holding that section.
_SECTION_ALIASES: dict[str, str] = {
    "indications": "indications",
    "contraindications": "contraindications",
    "warnings": "warnings",
    "adverse_reactions": "adverse_reactions",
    "adverse reactions": "adverse_reactions",
    "interactions": "drug_interactions",
    "drug_interactions": "drug_interactions",
    "drug interactions": "drug_interactions",
    "boxed_warning": "boxed_warning",
    "boxed warning": "boxed_warning",
}
# Sections surfaced as chunks when a drug label is pulled from SQL.
_RELEVANT_SECTIONS = (
    "boxed_warning",
    "warnings",
    "contraindications",
    "drug_interactions",
    "adverse_reactions",
    "indications",
)


def _format_section(row: dict, key: str) -> str:
    """Format one label section from a SQL row into readable text (may be empty)."""
    value = row.get(key)
    if not value:
        return ""
    if key == "boxed_warning":
        return str(value)
    if key == "adverse_reactions":
        return "; ".join(
            f"{r.get('reaction', '')} ({r.get('frequency', '')})"
            + (f": {r.get('description')}" if r.get("description") else "")
            for r in value
        )
    if key == "drug_interactions":
        return "; ".join(
            f"{d.get('interacting_drug', '')} ({d.get('severity', '')})"
            + (f": {d.get('description')}" if d.get("description") else "")
            for d in value
        )
    if isinstance(value, list):
        return "; ".join(str(v) for v in value)
    return str(value)


class LabelRetriever:
    """Retrieves drug-label evidence via vector search + structured sections."""

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

    def _fetch_label(self, drug_name: str) -> dict | None:
        """Fetch a label by exact drug name, falling back to a name search."""
        label = self.sql.get_drug_label(drug_name)
        if label is not None:
            return label
        matches = self.sql.search_labels(drug=drug_name)
        return matches[0] if matches else None

    def retrieve(
        self, query: str, drug_name: str | None = None, top_k: int = 10
    ) -> RetrievalResult:
        """Retrieve label evidence relevant to ``query`` (optionally drug-scoped).

        When ``drug_name`` is given, the full structured label is pulled from SQL
        and its relevant sections are appended as chunks so the label content is
        guaranteed present even if vector search missed it.

        Args:
            query: Natural-language query.
            drug_name: Optional drug whose full label should be included.
            top_k: Max vector results to fetch.

        Returns:
            A :class:`RetrievalResult` with label-section chunks.
        """
        chunks: list[dict] = []
        seen_vectors: set[tuple[str, object]] = set()

        try:
            embedding = self.embedding.embed_text(query)
            for hit in self.vector_store.search(
                DataSource.DRUG_LABELS, embedding, top_k=top_k
            ):
                meta = hit.get("metadata", {})
                key = (str(meta.get("record_id", "")), meta.get("chunk_index"))
                if key in seen_vectors:
                    continue
                seen_vectors.add(key)
                chunks.append(
                    {
                        "text": hit.get("text", ""),
                        "metadata": meta,
                        "score": hit.get("score", 0.0),
                        "origin": "vector",
                    }
                )
        except Exception as exc:  # noqa: BLE001 - fall back to SQL-only
            logger.error("Label vector search failed: {}", exc)

        if drug_name:
            try:
                label = self._fetch_label(drug_name)
                if label is None:
                    logger.info("No SQL label found for {}", drug_name)
                else:
                    record_id = label.get("drug_name", drug_name)
                    for section in _RELEVANT_SECTIONS:
                        text = _format_section(label, section)
                        if not text:
                            continue
                        chunks.append(
                            {
                                "text": f"[{record_id}] {section}: {text}",
                                "metadata": {
                                    "record_id": record_id,
                                    "source": DataSource.DRUG_LABELS.value,
                                    "section": section,
                                    "generic_name": label.get("generic_name"),
                                },
                                "score": 0.0,
                                "origin": "sql",
                            }
                        )
            except Exception as exc:  # noqa: BLE001
                logger.error("Label SQL fetch failed for {}: {}", drug_name, exc)

        logger.info("Label retrieval returned {} chunks", len(chunks))
        return RetrievalResult(
            chunks=chunks,
            source=DataSource.DRUG_LABELS,
            query=query,
            result_count=len(chunks),
        )

    def get_drug_label_section(self, drug_name: str, section: str) -> str | None:
        """Return one formatted section of a drug's label, or ``None``.

        Args:
            drug_name: Drug whose label to read.
            section: Section name or synonym (e.g. "warnings", "interactions",
                "adverse reactions", "boxed warning").

        Returns:
            The formatted section text, or ``None`` if the label/section is
            unknown or empty.
        """
        key = _SECTION_ALIASES.get(section.strip().lower())
        if key is None:
            logger.warning("Unknown label section requested: {}", section)
            return None
        label = self._fetch_label(drug_name)
        if label is None:
            return None
        text = _format_section(label, key)
        return text or None
