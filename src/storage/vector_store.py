"""Multi-collection Qdrant vector store (local persistent mode).

Maintains one Qdrant collection per data source so retrieval can be scoped to a
single source or fanned out across all four. Runs fully offline against an
on-disk Qdrant database at ``settings.QDRANT_PATH`` — only the *embeddings* fed
in require a model, so this store itself needs no network.
"""

import uuid
from pathlib import Path
from typing import Any

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client import models as qmodels

from src.config import settings
from src.models.enums import DataSource

# BioLORD-2023-C embedding dimensionality.
VECTOR_SIZE = 768
_UPSERT_BATCH = 100


def _resolve_source(source: DataSource | str) -> DataSource:
    """Coerce a ``DataSource`` or its enum name/value into a ``DataSource``."""
    if isinstance(source, DataSource):
        return source
    text = str(source)
    try:
        return DataSource(text)  # match by value, e.g. "faers"
    except ValueError:
        return DataSource[text.upper()]  # match by name, e.g. "FAERS"


class QdrantVectorStore:
    """A per-source collection wrapper around a local Qdrant instance."""

    #: Maps each data source to its Qdrant collection name.
    COLLECTIONS: dict[DataSource, str] = {
        DataSource.FAERS: "faers_chunks",
        DataSource.PUBMED: "pubmed_chunks",
        DataSource.CLINICAL_TRIALS: "trial_chunks",
        DataSource.DRUG_LABELS: "label_chunks",
    }

    def __init__(self) -> None:
        """Open the local Qdrant DB and ensure all four collections exist."""
        Path(settings.QDRANT_PATH).mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=settings.QDRANT_PATH)
        for source in self.COLLECTIONS:
            self._ensure_collection(source)
        logger.info("QdrantVectorStore ready at {}", settings.QDRANT_PATH)

    # -- collection management --------------------------------------------- #
    def _collection(self, source: DataSource | str) -> str:
        """Return the collection name for a source."""
        return self.COLLECTIONS[_resolve_source(source)]

    def _ensure_collection(self, source: DataSource) -> None:
        """Create the collection for ``source`` if it does not already exist."""
        name = self.COLLECTIONS[source]
        if not self.client.collection_exists(name):
            self.client.create_collection(
                collection_name=name,
                vectors_config=qmodels.VectorParams(
                    size=VECTOR_SIZE, distance=qmodels.Distance.COSINE
                ),
            )
            logger.debug("Created Qdrant collection: {}", name)

    @staticmethod
    def _build_filter(filters: dict | None) -> qmodels.Filter | None:
        """Build a Qdrant payload filter from a ``{key: value}`` dict."""
        if not filters:
            return None
        conditions = [
            qmodels.FieldCondition(key=key, match=qmodels.MatchValue(value=value))
            for key, value in filters.items()
        ]
        return qmodels.Filter(must=conditions)

    # -- writes ------------------------------------------------------------- #
    def add_chunks(
        self,
        source: DataSource | str,
        texts: list[str],
        metadatas: list[dict],
        embeddings: list[list[float]],
    ) -> None:
        """Upsert chunks (text + metadata + vector) into a source collection.

        Args:
            source: Target data source / collection.
            texts: Chunk texts.
            metadatas: Per-chunk metadata dicts (stored in the payload).
            embeddings: Per-chunk embedding vectors.

        Raises:
            ValueError: If the three input lists differ in length.
        """
        if not (len(texts) == len(metadatas) == len(embeddings)):
            raise ValueError("texts, metadatas, and embeddings must be equal length")

        name = self._collection(source)
        points = [
            qmodels.PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={"text": text, **metadata},
            )
            for text, metadata, embedding in zip(texts, metadatas, embeddings)
        ]

        for start in range(0, len(points), _UPSERT_BATCH):
            self.client.upsert(
                collection_name=name, points=points[start : start + _UPSERT_BATCH]
            )
        logger.info("Upserted {} chunks into {}", len(points), name)

    # -- reads -------------------------------------------------------------- #
    @staticmethod
    def _hit_to_dict(point: Any) -> dict:
        """Convert a Qdrant hit into ``{text, metadata, score}``."""
        payload = dict(point.payload or {})
        text = payload.pop("text", "")
        return {
            "text": text,
            "metadata": payload,
            "score": getattr(point, "score", 0.0),
        }

    def search(
        self,
        source: DataSource | str,
        query_embedding: list[float],
        top_k: int | None = None,
        filters: dict | None = None,
    ) -> list[dict]:
        """Search a single source collection by vector similarity.

        Args:
            source: Source collection to search.
            query_embedding: Query vector.
            top_k: Max results (defaults to ``settings.VECTOR_TOP_K``).
            filters: Optional payload equality filters.

        Returns:
            A list of ``{text, metadata, score}`` dicts, highest score first.
        """
        name = self._collection(source)
        limit = top_k if top_k is not None else settings.VECTOR_TOP_K
        response = self.client.query_points(
            collection_name=name,
            query=query_embedding,
            limit=limit,
            query_filter=self._build_filter(filters),
            with_payload=True,
        )
        return [self._hit_to_dict(hit) for hit in response.points]

    def search_all_sources(
        self, query_embedding: list[float], top_k_per_source: int = 5
    ) -> dict[DataSource, list[dict]]:
        """Search every collection and group results by source.

        Args:
            query_embedding: Query vector.
            top_k_per_source: Max results per source.

        Returns:
            A dict mapping each :class:`DataSource` to its result list.
        """
        return {
            source: self.search(source, query_embedding, top_k=top_k_per_source)
            for source in self.COLLECTIONS
        }

    def get_by_record_id(
        self, source: DataSource | str, record_id: str
    ) -> list[dict]:
        """Return all chunks in a source collection for a given ``record_id``.

        Args:
            source: Source collection.
            record_id: The source record identifier stored in the payload.

        Returns:
            A list of ``{text, metadata, score}`` dicts (score 0.0; no ranking).
        """
        name = self._collection(source)
        points, _ = self.client.scroll(
            collection_name=name,
            scroll_filter=self._build_filter({"record_id": record_id}),
            with_payload=True,
            limit=1000,
        )
        return [self._hit_to_dict(point) for point in points]

    # -- stats & maintenance ------------------------------------------------ #
    def count(self, source: DataSource | str | None = None) -> dict | int:
        """Count points in one collection, or all collections.

        Args:
            source: A specific source, or ``None`` for all.

        Returns:
            An int count for a single source, or a ``{collection: count}`` dict.
        """
        if source is not None:
            name = self._collection(source)
            return self.client.count(collection_name=name).count
        return {
            name: self.client.count(collection_name=name).count
            for name in self.COLLECTIONS.values()
        }

    def reset(self, source: DataSource | str | None = None) -> None:
        """Empty one collection, or all collections, leaving them ready to use.

        All points are removed while the (schema-identical) collection is kept.
        We deliberately do *not* drop-and-recreate: qdrant-client's local
        persistent mode leaves a collection's on-disk storage in place after
        ``delete_collection``, so recreating a same-named collection re-attaches
        the old points and the "reset" is a silent no-op. Deleting every point
        via a match-all filter clears the storage reliably instead.

        Args:
            source: A specific source, or ``None`` to reset all.
        """
        targets = [_resolve_source(source)] if source is not None else list(self.COLLECTIONS)
        for target in targets:
            name = self.COLLECTIONS[target]
            self._ensure_collection(target)  # create if it does not exist yet
            self.client.delete(
                collection_name=name,
                points_selector=qmodels.FilterSelector(filter=qmodels.Filter()),
            )
        logger.info("Reset collections: {}", [self.COLLECTIONS[t] for t in targets])

    def get_stats(self) -> dict:
        """Return ``{collection_name: point_count}`` for all collections."""
        return {
            name: self.client.count(collection_name=name).count
            for name in self.COLLECTIONS.values()
        }
