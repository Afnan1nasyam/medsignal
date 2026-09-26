"""Cross-source evidence re-ranking with source-authority weighting.

Fuses per-source retrieval hits into a single ranked list, blending semantic
relevance with an authority prior (drug labels > trials > literature > FAERS).
"""

from __future__ import annotations

from loguru import logger

from src.models.enums import DataSource
from src.models.schemas import ChunkMetadata, RerankedResult

# Blend weights for the combined score.
_RELEVANCE_WEIGHT = 0.6
_AUTHORITY_WEIGHT = 0.4
_MAX_AUTHORITY = 5.0


class EvidenceReranker:
    """Ranks and deduplicates retrieval hits across all data sources."""

    #: Source-authority prior (higher = more trustworthy).
    AUTHORITY_SCORES: dict[DataSource, int] = {
        DataSource.DRUG_LABELS: 5,
        DataSource.CLINICAL_TRIALS: 4,
        DataSource.PUBMED: 3,
        DataSource.FAERS: 1,
    }

    def _combined_score(self, relevance: float, authority: int) -> float:
        """Blend relevance and normalized authority into one score."""
        return relevance * _RELEVANCE_WEIGHT + (authority / _MAX_AUTHORITY) * _AUTHORITY_WEIGHT

    @staticmethod
    def _dedup_key(source: DataSource, meta: dict) -> tuple:
        """Compute a deduplication key, preferring an explicit drug+AE pair."""
        drug = str(meta.get("drug_name") or meta.get("drug") or "").lower()
        ae = str(meta.get("adverse_event") or meta.get("ae") or "").lower()
        if drug or ae:
            return ("pair", drug, ae)
        return ("record", source.value, str(meta.get("record_id", "")))

    def _to_metadata(self, source: DataSource, meta: dict) -> ChunkMetadata:
        """Build a :class:`ChunkMetadata` from a raw hit's metadata dict."""
        return ChunkMetadata(
            source=source,
            record_id=str(meta.get("record_id", "")),
            chunk_index=int(meta.get("chunk_index", 0) or 0),
            source_file=meta.get("source_file"),
        )

    def rerank(
        self,
        results: dict[DataSource, list[dict]],
        query: str,
        top_k: int | None = None,
    ) -> list[RerankedResult]:
        """Flatten, score, deduplicate and rank per-source results.

        Args:
            results: Mapping of each :class:`DataSource` to its list of hit
                dicts (``{text, metadata, score}``).
            query: The originating query (accepted for interface symmetry /
                future query-aware scoring).
            top_k: If given, return only the top ``top_k`` results.

        Returns:
            A list of :class:`RerankedResult`, highest combined score first,
            with duplicate drug+AE (or record) hits collapsed to the best one.
        """
        best_by_key: dict[tuple, RerankedResult] = {}

        for source, hits in results.items():
            authority = self.AUTHORITY_SCORES.get(source, 1)
            for hit in hits or []:
                meta = hit.get("metadata", {}) or {}
                relevance = float(hit.get("relevance_score", hit.get("score", 0.0)) or 0.0)
                combined = self._combined_score(relevance, authority)
                candidate = RerankedResult(
                    content=hit.get("text", "") or hit.get("content", ""),
                    metadata=self._to_metadata(source, meta),
                    relevance_score=relevance,
                    authority_score=authority,
                    combined_score=combined,
                )
                key = self._dedup_key(source, meta)
                incumbent = best_by_key.get(key)
                if incumbent is None or candidate.combined_score > incumbent.combined_score:
                    best_by_key[key] = candidate

        ranked = sorted(
            best_by_key.values(), key=lambda r: r.combined_score, reverse=True
        )
        if top_k is not None:
            ranked = ranked[:top_k]
        logger.info("Reranked to {} unique results", len(ranked))
        return ranked
