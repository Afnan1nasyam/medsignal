"""Tests for the cross-source evidence reranker."""

from __future__ import annotations

from src.models.enums import DataSource
from src.retrieval.reranker import EvidenceReranker

RERANKER = EvidenceReranker()


def test_authority_boost():
    """A label hit outranks a FAERS hit despite lower relevance (authority prior)."""
    results = {
        DataSource.FAERS: [{"text": "faers", "metadata": {"record_id": "F1"}, "score": 0.9}],
        DataSource.DRUG_LABELS: [{"text": "label", "metadata": {"record_id": "L1"}, "score": 0.5}],
    }
    ranked = RERANKER.rerank(results, "q")
    assert ranked[0].metadata.source == DataSource.DRUG_LABELS
    # label: 0.5*0.6 + (5/5)*0.4 = 0.70 ; faers: 0.9*0.6 + (1/5)*0.4 = 0.62
    assert ranked[0].combined_score > ranked[1].combined_score


def test_deduplication():
    """The same drug+AE pair from two sources collapses to the highest-scored one."""
    pair = {"drug_name": "metformin", "adverse_event": "lactic acidosis"}
    results = {
        DataSource.FAERS: [{"text": "faers", "metadata": {"record_id": "F1", **pair}, "score": 0.9}],
        DataSource.PUBMED: [{"text": "pubmed", "metadata": {"record_id": "P1", **pair}, "score": 0.9}],
    }
    ranked = RERANKER.rerank(results, "q")
    assert len(ranked) == 1
    # pubmed authority (3) beats faers (1) at equal relevance -> pubmed kept
    assert ranked[0].metadata.source == DataSource.PUBMED
