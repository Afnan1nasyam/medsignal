"""Offline end-to-end integration test over the committed sample data.

Runs the full non-LLM path — loaders -> SQL -> knowledge graph -> Qdrant -> the
agent — using a deterministic **stub embedder** so no sentence-transformers model
is downloaded. Stores are redirected to a temp directory. This is the only test
that exercises the real ingestion pipeline against real sample files.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from src.config import settings

_REPO = Path(__file__).resolve().parents[1]
_SAMPLE = _REPO / "data" / "sample"
_VECTOR_SIZE = 768


class _StubEmbedder:
    """Deterministic offline embedder: a sparse one-hot 768-vector per text."""

    def _vec(self, text: str) -> list[float]:
        bucket = int(hashlib.md5((text or "").encode("utf-8")).hexdigest(), 16) % _VECTOR_SIZE
        vec = [0.0] * _VECTOR_SIZE
        vec[bucket] = 1.0
        return vec

    def embed_text(self, text: str) -> list[float]:
        return self._vec(text)

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def get_embedding_dim(self) -> int:
        return _VECTOR_SIZE


@pytest.fixture
def wired(tmp_path, monkeypatch):
    """Redirect stores to temp paths and return a pipeline with a stub embedder."""
    if not _SAMPLE.exists() or not any((_SAMPLE / "faers").glob("*.json")):
        pytest.skip("sample data not present; run scripts/seed_sample_data.py")

    monkeypatch.setattr(settings, "QDRANT_PATH", str(tmp_path / "qdrant"))
    monkeypatch.setattr(settings, "GRAPH_PATH", str(tmp_path / "kg.json"))
    monkeypatch.setattr(settings, "SQLITE_PATH", str(tmp_path / "db.sqlite"))

    from src.ingestion.pipeline import IngestionPipeline

    pipeline = IngestionPipeline()
    pipeline.embedding = _StubEmbedder()  # no model download
    return pipeline


def test_full_ingestion(wired):
    """Ingesting the sample data populates SQL, the graph, and Qdrant."""
    summary = wired.ingest_all(data_dir=_SAMPLE)
    assert summary["counts"] == {"faers": 50, "pubmed": 30, "trials": 20, "labels": 15}

    stats = wired.get_stats()
    sql = stats["sql"]
    assert sql["faers_reports"] == 50
    assert sql["pubmed_articles"] == 30
    assert sql["clinical_trials"] == 20
    assert sql["drug_labels"] == 15

    # Every source collection received chunks.
    vector = stats["vector"]
    assert all(count > 0 for count in vector.values()), vector

    # Graph has the core drugs and the key edge types.
    graph = stats["graph"]
    assert graph["node_counts"].get("drug", 0) >= 15
    assert graph["node_counts"].get("adverse_event", 0) > 0
    for edge_type in ("REPORTED_WITH", "INTERACTS_WITH", "SAME_CLASS_AS"):
        assert edge_type in graph["edge_counts"], edge_type

    # Persisted graph file was written.
    assert Path(settings.GRAPH_PATH).exists()


def test_graph_signal_metformin(wired):
    """The metformin -> lactic acidosis FAERS signal aggregated correctly."""
    wired.ingest_all(data_dir=_SAMPLE)
    profile = wired.graph.get_drug_profile("metformin")
    assert profile["total_faers_reports"] > 0
    terms = {ae["preferred_term"].lower() for ae in profile["top_adverse_events"]}
    assert any("lactic acidosis" in t for t in terms)


def test_agent_end_to_end_offline(wired):
    """The agent answers over the ingested stores with no LLM (keyword + stub)."""
    wired.ingest_all(data_dir=_SAMPLE)

    from src.agents import nodes
    from src.agents.graph import run_agent
    from src.agents.nodes import AgentServices
    from src.models.enums import EvidenceGrade
    from src.retrieval.evidence_grader import EvidenceGrader
    from src.retrieval.faers_retriever import FaersRetriever
    from src.retrieval.graph_retriever import GraphRetriever
    from src.retrieval.label_retriever import LabelRetriever
    from src.retrieval.pubmed_retriever import PubMedRetriever
    from src.retrieval.query_planner import QueryPlanner
    from src.retrieval.reranker import EvidenceReranker
    from src.retrieval.trials_retriever import TrialsRetriever

    vs, sql, emb = wired.vector_store, wired.sql, wired.embedding
    nodes.set_services(
        AgentServices(
            planner=QueryPlanner(llm_client=None),
            faers=FaersRetriever(vs, sql, emb),
            pubmed=PubMedRetriever(vs, sql, emb),
            trials=TrialsRetriever(vs, sql, emb),
            labels=LabelRetriever(vs, sql, emb),
            graph=GraphRetriever(wired.graph),
            reranker=EvidenceReranker(),
            grader=EvidenceGrader(),
            llm=None,
        )
    )
    try:
        result = run_agent("Is metformin linked to lactic acidosis?", max_iterations=3)
        assert result.answer.strip()
        assert result.evidence_grade in EvidenceGrade
        assert result.sources_consulted, "expected at least one source consulted"
        assert result.citations, "expected citations from real ingested evidence"
    finally:
        nodes.set_services(None)
