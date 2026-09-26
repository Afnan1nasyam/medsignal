"""Tests for the LangGraph agent (marked slow; needs the LLM in production).

These run with injected stub services so they are deterministic and offline,
but they exercise the real compiled graph end-to-end.
"""

from __future__ import annotations

import pytest

from src.agents import nodes
from src.agents.graph import run_agent
from src.agents.nodes import AgentServices
from src.models.enums import DataSource, EvidenceGrade
from src.models.schemas import AgentResult, DrugSafetyProfile, RetrievalResult
from src.retrieval.evidence_grader import EvidenceGrader
from src.retrieval.query_planner import QueryPlanner
from src.retrieval.reranker import EvidenceReranker

pytestmark = pytest.mark.slow


class _StubRetriever:
    def __init__(self, source: DataSource):
        self.source = source

    def retrieve(self, query, top_k=10, **kwargs):
        chunks = [
            {"text": f"{self.source.value} evidence {i}", "metadata": {"record_id": f"{self.source.value}-{i}"}, "score": 0.8}
            for i in range(2)
        ]
        return RetrievalResult(chunks=chunks, source=self.source, query=query, result_count=len(chunks))


class _StubGraph:
    def get_safety_profile(self, drug_name):
        return DrugSafetyProfile(drug_name=drug_name, total_faers_reports=5, evidence_summary=f"{drug_name} summary")


def _stub_services(planner) -> AgentServices:
    return AgentServices(
        planner=planner,
        faers=_StubRetriever(DataSource.FAERS),
        pubmed=_StubRetriever(DataSource.PUBMED),
        trials=_StubRetriever(DataSource.CLINICAL_TRIALS),
        labels=_StubRetriever(DataSource.DRUG_LABELS),
        graph=_StubGraph(),
        reranker=EvidenceReranker(),
        grader=EvidenceGrader(),
        llm=None,
    )


@pytest.fixture(autouse=True)
def _reset_services():
    """Ensure the module-level services are reset after each test."""
    yield
    nodes.set_services(None)


def test_agent_returns_result():
    """A basic query returns an AgentResult with a non-empty answer."""
    nodes.set_services(_stub_services(QueryPlanner(llm_client=None)))
    result = run_agent("Is metformin linked to lactic acidosis?", max_iterations=3)
    assert isinstance(result, AgentResult)
    assert result.answer.strip()
    assert result.evidence_grade in EvidenceGrade


def test_agent_handles_error():
    """A failure inside the graph produces a graceful grade-E error result."""

    class _BoomPlanner:
        def plan(self, query):
            raise RuntimeError("planner boom")

    nodes.set_services(_stub_services(_BoomPlanner()))
    result = run_agent("trigger error", max_iterations=2)
    assert result.evidence_grade == EvidenceGrade.E
    assert "error occurred" in result.answer.lower()
