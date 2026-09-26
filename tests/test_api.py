"""Tests for the FastAPI app via TestClient (offline).

Store paths are redirected to a temp directory and the agent dependency is
stubbed, so no real embedding model or LLM is ever loaded.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.config import settings

_DEP_CACHES = [
    "get_vector_store",
    "get_graph_store_dep",
    "get_sql_store",
    "get_embedding_model_dep",
    "get_llm_client",
    "get_agent",
]


class _FakeAgent:
    """Stand-in for the compiled graph; returns a canned final state."""

    def invoke(self, state):
        return {
            "final_answer": "stub answer",
            "evidence_grade": "E",
            "citations": [],
            "retrieved_evidence": [],
            "iteration": 1,
            "plan": None,
        }


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A TestClient with temp store paths and a stubbed agent dependency."""
    monkeypatch.setattr(settings, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(settings, "QDRANT_PATH", str(tmp_path / "qdrant"))
    monkeypatch.setattr(settings, "GRAPH_PATH", str(tmp_path / "kg.json"))
    monkeypatch.setattr(settings, "SQLITE_PATH", str(tmp_path / "medsignal.db"))

    from api import dependencies as deps

    for name in _DEP_CACHES:
        getattr(deps, name).cache_clear()

    from api.dependencies import get_agent
    from api.main import app

    app.dependency_overrides[get_agent] = lambda: _FakeAgent()
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
    for name in _DEP_CACHES:
        getattr(deps, name).cache_clear()


def test_health_check(client):
    """GET / returns 200 with status 'ok'."""
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"]


def test_query_endpoint(client):
    """POST /api/query returns 200 with a structured result (stubbed agent)."""
    response = client.post("/api/query", json={"query": "is metformin safe?"})
    assert response.status_code == 200
    body = response.json()
    assert "result" in body
    assert body["result"]["answer"] == "stub answer"
    assert "processing_time_seconds" in body


def test_drugs_list(client):
    """GET /api/drugs returns a drug list structure."""
    response = client.get("/api/drugs")
    assert response.status_code == 200
    body = response.json()
    assert "drugs" in body and "count" in body
    assert isinstance(body["drugs"], list)


def test_graph_stats(client):
    """GET /api/graph/stats returns a stats dict."""
    response = client.get("/api/graph/stats")
    assert response.status_code == 200
    body = response.json()
    assert "total_nodes" in body
    assert "node_counts" in body
