"""Query route — run the agentic RAG engine over a drug-safety question."""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends
from loguru import logger

from api.dependencies import get_agent, get_llm_client, get_sql_store
from src.agents.graph import _to_agent_result
from src.agents.state import initial_state
from src.config import settings
from src.models.enums import EvidenceGrade
from src.models.schemas import AgentResult, QueryRequest, QueryResponse
from src.retrieval.query_planner import QueryPlanner

router = APIRouter(prefix="/api", tags=["query"])


def _fallback_result(query: str) -> AgentResult:
    """Build a minimal keyword-planned result when the agent cannot run."""
    plan = QueryPlanner(llm_client=None).plan_without_llm(query)
    return AgentResult(
        answer=(
            "The reasoning engine is unavailable, so a full answer could not be "
            "synthesized. A keyword-based plan was produced instead; ingest data "
            "and configure the LLM/embedding model for complete answers."
        ),
        citations=[],
        evidence_grade=EvidenceGrade.E,
        sources_consulted=[],
        iterations_used=0,
        query_plan=plan,
    )


@router.post("/query", response_model=QueryResponse)
async def run_query(request: QueryRequest, agent=Depends(get_agent)) -> QueryResponse:
    """Answer a drug-safety question via the multi-source agentic RAG engine.

    Runs the compiled LangGraph agent (planning → multi-source retrieval →
    evidence evaluation → graded synthesis). If the agent raises, falls back to
    a keyword-based plan so the endpoint still returns a structured response.
    """
    start = time.time()
    max_iters = request.max_iterations or settings.AGENT_MAX_ITERATIONS
    try:
        final_state = agent.invoke(initial_state(request.query, max_iters))
        result = _to_agent_result(final_state)
    except Exception as exc:  # noqa: BLE001 - degrade gracefully
        logger.error("Agent run failed; using keyword fallback: {}", exc)
        result = _fallback_result(request.query)

    elapsed = time.time() - start

    # Best-effort audit logging (never fails the request).
    try:
        get_sql_store().log_query(
            query=request.query,
            intent=result.query_plan.intent.value if result.query_plan else None,
            sources=[s.value for s in result.sources_consulted],
            evidence_grade=result.evidence_grade.value,
            result_count=len(result.citations),
            processing_time=elapsed,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Query logging failed: {}", exc)

    logger.info("Query answered in {:.2f}s (grade {})", elapsed, result.evidence_grade.value)
    return QueryResponse(result=result, processing_time_seconds=elapsed)
