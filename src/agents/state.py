"""LangGraph agent state schema.

Defines the :class:`AgentState` ``TypedDict`` threaded through the graph. List
fields that accumulate across (possibly parallel) retrieval nodes use
``Annotated[..., operator.add]`` so LangGraph concatenates their updates rather
than overwriting; ``source_results`` uses a dict-merge reducer for the same
reason.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from src.models.schemas import Citation, Evidence, QueryPlan


def merge_dicts(left: dict, right: dict) -> dict:
    """Reducer that shallow-merges two dicts (right wins on key collision)."""
    return {**(left or {}), **(right or {})}


class AgentState(TypedDict, total=False):
    """State object passed between LangGraph nodes.

    Fields mirror the Agent State Schema in ``ARCHITECTURE.md``. Accumulating
    list fields (``retrieved_evidence``, ``contradictions``, ``citations``) and
    ``source_results`` carry reducers so concurrent node updates merge cleanly.
    """

    # -- inputs / plan --
    query: str
    plan: QueryPlan | None
    max_iterations: int

    # -- loop control --
    current_step: int
    iteration: int
    is_sufficient: bool

    # -- accumulated retrieval state --
    retrieved_evidence: Annotated[list[Evidence], operator.add]
    source_results: Annotated[dict[str, Any], merge_dicts]
    contradictions: Annotated[list[dict], operator.add]

    # -- outputs --
    final_answer: str
    citations: Annotated[list[Citation], operator.add]
    evidence_grade: str
    error: str | None


def initial_state(query: str, max_iterations: int) -> AgentState:
    """Build a fresh :class:`AgentState` for a new query run."""
    return {
        "query": query,
        "plan": None,
        "max_iterations": max_iterations,
        "current_step": 0,
        "iteration": 0,
        "is_sufficient": False,
        "retrieved_evidence": [],
        "source_results": {},
        "contradictions": [],
        "final_answer": "",
        "citations": [],
        "evidence_grade": "",
        "error": None,
    }
