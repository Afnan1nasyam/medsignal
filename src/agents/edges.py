"""Conditional-edge functions for the agent graph."""

from __future__ import annotations

from loguru import logger

from src.agents.nodes import selected_sources
from src.agents.state import AgentState
from src.config import settings
from src.models.enums import DataSource

# Map each targetable source to its retrieval node name.
_SOURCE_TO_NODE = {
    DataSource.FAERS: "retrieve_faers",
    DataSource.PUBMED: "retrieve_pubmed",
    DataSource.CLINICAL_TRIALS: "retrieve_trials",
    DataSource.DRUG_LABELS: "retrieve_labels",
}


def should_continue(state: AgentState) -> str:
    """Route after evaluation: to synthesis, another retrieval pass, or error.

    Returns one of ``"synthesize"``, ``"retrieve_more"``, or ``"error"``.
    """
    if state.get("error"):
        return "error"
    if state.get("is_sufficient"):
        return "synthesize"
    if state.get("iteration", 0) >= state.get("max_iterations", settings.AGENT_MAX_ITERATIONS):
        return "synthesize"
    return "retrieve_more"


def select_sources(state: AgentState) -> list[str]:
    """Return the retrieval node names to run for the current sub-query.

    Enables source-scoped fan-out; the graph module also gates each retriever
    internally, so this is safe whether used with the Send API or plain edges.
    Includes ``retrieve_graph`` on the first pass when the plan named drugs.
    """
    nodes = [
        _SOURCE_TO_NODE[src]
        for src in selected_sources(state)
        if src in _SOURCE_TO_NODE
    ]
    plan = state.get("plan")
    if plan and plan.drugs_mentioned and state.get("current_step", 0) == 0:
        nodes.append("retrieve_graph")
    logger.debug("select_sources -> {}", nodes)
    return nodes or ["retrieve_faers"]
