"""LangGraph state-machine definition for the agentic RAG engine.

Builds a compiled graph that plans a query, retrieves from the targeted sources
(each retriever self-gates to the current sub-query), evaluates sufficiency, and
loops up to ``max_iterations`` before synthesizing a graded, cited answer. The
retrieval stage is sequential for robustness — each retriever is a no-op when
its source is not selected — which keeps the graph simple and always runnable.
"""

from __future__ import annotations

from loguru import logger

from langgraph.graph import END, START, StateGraph

from src.agents.edges import should_continue
from src.agents.nodes import (
    evaluate_evidence,
    handle_error,
    plan_query,
    retrieve_faers,
    retrieve_graph,
    retrieve_labels,
    retrieve_pubmed,
    retrieve_trials,
    synthesize_answer,
)
from src.agents.state import AgentState, initial_state
from src.config import settings
from src.models.enums import DataSource, EvidenceGrade
from src.models.schemas import AgentResult, QueryPlan

# Order of the sequential retrieval chain (loop head is the first entry).
_RETRIEVAL_CHAIN = [
    "retrieve_faers",
    "retrieve_pubmed",
    "retrieve_trials",
    "retrieve_labels",
    "retrieve_graph",
]


def build_agent_graph():
    """Construct and compile the agent :class:`~langgraph.graph.StateGraph`.

    Returns:
        The compiled LangGraph runnable.
    """
    builder = StateGraph(AgentState)

    builder.add_node("plan_query", plan_query)
    builder.add_node("retrieve_faers", retrieve_faers)
    builder.add_node("retrieve_pubmed", retrieve_pubmed)
    builder.add_node("retrieve_trials", retrieve_trials)
    builder.add_node("retrieve_labels", retrieve_labels)
    builder.add_node("retrieve_graph", retrieve_graph)
    builder.add_node("evaluate_evidence", evaluate_evidence)
    builder.add_node("synthesize_answer", synthesize_answer)
    builder.add_node("handle_error", handle_error)

    # START -> plan -> retrieval chain -> evaluate
    builder.add_edge(START, "plan_query")
    builder.add_edge("plan_query", _RETRIEVAL_CHAIN[0])
    for src_node, dst_node in zip(_RETRIEVAL_CHAIN, _RETRIEVAL_CHAIN[1:]):
        builder.add_edge(src_node, dst_node)
    builder.add_edge(_RETRIEVAL_CHAIN[-1], "evaluate_evidence")

    # evaluate -> {synthesize | loop back to retrieval head | error}
    builder.add_conditional_edges(
        "evaluate_evidence",
        should_continue,
        {
            "synthesize": "synthesize_answer",
            "retrieve_more": _RETRIEVAL_CHAIN[0],
            "error": "handle_error",
        },
    )

    builder.add_edge("synthesize_answer", END)
    builder.add_edge("handle_error", END)

    compiled = builder.compile()
    logger.info("Agent graph compiled")
    return compiled


def _sources_consulted(state: AgentState) -> list[DataSource]:
    """Derive the distinct data sources that contributed evidence."""
    sources: set[DataSource] = {e.source for e in state.get("retrieved_evidence", [])}
    return sorted(sources, key=lambda s: s.value)


def _to_agent_result(state: AgentState) -> AgentResult:
    """Extract a validated :class:`AgentResult` from the final graph state."""
    grade_str = state.get("evidence_grade") or EvidenceGrade.E.value
    try:
        grade = EvidenceGrade(grade_str)
    except ValueError:
        grade = EvidenceGrade.E

    plan: QueryPlan | None = state.get("plan")
    return AgentResult(
        answer=state.get("final_answer", ""),
        citations=state.get("citations", []),
        evidence_grade=grade,
        sources_consulted=_sources_consulted(state),
        iterations_used=state.get("iteration", 0),
        query_plan=plan,
    )


# Module-level compiled graph, built lazily so importing this module is cheap.
_GRAPH = None


def get_agent_graph():
    """Return the process-wide compiled agent graph (built once)."""
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_agent_graph()
    return _GRAPH


def run_agent(query: str, max_iterations: int = 3) -> AgentResult:
    """Run the agent end-to-end on a query and return the final result.

    Args:
        query: The user's drug-safety question.
        max_iterations: Maximum retrieval/evaluation loops.

    Returns:
        The synthesized :class:`AgentResult`.
    """
    max_iters = max_iterations or settings.AGENT_MAX_ITERATIONS
    graph = get_agent_graph()
    state = initial_state(query, max_iters)
    logger.info("Running agent for query: {}", query)
    final_state = graph.invoke(state)
    return _to_agent_result(final_state)
