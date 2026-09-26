"""LangGraph node functions for the agentic RAG engine.

Each node takes the :class:`AgentState` and returns a partial-state update dict.
Nodes are wrapped in ``try/except`` and log via loguru so a single failure
degrades gracefully (surfacing ``error`` for the graph to route to
``handle_error``) instead of crashing the run.

Shared services (retrievers, planner, reranker, grader, LLM) live in an
injectable :class:`AgentServices` container built lazily on first use, so
importing this module performs no store/model construction and tests can supply
stubs via :func:`set_services`.
"""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from src.agents.state import AgentState
from src.config import settings
from src.models.enums import DataSource, EvidenceGrade
from src.models.schemas import Citation, Evidence, QueryPlan
from src.retrieval.evidence_grader import EvidenceGrader
from src.retrieval.faers_retriever import FaersRetriever
from src.retrieval.graph_retriever import GraphRetriever
from src.retrieval.label_retriever import LabelRetriever
from src.retrieval.pubmed_retriever import PubMedRetriever
from src.retrieval.query_planner import QueryPlanner
from src.retrieval.reranker import EvidenceReranker
from src.retrieval.trials_retriever import TrialsRetriever

# Per-source authority weights (mirrors EvidenceReranker.AUTHORITY_SCORES).
_AUTHORITY = {
    DataSource.DRUG_LABELS: 5,
    DataSource.CLINICAL_TRIALS: 4,
    DataSource.PUBMED: 3,
    DataSource.FAERS: 1,
}

# System prompt (V1) for final multi-source synthesis.
SYNTHESIS_PROMPT_V1 = (  # V1
    "You are a drug safety intelligence analyst. Based on the following evidence "
    "from multiple sources, provide a comprehensive answer. Cite each claim with "
    "[Source: ID]. Grade your confidence. Flag contradictions."
)


@dataclass
class AgentServices:
    """Container for the shared components every node depends on."""

    planner: QueryPlanner
    faers: FaersRetriever
    pubmed: PubMedRetriever
    trials: TrialsRetriever
    labels: LabelRetriever
    graph: GraphRetriever
    reranker: EvidenceReranker
    grader: EvidenceGrader
    llm: object | None  # GroqClient | None


_SERVICES: AgentServices | None = None


def _build_default_services() -> AgentServices:
    """Construct the default services from real stores (offline-safe to build)."""
    from src.storage.sql_store import SQLStore
    from src.storage.vector_store import QdrantVectorStore
    from src.storage.graph_store import get_graph_store
    from src.utils.embedding import get_embedding_model

    vector_store = QdrantVectorStore()
    sql_store = SQLStore()
    embedding = get_embedding_model()
    graph_store = get_graph_store()

    llm: object | None = None
    try:
        from src.utils.llm_client import GroqClient

        llm = GroqClient()
    except Exception as exc:  # noqa: BLE001 - Groq/deps may be unavailable
        logger.warning("GroqClient unavailable ({}); using offline planner/synthesis", exc)

    return AgentServices(
        planner=QueryPlanner(llm_client=llm),  # type: ignore[arg-type]
        faers=FaersRetriever(vector_store, sql_store, embedding),
        pubmed=PubMedRetriever(vector_store, sql_store, embedding),
        trials=TrialsRetriever(vector_store, sql_store, embedding),
        labels=LabelRetriever(vector_store, sql_store, embedding),
        graph=GraphRetriever(graph_store),
        reranker=EvidenceReranker(),
        grader=EvidenceGrader(),
        llm=llm,
    )


def get_services() -> AgentServices:
    """Return the process-wide services, building defaults on first use."""
    global _SERVICES
    if _SERVICES is None:
        _SERVICES = _build_default_services()
    return _SERVICES


def set_services(services: AgentServices | None) -> None:
    """Inject (or clear) the services container — used for testing."""
    global _SERVICES
    _SERVICES = services


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _current_subquery(state: AgentState):
    """Return the sub-query for the current step (clamped), or ``None``."""
    plan: QueryPlan | None = state.get("plan")
    if plan and plan.sub_queries:
        step = min(state.get("current_step", 0), len(plan.sub_queries) - 1)
        return plan.sub_queries[step]
    return None


def selected_sources(state: AgentState) -> set[DataSource]:
    """Sources targeted by the current sub-query (all four if no plan)."""
    sub = _current_subquery(state)
    if sub and sub.target_sources:
        return set(sub.target_sources)
    return set(_AUTHORITY.keys())


def _query_for(state: AgentState) -> str:
    """Text to search with for the current step (sub-query, else raw query)."""
    sub = _current_subquery(state)
    return sub.query if sub else state.get("query", "")


def _first_drug(state: AgentState) -> str | None:
    """First drug mentioned in the plan, if any."""
    plan: QueryPlan | None = state.get("plan")
    if plan and plan.drugs_mentioned:
        return plan.drugs_mentioned[0]
    return None


def _first_ae(state: AgentState) -> str | None:
    """First adverse event mentioned in the plan, if any."""
    plan: QueryPlan | None = state.get("plan")
    if plan and plan.adverse_events_mentioned:
        return plan.adverse_events_mentioned[0]
    return None


def _to_evidence(chunk: dict, source: DataSource) -> Evidence:
    """Convert a retriever chunk dict into an :class:`Evidence` object."""
    metadata = chunk.get("metadata", {}) or {}
    return Evidence(
        source=source,
        content=chunk.get("text", "") or chunk.get("content", ""),
        metadata=metadata,
        relevance_score=float(chunk.get("score", 0.0) or 0.0),
        source_authority=_AUTHORITY[source],
    )


# --------------------------------------------------------------------------- #
# Planning
# --------------------------------------------------------------------------- #
def plan_query(state: AgentState) -> dict:
    """Plan the query via the LLM planner (falls back to keyword planner)."""
    try:
        services = get_services()
        plan = services.planner.plan(state["query"])
        logger.info("Planned query: intent={}, {} sub-queries", plan.intent.value, len(plan.sub_queries))
        return {"plan": plan, "current_step": 0}
    except Exception as exc:  # noqa: BLE001
        logger.error("plan_query failed: {}", exc)
        return {"error": f"planning failed: {exc}"}


def route_sources(state: AgentState) -> dict:
    """No-op passthrough; source selection happens in the edge functions."""
    return {"current_step": state.get("current_step", 0)}


# --------------------------------------------------------------------------- #
# Retrieval nodes
# --------------------------------------------------------------------------- #
def retrieve_faers(state: AgentState) -> dict:
    """Retrieve FAERS evidence for the current sub-query (if FAERS is targeted)."""
    if DataSource.FAERS not in selected_sources(state):
        return {}
    try:
        services = get_services()
        result = services.faers.retrieve(
            _query_for(state), drug_name=_first_drug(state), top_k=settings.VECTOR_TOP_K
        )
        evidence = [_to_evidence(c, DataSource.FAERS) for c in result.chunks]
        return {
            "retrieved_evidence": evidence,
            "source_results": {"FAERS": {"result_count": result.result_count}},
        }
    except Exception as exc:  # noqa: BLE001
        logger.error("retrieve_faers failed: {}", exc)
        return {"error": f"FAERS retrieval failed: {exc}"}


def retrieve_pubmed(state: AgentState) -> dict:
    """Retrieve PubMed evidence for the current sub-query (if PubMed is targeted)."""
    if DataSource.PUBMED not in selected_sources(state):
        return {}
    try:
        services = get_services()
        result = services.pubmed.retrieve(
            _query_for(state), drug=_first_drug(state), ae=_first_ae(state),
            top_k=settings.VECTOR_TOP_K,
        )
        evidence = [_to_evidence(c, DataSource.PUBMED) for c in result.chunks]
        return {
            "retrieved_evidence": evidence,
            "source_results": {"PUBMED": {"result_count": result.result_count}},
        }
    except Exception as exc:  # noqa: BLE001
        logger.error("retrieve_pubmed failed: {}", exc)
        return {"error": f"PubMed retrieval failed: {exc}"}


def retrieve_trials(state: AgentState) -> dict:
    """Retrieve clinical-trial evidence for the current sub-query (if targeted)."""
    if DataSource.CLINICAL_TRIALS not in selected_sources(state):
        return {}
    try:
        services = get_services()
        result = services.trials.retrieve(
            _query_for(state), drug=_first_drug(state), top_k=settings.VECTOR_TOP_K
        )
        evidence = [_to_evidence(c, DataSource.CLINICAL_TRIALS) for c in result.chunks]
        return {
            "retrieved_evidence": evidence,
            "source_results": {"CLINICAL_TRIALS": {"result_count": result.result_count}},
        }
    except Exception as exc:  # noqa: BLE001
        logger.error("retrieve_trials failed: {}", exc)
        return {"error": f"trials retrieval failed: {exc}"}


def retrieve_labels(state: AgentState) -> dict:
    """Retrieve drug-label evidence for the current sub-query (if targeted)."""
    if DataSource.DRUG_LABELS not in selected_sources(state):
        return {}
    try:
        services = get_services()
        result = services.labels.retrieve(
            _query_for(state), drug_name=_first_drug(state), top_k=settings.VECTOR_TOP_K
        )
        evidence = [_to_evidence(c, DataSource.DRUG_LABELS) for c in result.chunks]
        return {
            "retrieved_evidence": evidence,
            "source_results": {"DRUG_LABELS": {"result_count": result.result_count}},
        }
    except Exception as exc:  # noqa: BLE001
        logger.error("retrieve_labels failed: {}", exc)
        return {"error": f"label retrieval failed: {exc}"}


def retrieve_graph(state: AgentState) -> dict:
    """Add graph-derived safety-profile evidence for the query's drugs.

    Runs only on the first pass (``current_step == 0``) and only when the plan
    identified drugs, since the profile is a stable, drug-centric aggregate.
    """
    plan: QueryPlan | None = state.get("plan")
    drugs = plan.drugs_mentioned if plan else []
    if not drugs or state.get("current_step", 0) != 0:
        return {}
    try:
        services = get_services()
        evidence: list[Evidence] = []
        for drug in drugs[:3]:
            profile = services.graph.get_safety_profile(drug)
            evidence.append(
                Evidence(
                    source=DataSource.DRUG_LABELS,
                    content=profile.evidence_summary,
                    metadata={
                        "record_id": f"graph:{drug}",
                        "drug": drug,
                        "graph_derived": True,
                        "total_faers_reports": profile.total_faers_reports,
                        "active_trials": profile.active_trials,
                    },
                    relevance_score=1.0,
                    source_authority=_AUTHORITY[DataSource.DRUG_LABELS],
                )
            )
        return {
            "retrieved_evidence": evidence,
            "source_results": {"GRAPH": {"drugs": drugs[:3]}},
        }
    except Exception as exc:  # noqa: BLE001
        logger.error("retrieve_graph failed: {}", exc)
        return {"error": f"graph retrieval failed: {exc}"}


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def _detect_contradictions(evidence: list[Evidence]) -> list[dict]:
    """Heuristically flag label-vs-FAERS tension (best-effort, demo-grade).

    Flags a contradiction when a label reassures ("safe"/"well tolerated") yet
    spontaneous FAERS reports exist for the same query.
    """
    labels = [e for e in evidence if e.source == DataSource.DRUG_LABELS]
    faers = [e for e in evidence if e.source == DataSource.FAERS]
    if not labels or not faers:
        return []
    reassuring = [
        e for e in labels
        if any(kw in e.content.lower() for kw in ("well tolerated", "generally safe", "no significant"))
    ]
    if reassuring:
        return [
            {
                "type": "label_vs_faers",
                "detail": "Label language is reassuring while FAERS reports exist for the query.",
                "faers_pieces": len(faers),
            }
        ]
    return []


def evaluate_evidence(state: AgentState) -> dict:
    """Decide whether evidence suffices; advance the iteration/step counters."""
    try:
        evidence = state.get("retrieved_evidence", [])
        sources = {e.source for e in evidence}
        iteration = state.get("iteration", 0) + 1
        max_iters = state.get("max_iterations", settings.AGENT_MAX_ITERATIONS)

        sufficient = (
            len(evidence) >= settings.AGENT_MIN_EVIDENCE_PIECES and len(sources) >= 2
        )
        if iteration >= max_iters:
            sufficient = True  # stop looping regardless

        contradictions = _detect_contradictions(evidence)
        logger.info(
            "evaluate: {} pieces across {} sources -> sufficient={} (iter {}/{})",
            len(evidence), len(sources), sufficient, iteration, max_iters,
        )
        return {
            "is_sufficient": sufficient,
            "iteration": iteration,
            "contradictions": contradictions,
            "current_step": state.get("current_step", 0) + 1,
        }
    except Exception as exc:  # noqa: BLE001
        logger.error("evaluate_evidence failed: {}", exc)
        return {"error": f"evaluation failed: {exc}", "is_sufficient": True}


# --------------------------------------------------------------------------- #
# Synthesis
# --------------------------------------------------------------------------- #
def _build_citations(evidence: list[Evidence], grade: EvidenceGrade) -> list[Citation]:
    """Build one citation per distinct (source, record) evidence piece."""
    citations: list[Citation] = []
    seen: set[tuple[str, str]] = set()
    for e in evidence:
        record_id = str(e.metadata.get("record_id") or e.metadata.get("pmid") or "unknown")
        key = (e.source.value, record_id)
        if key in seen:
            continue
        seen.add(key)
        title = str(e.metadata.get("title") or e.metadata.get("drug") or e.source.value)
        snippet = e.content[:200]
        citations.append(
            Citation(
                source=e.source,
                reference_id=record_id,
                title=title,
                snippet=snippet,
                evidence_grade=grade,
            )
        )
    return citations


def _offline_synthesis(state: AgentState, evidence: list[Evidence], explanation: str) -> str:
    """Compose a deterministic answer when no LLM is available."""
    lines = [
        f"Query: {state.get('query', '')}",
        "",
        f"Evidence assessment: {explanation}",
        "",
        "Key evidence (top pieces):",
    ]
    for e in evidence[:6]:
        rid = e.metadata.get("record_id", "?")
        lines.append(f"  - [{e.source.value}:{rid}] {e.content[:160]}")
    contradictions = state.get("contradictions", [])
    if contradictions:
        lines.append("")
        lines.append("Contradictions flagged:")
        for c in contradictions:
            lines.append(f"  - {c.get('detail', c)}")
    return "\n".join(lines)


def synthesize_answer(state: AgentState) -> dict:
    """Grade the evidence and synthesize the final, cited answer."""
    try:
        services = get_services()
        evidence = state.get("retrieved_evidence", [])
        grade, explanation = services.grader.grade_with_explanation(evidence, state["query"])
        citations = _build_citations(evidence, grade)

        if services.llm is not None and evidence:
            context = "\n".join(
                f"[Source: {e.source.value}:{e.metadata.get('record_id', '?')}] {e.content[:400]}"
                for e in evidence[:20]
            )
            prompt = (
                f"Question: {state['query']}\n\n"
                f"Evidence grade: {grade.value} ({grade.label}). {explanation}\n\n"
                f"Evidence:\n{context}\n\n"
                "Write the analyst answer now."
            )
            try:
                answer = services.llm.generate(prompt, system_prompt=SYNTHESIS_PROMPT_V1)  # type: ignore[attr-defined]
            except Exception as exc:  # noqa: BLE001 - fall back to offline text
                logger.warning("LLM synthesis failed ({}); using offline synthesis", exc)
                answer = _offline_synthesis(state, evidence, explanation)
        else:
            answer = _offline_synthesis(state, evidence, explanation)

        logger.info("Synthesized answer (grade {}, {} citations)", grade.value, len(citations))
        return {"final_answer": answer, "citations": citations, "evidence_grade": grade.value}
    except Exception as exc:  # noqa: BLE001
        logger.error("synthesize_answer failed: {}", exc)
        return {
            "error": f"synthesis failed: {exc}",
            "final_answer": "An error occurred while synthesizing the answer.",
            "evidence_grade": EvidenceGrade.E.value,
        }


def handle_error(state: AgentState) -> dict:
    """Terminal node that surfaces a graceful error message."""
    error = state.get("error", "unknown error")
    logger.error("Agent error path: {}", error)
    return {
        "final_answer": f"An error occurred while processing the query: {error}",
        "error": str(error),
        "evidence_grade": EvidenceGrade.E.value,
    }
