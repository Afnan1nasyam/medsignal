"""FastAPI dependency providers.

Each store/model/client is created once and cached with ``functools.lru_cache``
so the whole app shares a single instance per process. Construction is lazy —
nothing here runs until a route first depends on it — which keeps app import and
startup cheap and avoids opening stores at import time.
"""

from __future__ import annotations

from functools import lru_cache

from loguru import logger

from src.config import settings
from src.storage.graph_store import GraphStoreProtocol, get_graph_store
from src.storage.sql_store import SQLStore
from src.storage.vector_store import QdrantVectorStore
from src.utils.embedding import BiomedEmbeddingModel, get_embedding_model


@lru_cache(maxsize=1)
def get_vector_store() -> QdrantVectorStore:
    """Return the shared :class:`QdrantVectorStore` (created once)."""
    logger.info("Initializing QdrantVectorStore dependency")
    return QdrantVectorStore()


@lru_cache(maxsize=1)
def get_graph_store_dep() -> GraphStoreProtocol:
    """Return the shared knowledge-graph store (backend per settings)."""
    logger.info("Initializing graph store dependency")
    return get_graph_store()


@lru_cache(maxsize=1)
def get_sql_store() -> SQLStore:
    """Return the shared :class:`SQLStore` (created once)."""
    logger.info("Initializing SQLStore dependency")
    return SQLStore()


@lru_cache(maxsize=1)
def get_embedding_model_dep() -> BiomedEmbeddingModel:
    """Return the shared embedding model wrapper (loads lazily on first embed)."""
    logger.info("Initializing embedding model dependency")
    return get_embedding_model()


@lru_cache(maxsize=1)
def get_llm_client():
    """Return a shared ``GroqClient``, or ``None`` when no API key is configured.

    Returns ``None`` (rather than raising) if ``GROQ_API_KEY`` is empty or the
    ``groq`` package/construction fails, so the API stays usable offline via the
    keyword planner and deterministic synthesis.
    """
    if not settings.GROQ_API_KEY:
        logger.info("GROQ_API_KEY empty; LLM features disabled")
        return None
    try:
        from src.utils.llm_client import GroqClient

        return GroqClient()
    except Exception as exc:  # noqa: BLE001
        logger.warning("GroqClient unavailable ({}); LLM features disabled", exc)
        return None


@lru_cache(maxsize=1)
def get_agent():
    """Build and cache the compiled LangGraph agent.

    The agent's services (retrievers, planner, grader, LLM) are wired from the
    cached dependencies above so the API shares one set of stores.
    """
    from src.agents.graph import build_agent_graph
    from src.agents.nodes import AgentServices, set_services
    from src.retrieval.evidence_grader import EvidenceGrader
    from src.retrieval.faers_retriever import FaersRetriever
    from src.retrieval.graph_retriever import GraphRetriever
    from src.retrieval.label_retriever import LabelRetriever
    from src.retrieval.pubmed_retriever import PubMedRetriever
    from src.retrieval.query_planner import QueryPlanner
    from src.retrieval.reranker import EvidenceReranker
    from src.retrieval.trials_retriever import TrialsRetriever

    vector_store = get_vector_store()
    sql_store = get_sql_store()
    embedding = get_embedding_model_dep()
    graph_store = get_graph_store_dep()
    llm = get_llm_client()

    set_services(
        AgentServices(
            planner=QueryPlanner(llm_client=llm),
            faers=FaersRetriever(vector_store, sql_store, embedding),
            pubmed=PubMedRetriever(vector_store, sql_store, embedding),
            trials=TrialsRetriever(vector_store, sql_store, embedding),
            labels=LabelRetriever(vector_store, sql_store, embedding),
            graph=GraphRetriever(graph_store),
            reranker=EvidenceReranker(),
            grader=EvidenceGrader(),
            llm=llm,
        )
    )
    logger.info("Compiled LangGraph agent dependency")
    return build_agent_graph()
