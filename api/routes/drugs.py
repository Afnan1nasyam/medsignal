"""Drug routes — list drugs and expose per-drug safety intelligence."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger

from api.dependencies import get_graph_store_dep, get_sql_store
from src.models.enums import NodeType
from src.models.schemas import DrugLookupResponse
from src.retrieval.graph_retriever import GraphRetriever
from src.storage.graph_store import GraphStoreProtocol
from src.storage.sql_store import SQLStore
from src.utils.medical_terms import normalize_drug_name

router = APIRouter(prefix="/api/drugs", tags=["drugs"])


def _all_drug_names(graph: GraphStoreProtocol) -> list[str]:
    """List every drug node's display name, falling back to graph stats."""
    inner = getattr(graph, "graph", None)  # NetworkX DiGraph if available
    if inner is not None:
        return sorted(
            {
                data.get("name", node_id.split(":", 1)[-1])
                for node_id, data in inner.nodes(data=True)
                if data.get("type") == NodeType.DRUG.value
            }
        )
    return [d.get("drug_name") for d in graph.get_stats().get("top_connected_drugs", [])]


@router.get("")
async def list_drugs(graph: GraphStoreProtocol = Depends(get_graph_store_dep)) -> dict:
    """List all drugs currently present in the knowledge graph."""
    drugs = _all_drug_names(graph)
    return {"drugs": drugs, "count": len(drugs)}


@router.get("/{drug_name}", response_model=DrugLookupResponse)
async def get_drug(
    drug_name: str,
    graph: GraphStoreProtocol = Depends(get_graph_store_dep),
    sql: SQLStore = Depends(get_sql_store),
) -> DrugLookupResponse:
    """Return a drug's aggregated safety profile (graph) enriched with its label.

    Raises:
        HTTPException: 404 if the drug is unknown to both the graph and SQL.
    """
    generic = normalize_drug_name(drug_name)
    retriever = GraphRetriever(graph)
    profile = retriever.get_safety_profile(drug_name)

    node = graph.get_node(f"drug:{generic}")
    label = sql.get_drug_label(drug_name)
    if label is None:
        matches = sql.search_labels(drug=drug_name)
        label = matches[0] if matches else None

    if node is None and label is None:
        raise HTTPException(status_code=404, detail=f"Unknown drug: {drug_name}")

    # Enrich sparse graph profiles with structured label data from SQL.
    if label is not None:
        if not profile.contraindications and label.get("contraindications"):
            profile.contraindications = list(label["contraindications"])
        if not profile.known_interactions and label.get("drug_interactions"):
            profile.known_interactions = [
                {"drug_name": di.get("interacting_drug"), "severity": di.get("severity")}
                for di in label["drug_interactions"]
            ]

    neighbors = retriever.get_related_drugs(drug_name)
    return DrugLookupResponse(profile=profile, graph_neighbors=neighbors)


@router.get("/{drug_name}/interactions")
async def get_drug_interactions(
    drug_name: str, graph: GraphStoreProtocol = Depends(get_graph_store_dep)
) -> dict:
    """List known interacting drugs for a drug, from the knowledge graph."""
    generic = normalize_drug_name(drug_name)
    try:
        interactions = graph.get_drug_interactions(generic)
    except Exception as exc:  # noqa: BLE001
        logger.error("interactions lookup failed for {}: {}", generic, exc)
        raise HTTPException(status_code=500, detail="interaction lookup failed") from exc
    return {"drug": generic, "interactions": interactions, "count": len(interactions)}


@router.get("/{drug_name}/adverse-events")
async def get_drug_adverse_events(
    drug_name: str, graph: GraphStoreProtocol = Depends(get_graph_store_dep)
) -> dict:
    """List adverse events reported with a drug, with report counts."""
    generic = normalize_drug_name(drug_name)
    try:
        events = graph.get_drug_adverse_events(generic)
    except Exception as exc:  # noqa: BLE001
        logger.error("AE lookup failed for {}: {}", generic, exc)
        raise HTTPException(status_code=500, detail="adverse-event lookup failed") from exc
    return {"drug": generic, "adverse_events": events, "count": len(events)}
