"""Graph routes — statistics and topology queries over the knowledge graph.

Neighbor and shortest-path traversal require the in-process NetworkX backend
(they read the underlying ``DiGraph``); with a server-side backend they return
HTTP 501.
"""

from __future__ import annotations

import networkx as nx
from fastapi import APIRouter, Depends, HTTPException
from loguru import logger

from api.dependencies import get_graph_store_dep
from src.storage.graph_store import GraphStoreProtocol
from src.utils.medical_terms import normalize_drug_name

router = APIRouter(prefix="/api/graph", tags=["graph"])


def _require_networkx(graph: GraphStoreProtocol) -> nx.DiGraph:
    """Return the underlying NetworkX graph, or 501 for other backends."""
    inner = getattr(graph, "graph", None)
    if inner is None:
        raise HTTPException(
            status_code=501,
            detail="Topology traversal is only supported by the NetworkX backend.",
        )
    return inner


@router.get("/stats")
async def graph_stats(graph: GraphStoreProtocol = Depends(get_graph_store_dep)) -> dict:
    """Return knowledge-graph statistics (node/edge counts, top drugs)."""
    return graph.get_stats()


@router.get("/neighbors/{node_id}")
async def graph_neighbors(
    node_id: str, graph: GraphStoreProtocol = Depends(get_graph_store_dep)
) -> dict:
    """Return the nodes directly connected to ``node_id`` (both directions).

    Args:
        node_id: A normalized node id, e.g. ``drug:metformin`` or ``ae:nausea``.
    """
    inner = _require_networkx(graph)
    if node_id not in inner:
        raise HTTPException(status_code=404, detail=f"Node not found: {node_id}")

    neighbors: list[dict] = []
    for successor in inner.successors(node_id):
        neighbors.append(
            {
                "node_id": successor,
                "direction": "out",
                "edge_type": inner.edges[node_id, successor].get("type"),
                "node": dict(inner.nodes[successor]),
            }
        )
    for predecessor in inner.predecessors(node_id):
        neighbors.append(
            {
                "node_id": predecessor,
                "direction": "in",
                "edge_type": inner.edges[predecessor, node_id].get("type"),
                "node": dict(inner.nodes[predecessor]),
            }
        )
    return {
        "node_id": node_id,
        "node": dict(inner.nodes[node_id]),
        "neighbors": neighbors,
        "count": len(neighbors),
    }


@router.get("/path/{drug1}/{drug2}")
async def graph_path(
    drug1: str, drug2: str, graph: GraphStoreProtocol = Depends(get_graph_store_dep)
) -> dict:
    """Return the shortest (undirected) path between two drugs in the graph.

    Args:
        drug1: First drug (brand or generic).
        drug2: Second drug (brand or generic).
    """
    inner = _require_networkx(graph)
    id1 = f"drug:{normalize_drug_name(drug1)}"
    id2 = f"drug:{normalize_drug_name(drug2)}"
    for node in (id1, id2):
        if node not in inner:
            raise HTTPException(status_code=404, detail=f"Drug node not found: {node}")

    undirected = inner.to_undirected(as_view=True)
    try:
        path = nx.shortest_path(undirected, id1, id2)
    except nx.NetworkXNoPath:
        return {"drug1": id1, "drug2": id2, "found": False, "path": [], "length": 0}

    hops = []
    for src_node, dst_node in zip(path, path[1:]):
        edge = inner.get_edge_data(src_node, dst_node) or inner.get_edge_data(dst_node, src_node) or {}
        hops.append({"from": src_node, "to": dst_node, "edge_type": edge.get("type")})
    return {
        "drug1": id1,
        "drug2": id2,
        "found": True,
        "path": path,
        "names": [inner.nodes[n].get("name", n) for n in path],
        "length": len(path) - 1,
        "hops": hops,
    }
