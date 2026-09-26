"""Ingestion routes — trigger the pipeline and report store status."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from pydantic import BaseModel

from api.dependencies import get_graph_store_dep, get_sql_store, get_vector_store
from src.config import settings
from src.storage.graph_store import GraphStoreProtocol
from src.storage.sql_store import SQLStore
from src.storage.vector_store import QdrantVectorStore

router = APIRouter(prefix="/api/ingest", tags=["ingest"])

# source key -> pipeline method + source subdirectory
_SOURCE_METHODS: dict[str, tuple[str, str]] = {
    "faers": ("ingest_faers", "faers"),
    "pubmed": ("ingest_pubmed", "pubmed"),
    "trials": ("ingest_clinical_trials", "clinical_trials"),
    "labels": ("ingest_drug_labels", "drug_labels"),
}


class IngestRequest(BaseModel):
    """Body for triggering ingestion."""

    source: str = "all"
    data_dir: str | None = None
    reset: bool = False
    enrich: bool = False


@router.post("")
async def trigger_ingest(request: IngestRequest) -> dict:
    """Run ingestion for one source or all, from the sample data (or a directory).

    Note: ingestion loads the embedding model (and, with ``enrich``, the LLM);
    intended for the personal laptop. Raises 400 for an unknown source.
    """
    if request.source != "all" and request.source not in _SOURCE_METHODS:
        raise HTTPException(status_code=400, detail=f"Unknown source: {request.source}")

    from src.ingestion.pipeline import IngestionPipeline

    pipeline = IngestionPipeline(use_extractor=request.enrich)
    if request.reset:
        pipeline.reset_stores()

    base = Path(request.data_dir) if request.data_dir else Path(settings.DATA_DIR) / "sample"

    try:
        if request.source == "all":
            summary = pipeline.ingest_all(data_dir=base, enrich=request.enrich)
            return {"status": "ok", "summary": summary}
        method_name, subdir = _SOURCE_METHODS[request.source]
        method = getattr(pipeline, method_name)
        count = (
            method(base / subdir, enrich=request.enrich)
            if request.source == "pubmed"
            else method(base / subdir)
        )
        if hasattr(pipeline.graph, "save"):
            pipeline.graph.save()
        return {"status": "ok", "source": request.source, "count": count}
    except Exception as exc:  # noqa: BLE001
        logger.error("Ingestion failed: {}", exc)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc


@router.get("/status")
async def ingest_status(
    vector: QdrantVectorStore = Depends(get_vector_store),
    sql: SQLStore = Depends(get_sql_store),
    graph: GraphStoreProtocol = Depends(get_graph_store_dep),
) -> dict:
    """Return current record/collection/node counts across all stores."""
    graph_stats = graph.get_stats()
    return {
        "sql": sql.get_all_stats(),
        "vector": vector.get_stats(),
        "graph": {
            "total_nodes": graph_stats.get("total_nodes", 0),
            "total_edges": graph_stats.get("total_edges", 0),
            "node_counts": graph_stats.get("node_counts", {}),
            "edge_counts": graph_stats.get("edge_counts", {}),
        },
    }
