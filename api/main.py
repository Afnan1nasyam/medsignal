"""MedSignal FastAPI application.

Wires the query, drugs, graph, ingest, and analytics routers behind a permissive
(dev) CORS policy. Heavy resources (stores, embedding model, agent) are created
lazily via cached dependencies, never at import or startup, so the app boots
instantly and ``/docs`` renders without any data present.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from api.routes import analytics, drugs, graph, ingest, query
from src.config import settings
from src.models.schemas import HealthResponse

API_VERSION = "0.1.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown hook; stores/models are initialized lazily per request."""
    logger.info("MedSignal API v{} starting (graph backend: {})", API_VERSION, settings.GRAPH_BACKEND)
    yield
    logger.info("MedSignal API shutting down")


app = FastAPI(
    title="MedSignal API",
    version=API_VERSION,
    description=(
        "Multi-source agentic RAG for drug-safety intelligence. Ingests FDA "
        "FAERS, PubMed, ClinicalTrials.gov, and DailyMed labels into a vector "
        "store, SQL store, and biomedical knowledge graph, then answers queries "
        "with a LangGraph agent and A–E evidence grading."
    ),
    lifespan=lifespan,
)

# Permissive CORS for local development / the Streamlit front-end.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query.router)
app.include_router(drugs.router)
app.include_router(graph.router)
app.include_router(ingest.router)
app.include_router(analytics.router)


@app.get("/", response_model=HealthResponse, tags=["health"])
async def health() -> HealthResponse:
    """Health check reporting configured store backends (no heavy init)."""
    return HealthResponse(
        status="ok",
        stores={
            "graph_backend": settings.GRAPH_BACKEND,
            "qdrant_path": settings.QDRANT_PATH,
            "sqlite_path": settings.SQLITE_PATH,
        },
        version=API_VERSION,
    )
