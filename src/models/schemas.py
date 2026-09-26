"""Pydantic v2 data models for MedSignal.

Part 1 — the four data-source models (FAERS, PubMed, ClinicalTrials.gov, drug
labels), mirroring the extraction schemas in ``ARCHITECTURE.md``. All schemas
strip surrounding whitespace on string fields so that messy source data is
normalized on the way in.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from src.models.enums import (
    AEFrequency,
    DataSource,
    DrugRole,
    EdgeType,
    EvidenceGrade,
    InteractionSeverity,
    NodeType,
    QueryIntent,
    ReactionOutcome,
    ReporterType,
    StudyType,
    TrialPhase,
    TrialStatus,
)

# Applied to every model: trim whitespace on all string fields.
_STRICT_STR = ConfigDict(str_strip_whitespace=True)


# --------------------------------------------------------------------------- #
# FAERS — FDA Adverse Event Reporting System
# --------------------------------------------------------------------------- #
class FaersPatient(BaseModel):
    """Patient demographics attached to a FAERS report."""

    model_config = _STRICT_STR

    age: float | None = None
    age_unit: str | None = None
    sex: Literal["male", "female", "unknown"] = "unknown"
    weight: float | None = None


class FaersDrug(BaseModel):
    """A drug named in a FAERS report, with its suspected role."""

    model_config = _STRICT_STR

    name: str
    generic_name: str | None = None
    role: DrugRole
    indication: str | None = None
    dosage: str | None = None
    route: str | None = None


class FaersReaction(BaseModel):
    """A single adverse reaction (MedDRA preferred term) and its outcome."""

    model_config = _STRICT_STR

    preferred_term: str
    outcome: ReactionOutcome


class FaersReport(BaseModel):
    """A structured FAERS individual case safety report."""

    model_config = _STRICT_STR

    report_id: str
    patient: FaersPatient
    drugs: list[FaersDrug] = Field(default_factory=list)
    reactions: list[FaersReaction] = Field(default_factory=list)
    report_date: str | None = None
    reporter_type: ReporterType = ReporterType.OTHER
    serious: bool = False


# --------------------------------------------------------------------------- #
# PubMed — literature abstracts
# --------------------------------------------------------------------------- #
class PubMedArticle(BaseModel):
    """A PubMed abstract with LLM-extracted drug/adverse-event mentions."""

    model_config = _STRICT_STR

    pmid: str
    title: str
    abstract: str
    authors: list[str] = Field(default_factory=list)
    journal: str
    pub_date: str | None = None
    mesh_terms: list[str] = Field(default_factory=list)
    drugs_mentioned: list[str] = Field(default_factory=list)
    adverse_events_mentioned: list[str] = Field(default_factory=list)
    study_type: StudyType = StudyType.OTHER
    key_findings: str = ""


# --------------------------------------------------------------------------- #
# ClinicalTrials.gov — trial records
# --------------------------------------------------------------------------- #
class TrialIntervention(BaseModel):
    """An intervention (arm) within a clinical trial."""

    model_config = _STRICT_STR

    name: str
    type: Literal["drug", "biological", "other"] = "other"


class TrialAdverseEvent(BaseModel):
    """An adverse event reported within a clinical trial arm."""

    model_config = _STRICT_STR

    term: str
    organ_system: str
    affected_count: int
    at_risk_count: int
    frequency_percent: float


class ClinicalTrialRecord(BaseModel):
    """A structured ClinicalTrials.gov study record."""

    model_config = _STRICT_STR

    nct_id: str
    title: str
    phase: TrialPhase
    status: TrialStatus
    conditions: list[str] = Field(default_factory=list)
    interventions: list[TrialIntervention] = Field(default_factory=list)
    adverse_events: list[TrialAdverseEvent] = Field(default_factory=list)
    enrollment: int | None = None
    start_date: str | None = None
    completion_date: str | None = None


# --------------------------------------------------------------------------- #
# Drug labels — DailyMed / openFDA labeling
# --------------------------------------------------------------------------- #
class LabelAdverseReaction(BaseModel):
    """An adverse reaction listed on a drug label, with stated frequency."""

    model_config = _STRICT_STR

    reaction: str
    frequency: AEFrequency
    description: str = ""


class LabelDrugInteraction(BaseModel):
    """A drug-drug interaction described on a drug label."""

    model_config = _STRICT_STR

    interacting_drug: str
    severity: InteractionSeverity
    description: str = ""


class DrugLabel(BaseModel):
    """A structured drug label (prescribing information)."""

    model_config = _STRICT_STR

    drug_name: str
    generic_name: str
    active_ingredient: str
    manufacturer: str | None = None
    indications: list[str] = Field(default_factory=list)
    contraindications: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    adverse_reactions: list[LabelAdverseReaction] = Field(default_factory=list)
    drug_interactions: list[LabelDrugInteraction] = Field(default_factory=list)
    boxed_warning: str | None = None


# --------------------------------------------------------------------------- #
# Agent & query planning
# --------------------------------------------------------------------------- #
class SubQuery(BaseModel):
    """A decomposed sub-query targeting one or more data sources."""

    model_config = _STRICT_STR

    query: str
    target_sources: list[DataSource] = Field(default_factory=list)
    reasoning: str = ""


class QueryPlan(BaseModel):
    """The planner's decomposition of a user query into sub-queries."""

    model_config = _STRICT_STR

    original_query: str
    intent: QueryIntent = QueryIntent.GENERAL
    sub_queries: list[SubQuery] = Field(default_factory=list)
    drugs_mentioned: list[str] = Field(default_factory=list)
    adverse_events_mentioned: list[str] = Field(default_factory=list)


class Evidence(BaseModel):
    """A single piece of retrieved evidence, scored and authority-ranked.

    ``source_authority`` is a 1-5 rank of source trustworthiness:
    5 = drug label, 4 = clinical trial, 3 = systematic review,
    2 = case report, 1 = individual FAERS report.
    """

    model_config = _STRICT_STR

    source: DataSource
    content: str
    metadata: dict = Field(default_factory=dict)
    relevance_score: float = 0.0
    source_authority: int = 0


class Citation(BaseModel):
    """A graded citation attached to a synthesized answer."""

    model_config = _STRICT_STR

    source: DataSource
    reference_id: str
    title: str
    snippet: str
    evidence_grade: EvidenceGrade
    url: str | None = None


class AgentResult(BaseModel):
    """The final synthesized answer produced by the agent."""

    model_config = _STRICT_STR

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    evidence_grade: EvidenceGrade
    sources_consulted: list[DataSource] = Field(default_factory=list)
    iterations_used: int
    query_plan: QueryPlan | None = None


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #
class ChunkMetadata(BaseModel):
    """Provenance metadata carried alongside a stored/retrieved chunk."""

    model_config = _STRICT_STR

    source: DataSource
    record_id: str
    chunk_index: int
    source_file: str | None = None


class RetrievalResult(BaseModel):
    """Raw retrieval output from a single source retriever."""

    model_config = _STRICT_STR

    chunks: list[dict] = Field(default_factory=list)
    source: DataSource
    query: str
    result_count: int = 0


class RerankedResult(BaseModel):
    """A retrieved chunk after cross-source authority-weighted re-ranking."""

    model_config = _STRICT_STR

    content: str
    metadata: ChunkMetadata
    relevance_score: float
    authority_score: int
    combined_score: float


# --------------------------------------------------------------------------- #
# Knowledge graph
# --------------------------------------------------------------------------- #
class GraphNode(BaseModel):
    """A node in the biomedical knowledge graph."""

    model_config = _STRICT_STR

    id: str
    type: NodeType
    properties: dict = Field(default_factory=dict)


class GraphEdge(BaseModel):
    """A directed, typed edge between two knowledge-graph nodes."""

    model_config = _STRICT_STR

    source_id: str
    target_id: str
    type: EdgeType
    properties: dict = Field(default_factory=dict)


class DrugSafetyProfile(BaseModel):
    """Aggregated safety profile for a single drug, assembled across sources."""

    model_config = _STRICT_STR

    drug_name: str
    total_faers_reports: int = 0
    top_adverse_events: list[dict] = Field(default_factory=list)
    known_interactions: list[dict] = Field(default_factory=list)
    contraindications: list[str] = Field(default_factory=list)
    active_trials: int = 0
    related_publications: int = 0
    evidence_summary: str = ""


# --------------------------------------------------------------------------- #
# API request / response
# --------------------------------------------------------------------------- #
class QueryRequest(BaseModel):
    """Request body for the main ``POST /api/query`` endpoint."""

    model_config = _STRICT_STR

    query: str
    max_iterations: int = 3
    sources: list[DataSource] | None = None


class QueryResponse(BaseModel):
    """Response for ``POST /api/query``."""

    model_config = _STRICT_STR

    result: AgentResult
    processing_time_seconds: float


class DrugLookupResponse(BaseModel):
    """Response for a drug lookup, combining profile and graph context."""

    model_config = _STRICT_STR

    profile: DrugSafetyProfile
    graph_neighbors: list[dict] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Response for the health-check endpoint."""

    model_config = _STRICT_STR

    status: str
    stores: dict = Field(default_factory=dict)
    version: str
