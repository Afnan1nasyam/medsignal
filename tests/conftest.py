"""Shared pytest fixtures and configuration for the MedSignal test suite.

Provides hand-built sample models for each data source, a temp-directory fixture
that redirects the store paths in ``settings``, and a pre-populated NetworkX
knowledge graph. All fixtures are offline — no Groq API, no model downloads.
"""

from __future__ import annotations

import pytest

from src.config import settings
from src.models.enums import (
    AEFrequency,
    DrugRole,
    InteractionSeverity,
    ReactionOutcome,
    ReporterType,
    StudyType,
    TrialPhase,
    TrialStatus,
)
from src.models.schemas import (
    ClinicalTrialRecord,
    DrugLabel,
    FaersDrug,
    FaersPatient,
    FaersReaction,
    FaersReport,
    LabelAdverseReaction,
    LabelDrugInteraction,
    PubMedArticle,
    TrialAdverseEvent,
    TrialIntervention,
)
from src.storage.graph_store import NetworkXGraphStore


def pytest_configure(config):
    """Register custom markers so ``pytest --strict-markers`` stays clean."""
    config.addinivalue_line(
        "markers",
        'slow: marks tests that need external API access (deselect with -m "not slow")',
    )


# --------------------------------------------------------------------------- #
# Sample models
# --------------------------------------------------------------------------- #
@pytest.fixture
def sample_faers_report() -> FaersReport:
    """A FAERS report: metformin (primary suspect) with lactic acidosis."""
    return FaersReport(
        report_id="TEST-FAERS-001",
        patient=FaersPatient(age=72.0, age_unit="year", sex="female", weight=68.0),
        drugs=[
            FaersDrug(
                name="Glucophage",
                generic_name="metformin",
                role=DrugRole.PRIMARY_SUSPECT,
                indication="Type 2 diabetes mellitus",
            )
        ],
        reactions=[
            FaersReaction(preferred_term="Lactic acidosis", outcome=ReactionOutcome.HOSPITALIZATION)
        ],
        report_date="2024-05-01",
        reporter_type=ReporterType.PHYSICIAN,
        serious=True,
    )


@pytest.fixture
def sample_pubmed_article() -> PubMedArticle:
    """A PubMed article on metformin safety with a long, multi-sentence abstract."""
    abstract = (
        "Metformin remains the first-line pharmacotherapy for type 2 diabetes mellitus. "
        "Despite its favourable profile, metformin-associated lactic acidosis is a rare "
        "but serious adverse event. In this review we synthesise a decade of spontaneous "
        "adverse-event reports and observational cohorts. Reports of lactic acidosis "
        "clustered strongly in patients with acute kidney injury and dehydration. "
        "Concomitant nephrotoxic drugs further elevated the observed risk. "
        "Case-fatality estimates ranged widely across the included studies. "
        "We recommend renal monitoring and adherence to sick-day rules. "
        "Clinicians should withhold metformin during acute intercurrent illness. "
        "Overall the benefit-risk balance for metformin remains favourable when contraindications are respected."
    )
    return PubMedArticle(
        pmid="TEST-PMID-1",
        title="Metformin-Associated Lactic Acidosis: A Review",
        abstract=abstract,
        authors=["Doe J", "Smith A"],
        journal="Drug Safety",
        pub_date="2023-01-15",
        mesh_terms=["Metformin", "Acidosis, Lactic"],
        drugs_mentioned=["metformin"],
        adverse_events_mentioned=["lactic acidosis"],
        study_type=StudyType.REVIEW,
        key_findings="Rare but serious; renal function is the key modifier.",
    )


@pytest.fixture
def sample_trial() -> ClinicalTrialRecord:
    """A completed phase-3 metformin trial with adverse-event results."""
    return ClinicalTrialRecord(
        nct_id="NCT00000001",
        title="Metformin for Glycaemic Control in Type 2 Diabetes",
        phase=TrialPhase.PHASE_3,
        status=TrialStatus.COMPLETED,
        conditions=["Type 2 Diabetes Mellitus"],
        interventions=[TrialIntervention(name="metformin", type="drug")],
        adverse_events=[
            TrialAdverseEvent(
                term="Diarrhoea",
                organ_system="Gastrointestinal disorders",
                affected_count=42,
                at_risk_count=300,
                frequency_percent=14.0,
            )
        ],
        enrollment=300,
        start_date="2019-03-01",
        completion_date="2021-06-30",
    )


@pytest.fixture
def sample_drug_label() -> DrugLabel:
    """A metformin drug label with five populated sections (no boxed warning)."""
    return DrugLabel(
        drug_name="Glucophage",
        generic_name="metformin",
        active_ingredient="metformin hydrochloride",
        manufacturer="Test Pharma",
        indications=["Type 2 diabetes mellitus"],
        contraindications=["Severe renal impairment (eGFR < 30)", "Metabolic acidosis"],
        warnings=["Risk of lactic acidosis", "Vitamin B12 deficiency with long-term use"],
        adverse_reactions=[
            LabelAdverseReaction(reaction="Diarrhoea", frequency=AEFrequency.COMMON),
            LabelAdverseReaction(
                reaction="Lactic acidosis", frequency=AEFrequency.VERY_RARE, description="Serious"
            ),
        ],
        drug_interactions=[
            LabelDrugInteraction(
                interacting_drug="ibuprofen",
                severity=InteractionSeverity.MODERATE,
                description="Reduced renal clearance may raise risk.",
            )
        ],
        boxed_warning=None,
    )


# --------------------------------------------------------------------------- #
# Temp store paths
# --------------------------------------------------------------------------- #
@pytest.fixture
def tmp_data_dir(tmp_path, monkeypatch):
    """Redirect all store paths in ``settings`` to a temp directory."""
    monkeypatch.setattr(settings, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(settings, "QDRANT_PATH", str(tmp_path / "qdrant"))
    monkeypatch.setattr(settings, "GRAPH_PATH", str(tmp_path / "knowledge_graph.json"))
    monkeypatch.setattr(settings, "SQLITE_PATH", str(tmp_path / "medsignal.db"))
    return tmp_path


# --------------------------------------------------------------------------- #
# Pre-populated knowledge graph
# --------------------------------------------------------------------------- #
@pytest.fixture
def sample_graph(tmp_data_dir) -> NetworkXGraphStore:
    """A NetworkX graph with 3 drugs, 5 adverse events, and typed edges.

    metformin (biguanide) with lactic acidosis + diarrhoea, one trial, one
    publication, one contraindication, and an interaction with simvastatin;
    simvastatin & atorvastatin (statins) sharing a class edge and reporting
    rhabdomyolysis / myalgia.
    """
    graph = NetworkXGraphStore(auto_load=False)

    graph.add_drug("metformin", {"drug_class": "biguanides"})
    graph.add_drug("simvastatin", {"drug_class": "statins"})
    graph.add_drug("atorvastatin", {"drug_class": "statins"})

    for term in ["lactic acidosis", "rhabdomyolysis", "myalgia", "nausea", "diarrhoea"]:
        graph.add_adverse_event(term, {})

    graph.add_condition("Severe renal impairment", {})
    graph.add_clinical_trial("NCT00000001", {"title": "Metformin trial"})
    graph.add_publication("TEST-PMID-1", {"title": "Metformin review"})

    # REPORTED_WITH edges (with FAERS report counts)
    graph.add_edge("drug:metformin", "ae:lactic acidosis", "REPORTED_WITH", {"report_count": 9, "source": "FAERS"})
    graph.add_edge("drug:metformin", "ae:diarrhoea", "REPORTED_WITH", {"report_count": 5, "source": "FAERS"})
    graph.add_edge("drug:simvastatin", "ae:rhabdomyolysis", "REPORTED_WITH", {"report_count": 6, "source": "FAERS"})
    graph.add_edge("drug:simvastatin", "ae:myalgia", "REPORTED_WITH", {"report_count": 10, "source": "FAERS"})
    graph.add_edge("drug:atorvastatin", "ae:myalgia", "REPORTED_WITH", {"report_count": 4, "source": "FAERS"})

    # class / interaction / context edges
    graph.add_edge("drug:simvastatin", "drug:atorvastatin", "SAME_CLASS_AS", {"drug_class": "statins"})
    graph.add_edge("drug:metformin", "drug:simvastatin", "INTERACTS_WITH", {"severity": "moderate"})
    graph.add_edge("drug:metformin", "condition:severe renal impairment", "CONTRAINDICATED_FOR", {})
    graph.add_edge("drug:metformin", "trial:NCT00000001", "STUDIED_IN", {})
    graph.add_edge("pub:TEST-PMID-1", "drug:metformin", "DESCRIBES", {})

    return graph
