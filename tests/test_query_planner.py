"""Tests for the offline keyword query planner."""

from __future__ import annotations

from src.models.enums import DataSource, QueryIntent
from src.retrieval.query_planner import QueryPlanner

PLANNER = QueryPlanner(llm_client=None)


def test_drug_detection():
    """A drug name is extracted from the query."""
    plan = PLANNER.plan_without_llm("metformin safety")
    assert "metformin" in plan.drugs_mentioned


def test_ae_detection():
    """Both an adverse event and a drug are detected."""
    plan = PLANNER.plan_without_llm("nausea from aspirin")
    assert "nausea" in plan.adverse_events_mentioned
    assert "aspirin" in plan.drugs_mentioned


def test_interaction_intent():
    """Interaction phrasing classifies as DRUG_INTERACTION."""
    plan = PLANNER.plan_without_llm("warfarin and aspirin interaction")
    assert plan.intent == QueryIntent.DRUG_INTERACTION


def test_comparison_intent():
    """Comparison phrasing classifies as DRUG_COMPARISON."""
    plan = PLANNER.plan_without_llm("compare atorvastatin vs simvastatin")
    assert plan.intent == QueryIntent.DRUG_COMPARISON


def test_safety_intent():
    """Safety-profile phrasing classifies as SAFETY_PROFILE."""
    plan = PLANNER.plan_without_llm("safety profile of metformin")
    assert plan.intent == QueryIntent.SAFETY_PROFILE


def test_side_effects_is_safety_profile():
    """Generic 'side effects' phrasing classifies as SAFETY_PROFILE."""
    plan = PLANNER.plan_without_llm("What are the common side effects of metformin?")
    assert plan.intent == QueryIntent.SAFETY_PROFILE


def test_contraindications_is_safety_profile():
    """'contraindications' classifies as SAFETY_PROFILE."""
    plan = PLANNER.plan_without_llm("What are the contraindications for lisinopril?")
    assert plan.intent == QueryIntent.SAFETY_PROFILE


def test_boxed_warning_is_safety_profile():
    """'black box warning' classifies as SAFETY_PROFILE."""
    plan = PLANNER.plan_without_llm("Does amiodarone have a black box warning?")
    assert plan.intent == QueryIntent.SAFETY_PROFILE


def test_two_drug_fallback_is_interaction():
    """Two drugs with no explicit keyword falls back to DRUG_INTERACTION."""
    plan = PLANNER.plan_without_llm(
        "What should a pharmacist check before dispensing metformin to a patient already taking ibuprofen?"
    )
    assert set(plan.drugs_mentioned) >= {"metformin", "ibuprofen"}
    assert plan.intent == QueryIntent.DRUG_INTERACTION


def test_disproportionate_is_signal_detection():
    """'disproportionately reported' classifies as SIGNAL_DETECTION."""
    plan = PLANNER.plan_without_llm(
        "Which adverse events are disproportionately reported for gabapentin?"
    )
    assert plan.intent == QueryIntent.SIGNAL_DETECTION


def test_gabapentin_detected():
    """gabapentin (now in medical_terms) is detected as a drug."""
    plan = PLANNER.plan_without_llm("safety of gabapentin")
    assert "gabapentin" in plan.drugs_mentioned


def test_sub_query_generation():
    """A drug query targets the label + FAERS sources first, then literature."""
    plan = PLANNER.plan_without_llm("metformin safety")
    assert 1 <= len(plan.sub_queries) <= 4
    first_sources = set(plan.sub_queries[0].target_sources)
    assert DataSource.DRUG_LABELS in first_sources
    assert DataSource.FAERS in first_sources
    all_sources = {s for sq in plan.sub_queries for s in sq.target_sources}
    assert DataSource.PUBMED in all_sources
