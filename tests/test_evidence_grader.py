"""Tests for the evidence grader (A-E rubric)."""

from __future__ import annotations

from src.models.enums import DataSource, EvidenceGrade
from src.models.schemas import Evidence
from src.retrieval.evidence_grader import EvidenceGrader

GRADER = EvidenceGrader()


def _ev(source: DataSource, **metadata) -> Evidence:
    return Evidence(source=source, content="x", metadata=metadata, relevance_score=0.5, source_authority=1)


def test_grade_a():
    """Label confirmation plus trial support grades A."""
    evidence = [_ev(DataSource.DRUG_LABELS), _ev(DataSource.CLINICAL_TRIALS)]
    assert GRADER.grade(evidence, "q") == EvidenceGrade.A


def test_grade_b():
    """50+ FAERS reports plus 2 PubMed articles grades B."""
    evidence = [
        _ev(DataSource.FAERS, report_count=60),
        _ev(DataSource.PUBMED, record_id="p1"),
        _ev(DataSource.PUBMED, record_id="p2"),
    ]
    assert GRADER.grade(evidence, "q") == EvidenceGrade.B


def test_grade_c():
    """Literature plus a moderate (10+) FAERS signal grades C."""
    evidence = [_ev(DataSource.FAERS, report_count=15), _ev(DataSource.PUBMED, record_id="p1")]
    assert GRADER.grade(evidence, "q") == EvidenceGrade.C


def test_grade_d():
    """A few isolated FAERS reports grade D."""
    evidence = [_ev(DataSource.FAERS, report_count=3)]
    assert GRADER.grade(evidence, "q") == EvidenceGrade.D


def test_grade_e():
    """No evidence grades E."""
    assert GRADER.grade([], "q") == EvidenceGrade.E


def test_grade_with_explanation():
    """The explanation variant returns a grade and a non-empty rationale."""
    grade, explanation = GRADER.grade_with_explanation([_ev(DataSource.FAERS, report_count=3)], "q")
    assert grade == EvidenceGrade.D
    assert isinstance(explanation, str) and explanation
