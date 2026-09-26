"""Evidence grading — assign an A–E strength grade to retrieved evidence.

Implements the grading rubric from ``ARCHITECTURE.md``: authoritative drug-label
confirmation plus clinical-trial support is strongest (A); isolated spontaneous
reports are weakest (D); no evidence is insufficient (E).
"""

from __future__ import annotations

from loguru import logger

from src.models.enums import DataSource, EvidenceGrade
from src.models.schemas import Evidence

# FAERS report-count thresholds from the rubric.
_MODERATE_FAERS = 50
_SUGGESTIVE_FAERS = 10


class EvidenceGrader:
    """Grades a collection of :class:`Evidence` pieces against the rubric."""

    # -- component signals ------------------------------------------------- #
    def has_label_confirmation(self, evidence: list[Evidence]) -> bool:
        """Whether any evidence piece comes from a drug label."""
        return any(e.source == DataSource.DRUG_LABELS for e in evidence)

    def has_trial_evidence(self, evidence: list[Evidence]) -> bool:
        """Whether any evidence piece comes from a clinical trial."""
        return any(e.source == DataSource.CLINICAL_TRIALS for e in evidence)

    def count_faers_signals(self, evidence: list[Evidence]) -> int:
        """Estimate the total number of FAERS reports behind the evidence.

        Uses an explicit ``report_count`` in metadata when present, else the sum
        of any ``drug_event_counts`` map, else counts the FAERS piece as one
        report.
        """
        total = 0
        for e in evidence:
            if e.source != DataSource.FAERS:
                continue
            report_count = e.metadata.get("report_count")
            event_counts = e.metadata.get("drug_event_counts")
            if isinstance(report_count, (int, float)):
                total += int(report_count)
            elif isinstance(event_counts, dict) and event_counts:
                total += sum(int(v) for v in event_counts.values())
            else:
                total += 1
        return total

    def count_literature_support(self, evidence: list[Evidence]) -> int:
        """Count distinct supporting PubMed articles (by record id, else pieces)."""
        pmids: set[str] = set()
        loose = 0
        for e in evidence:
            if e.source != DataSource.PUBMED:
                continue
            record_id = str(e.metadata.get("record_id") or e.metadata.get("pmid") or "")
            if record_id:
                pmids.add(record_id)
            else:
                loose += 1
        return len(pmids) + loose

    # -- grading ----------------------------------------------------------- #
    def grade(self, evidence_pieces: list[Evidence], query: str) -> EvidenceGrade:
        """Return the :class:`EvidenceGrade` for a set of evidence pieces."""
        grade, _ = self.grade_with_explanation(evidence_pieces, query)
        return grade

    def grade_with_explanation(
        self, evidence_pieces: list[Evidence], query: str
    ) -> tuple[EvidenceGrade, str]:
        """Return the grade plus a human-readable justification.

        Args:
            evidence_pieces: Retrieved evidence to grade.
            query: The originating query (for logging/explanation context).

        Returns:
            A ``(grade, explanation)`` tuple.
        """
        if not evidence_pieces:
            return EvidenceGrade.E, "No evidence was retrieved from any source."

        has_label = self.has_label_confirmation(evidence_pieces)
        has_trial = self.has_trial_evidence(evidence_pieces)
        faers = self.count_faers_signals(evidence_pieces)
        literature = self.count_literature_support(evidence_pieces)

        detail = (
            f"label={'yes' if has_label else 'no'}, "
            f"trial={'yes' if has_trial else 'no'}, "
            f"FAERS reports={faers}, PubMed articles={literature}"
        )

        if has_label and has_trial:
            grade = EvidenceGrade.A
            reason = (
                "Confirmed on an authoritative drug label and supported by "
                "clinical-trial data."
            )
        elif faers >= _MODERATE_FAERS and literature >= 2:
            grade = EvidenceGrade.B
            reason = (
                f"{faers} FAERS reports converge with {literature} supporting "
                "publications."
            )
        elif literature >= 1 and faers >= _SUGGESTIVE_FAERS:
            grade = EvidenceGrade.C
            reason = (
                f"Literature ({literature} article(s)) plus a FAERS signal "
                f"({faers} reports) suggest an association."
            )
        elif faers >= 1 or literature >= 1 or has_label or has_trial:
            grade = EvidenceGrade.D
            reason = (
                "Only limited or isolated evidence "
                f"({faers} FAERS report(s), {literature} article(s)) is available."
            )
        else:
            grade = EvidenceGrade.E
            reason = "No usable evidence from any source."

        explanation = f"Grade {grade.value} ({grade.label}): {reason} [{detail}]"
        logger.info("Evidence graded {} for query: {}", grade.value, query)
        return grade, explanation
