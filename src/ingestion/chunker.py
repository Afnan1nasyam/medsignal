"""Source-aware chunking of validated records into text + metadata.

Each data source has a distinct chunking strategy that turns a structured
record into one or more natural-language chunks suitable for embedding, paired
with :class:`ChunkMetadata` provenance. All functions are pure Python (offline).
"""

import re

from src.config import settings
from src.models.enums import DataSource
from src.models.schemas import (
    ClinicalTrialRecord,
    DrugLabel,
    FaersReport,
    PubMedArticle,
)
from src.models.schemas import ChunkMetadata

# A chunk is the text plus its provenance metadata.
Chunk = tuple[str, ChunkMetadata]


def split_sentences(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping, sentence-aligned chunks.

    Sentences are greedily packed into chunks up to ``chunk_size`` characters.
    Each subsequent chunk is seeded with trailing whole sentences from the
    previous chunk totalling up to ``overlap`` characters, preserving context
    across boundaries. A single sentence longer than ``chunk_size`` becomes its
    own chunk rather than being split mid-sentence.

    Args:
        text: The text to chunk.
        chunk_size: Maximum chunk length in characters.
        overlap: Approximate character overlap between consecutive chunks.

    Returns:
        A list of chunk strings (empty if ``text`` is blank).
    """
    text = (text or "").strip()
    if not text:
        return []

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        addition = len(sentence) + (1 if current else 0)
        if current and current_len + addition > chunk_size:
            chunks.append(" ".join(current))
            # Seed the next chunk with trailing sentences up to `overlap` chars.
            carry: list[str] = []
            carry_len = 0
            for prev in reversed(current):
                if carry_len + len(prev) > overlap:
                    break
                carry.insert(0, prev)
                carry_len += len(prev) + 1
            current = carry
            current_len = sum(len(s) + 1 for s in current)

        current.append(sentence)
        current_len += len(sentence) + 1

    if current:
        chunks.append(" ".join(current))
    return chunks


def _meta(source: DataSource, record_id: str, index: int) -> ChunkMetadata:
    """Build chunk metadata for a source/record/index triple."""
    return ChunkMetadata(source=source, record_id=record_id, chunk_index=index)


def chunk_faers_report(report: FaersReport) -> list[Chunk]:
    """Chunk a FAERS report into a single natural-language narrative.

    FAERS reports are short, so each yields exactly one chunk.
    """
    patient = report.patient
    age_part = "Unknown-age"
    if patient.age is not None:
        unit = patient.age_unit or "year"
        age_part = f"{patient.age:g}-{unit}-old"
    patient_str = f"{age_part} {patient.sex} patient"

    if report.drugs:
        drug_bits = []
        for drug in report.drugs:
            bit = f"{drug.name} ({drug.role.value.replace('_', ' ')})"
            if drug.indication:
                bit += f" for {drug.indication}"
            drug_bits.append(bit)
        drugs_str = "Drugs: " + ", ".join(drug_bits) + "."
    else:
        drugs_str = "Drugs: none recorded."

    if report.reactions:
        reaction_bits = [
            f"{r.preferred_term} ({r.outcome.value.replace('_', ' ')})"
            for r in report.reactions
        ]
        reactions_str = "Reactions: " + ", ".join(reaction_bits) + "."
    else:
        reactions_str = "Reactions: none recorded."

    serious_str = "Serious report." if report.serious else "Non-serious report."
    date_str = f"Report date: {report.report_date}." if report.report_date else ""

    narrative = " ".join(
        part
        for part in [
            f"Patient: {patient_str}.",
            drugs_str,
            reactions_str,
            serious_str,
            f"Reporter: {report.reporter_type.value}.",
            date_str,
        ]
        if part
    )
    return [(narrative, _meta(DataSource.FAERS, report.report_id, 0))]


def chunk_pubmed_article(article: PubMedArticle) -> list[Chunk]:
    """Chunk a PubMed article, splitting long abstracts by sentence with overlap.

    Each chunk is prefixed with the title/journal header so it is
    self-contained for retrieval.
    """
    header = (
        f"[Title] {article.title}\n"
        f"[Journal] {article.journal} ({article.pub_date or 'n.d.'})\n"
        f"[Abstract] "
    )
    abstract = article.abstract.strip()

    if len(abstract) <= settings.CHUNK_SIZE:
        return [(header + abstract, _meta(DataSource.PUBMED, article.pmid, 0))]

    pieces = split_sentences(abstract, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
    return [
        (header + piece, _meta(DataSource.PUBMED, article.pmid, i))
        for i, piece in enumerate(pieces)
    ]


def chunk_clinical_trial(trial: ClinicalTrialRecord) -> list[Chunk]:
    """Chunk a clinical trial, splitting off adverse events if the record is long.

    The first chunk holds the trial overview plus as many adverse-event lines
    as fit; remaining adverse events spill into further chunks, each prefixed
    with the trial title for context.
    """
    interventions = ", ".join(
        f"{iv.name} ({iv.type})" for iv in trial.interventions
    ) or "none listed"
    conditions = ", ".join(trial.conditions) or "none listed"

    header = (
        f"Clinical Trial {trial.nct_id}: {trial.title}. "
        f"Phase: {trial.phase.value}. Status: {trial.status.value}. "
        f"Conditions: {conditions}. Interventions: {interventions}."
    )

    ae_lines = [
        f"Term: {ae.term}, Frequency: {ae.frequency_percent:g}% "
        f"({ae.affected_count}/{ae.at_risk_count})."
        for ae in trial.adverse_events
    ]

    if not ae_lines:
        return [(header, _meta(DataSource.CLINICAL_TRIALS, trial.nct_id, 0))]

    chunks: list[Chunk] = []
    index = 0
    current = header + " Adverse events: "
    for line in ae_lines:
        if len(current) + len(line) + 1 > settings.CHUNK_SIZE and current.strip():
            chunks.append(
                (current.strip(), _meta(DataSource.CLINICAL_TRIALS, trial.nct_id, index))
            )
            index += 1
            current = f"Clinical Trial {trial.nct_id} adverse events (cont.): "
        current += line + " "

    if current.strip():
        chunks.append(
            (current.strip(), _meta(DataSource.CLINICAL_TRIALS, trial.nct_id, index))
        )
    return chunks


def chunk_drug_label(label: DrugLabel) -> list[Chunk]:
    """Chunk a drug label into one chunk per major, non-empty section.

    Each section chunk is prefixed with the drug name for standalone context.
    """
    name = label.drug_name
    sections: list[str] = []

    if label.indications:
        sections.append("Indications: " + "; ".join(label.indications) + ".")
    if label.contraindications:
        sections.append("Contraindications: " + "; ".join(label.contraindications) + ".")
    if label.warnings:
        sections.append("Warnings: " + "; ".join(label.warnings) + ".")
    if label.adverse_reactions:
        ar = "; ".join(
            f"{r.reaction} ({r.frequency.value})"
            + (f": {r.description}" if r.description else "")
            for r in label.adverse_reactions
        )
        sections.append("Adverse reactions: " + ar + ".")
    if label.drug_interactions:
        di = "; ".join(
            f"{d.interacting_drug} ({d.severity.value})"
            + (f": {d.description}" if d.description else "")
            for d in label.drug_interactions
        )
        sections.append("Drug interactions: " + di + ".")
    if label.boxed_warning:
        sections.append("Boxed warning: " + label.boxed_warning + ".")

    if not sections:
        sections = [f"Drug label for {name} (no sections available)."]

    return [
        (f"[{name}] {section}", _meta(DataSource.DRUG_LABELS, name, i))
        for i, section in enumerate(sections)
    ]
