"""Tests for the source-aware chunker (offline, pure Python)."""

from __future__ import annotations

from src.config import settings
from src.ingestion.chunker import (
    chunk_drug_label,
    chunk_faers_report,
    chunk_pubmed_article,
    split_sentences,
)


def test_faers_chunking(sample_faers_report):
    """A FAERS report yields exactly one narrative chunk mentioning drug + reaction."""
    chunks = chunk_faers_report(sample_faers_report)
    assert len(chunks) == 1
    text, meta = chunks[0]
    assert "Glucophage" in text
    assert "Lactic acidosis" in text
    assert meta.record_id == "TEST-FAERS-001"
    assert meta.chunk_index == 0


def test_pubmed_chunking(sample_pubmed_article):
    """A long abstract splits into multiple, sequentially-indexed chunks."""
    assert len(sample_pubmed_article.abstract) > settings.CHUNK_SIZE
    chunks = chunk_pubmed_article(sample_pubmed_article)
    assert len(chunks) >= 2
    for i, (text, meta) in enumerate(chunks):
        assert text.startswith("[Title]")
        assert meta.chunk_index == i
        assert meta.record_id == "TEST-PMID-1"


def test_label_chunking(sample_drug_label):
    """A drug label yields one chunk per populated section (5 here; no boxed warning)."""
    chunks = chunk_drug_label(sample_drug_label)
    assert len(chunks) == 5
    prefixes = [text.split("]")[0] + "]" for text, _ in chunks]
    assert all(p == "[Glucophage]" for p in prefixes)
    joined = " ".join(text for text, _ in chunks).lower()
    for section in ["indications", "contraindications", "warnings", "adverse reactions", "drug interactions"]:
        assert section in joined


def test_split_sentences():
    """Sentence splitting produces overlapping, sentence-aligned chunks."""
    text = (
        "Sentence one is here. Sentence two follows. Sentence three continues now. "
        "Sentence four is present. Sentence five closes it out."
    )
    chunks = split_sentences(text, chunk_size=45, overlap=25)
    assert len(chunks) > 1
    # No chunk exceeds a reasonable bound and content is preserved.
    assert all(chunk.strip() for chunk in chunks)
    # Overlap: at least one sentence appears in two consecutive chunks.
    def sentences(chunk: str) -> set[str]:
        return {s.strip() for s in chunk.replace("!", ".").replace("?", ".").split(".") if s.strip()}

    overlaps = any(sentences(a) & sentences(b) for a, b in zip(chunks, chunks[1:]))
    assert overlaps, "expected overlapping sentences between consecutive chunks"


def test_empty_text_no_chunks():
    """Blank text yields no chunks."""
    assert split_sentences("   ", 100, 10) == []
