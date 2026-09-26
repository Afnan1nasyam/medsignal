"""Tests for the PubMed loader (offline)."""

from __future__ import annotations

import json

import pytest

from src.config import settings
from src.ingestion.pubmed_loader import load_pubmed_articles
from src.models.schemas import PubMedArticle


def test_load_sample_data():
    """Loading the sample PubMed file returns PubMedArticle objects."""
    sample_dir = settings.sample_pubmed_dir
    if not sample_dir.exists() or not any(sample_dir.glob("*.json")):
        pytest.skip("sample PubMed data not present; run scripts/seed_sample_data.py")
    articles = load_pubmed_articles(sample_dir)
    assert isinstance(articles, list)
    assert len(articles) > 0
    assert all(isinstance(a, PubMedArticle) for a in articles)


def test_empty_abstract_skipped(tmp_path):
    """Articles with an empty abstract are skipped by the loader."""
    payload = [
        {"pmid": "P-EMPTY", "title": "No abstract", "abstract": "", "journal": "J"},
        {"pmid": "P-OK", "title": "Has abstract", "abstract": "Real content here.", "journal": "J"},
    ]
    (tmp_path / "articles.json").write_text(json.dumps(payload), encoding="utf-8")
    articles = load_pubmed_articles(tmp_path)
    pmids = {a.pmid for a in articles}
    assert pmids == {"P-OK"}
