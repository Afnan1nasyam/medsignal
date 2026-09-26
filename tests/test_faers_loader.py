"""Tests for the FAERS loader (offline; reads committed sample data)."""

from __future__ import annotations

import pytest

from src.config import settings
from src.ingestion.faers_loader import load_faers_reports
from src.models.schemas import FaersReport


def test_load_sample_data():
    """Loading the sample FAERS directory returns FaersReport objects."""
    sample_dir = settings.sample_faers_dir
    if not sample_dir.exists() or not any(sample_dir.glob("*.json")):
        pytest.skip("sample FAERS data not present; run scripts/seed_sample_data.py")
    reports = load_faers_reports(sample_dir)
    assert isinstance(reports, list)
    assert len(reports) > 0
    assert all(isinstance(r, FaersReport) for r in reports)


def test_load_single_report():
    """Loaded reports carry normalized (lowercase) generic drug names."""
    sample_dir = settings.sample_faers_dir
    if not sample_dir.exists() or not any(sample_dir.glob("*.json")):
        pytest.skip("sample FAERS data not present")
    reports = load_faers_reports(sample_dir)
    drugs = [d for r in reports for d in r.drugs]
    assert drugs, "expected at least one drug across reports"
    for drug in drugs:
        assert drug.generic_name is not None
        assert drug.generic_name == drug.generic_name.lower()


def test_empty_directory(tmp_path):
    """Loading an empty directory returns an empty list (no crash)."""
    assert load_faers_reports(tmp_path) == []
