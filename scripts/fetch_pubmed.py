# Requires network access (NCBI E-utilities). Run on personal laptop — blocked on the dev proxy.
"""Fetch PubMed abstracts via NCBI E-utilities (esearch + efetch).

E-utilities ``efetch`` returns XML, so this script maps each article into
MedSignal's simplified PubMed schema and saves an array into
``data/raw/pubmed/pubmed_articles.json`` — the shape ``pubmed_loader`` parses
via its sample-format code path. ``drugs_mentioned`` / ``adverse_events_mentioned``
are left empty here and populated later by the LLM enrichment step at ingest.

Run (personal laptop):
    python scripts/fetch_pubmed.py --limit 500 --term "drug safety adverse event"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests
from loguru import logger
from lxml import etree

from src.config import settings

_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
_BATCH = 200

_STUDY_TYPE_MAP = {
    "meta-analysis": "meta_analysis",
    "review": "review",
    "systematic review": "review",
    "clinical trial": "clinical_trial",
    "randomized controlled trial": "clinical_trial",
    "case reports": "case_report",
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch PubMed abstracts via E-utilities.")
    parser.add_argument("--limit", type=int, default=500, help="Number of abstracts to fetch.")
    parser.add_argument("--term", type=str, default="drug safety adverse event", help="PubMed search term.")
    parser.add_argument("--out", type=str, default=None, help="Output directory (default data/raw/pubmed).")
    parser.add_argument("--api-key", type=str, default=settings.NCBI_API_KEY, help="NCBI API key (optional).")
    return parser.parse_args(argv)


def _search_pmids(term: str, limit: int, api_key: str) -> list[str]:
    """Return up to ``limit`` PMIDs matching a search term."""
    params = {"db": "pubmed", "term": term, "retmax": limit, "retmode": "json"}
    if api_key:
        params["api_key"] = api_key
    response = requests.get(_ESEARCH, params=params, timeout=60)
    response.raise_for_status()
    return response.json().get("esearchresult", {}).get("idlist", [])


def _study_type(pub_types: list[str]) -> str:
    """Map PubMed publication types to our StudyType value."""
    lowered = {pt.lower() for pt in pub_types}
    for key, value in _STUDY_TYPE_MAP.items():
        if key in lowered:
            return value
    return "other"


def _parse_article(node: etree._Element) -> dict | None:
    """Map one <PubmedArticle> XML node into our simplified schema."""
    citation = node.find("MedlineCitation")
    if citation is None:
        return None
    article = citation.find("Article")
    if article is None:
        return None

    pmid = citation.findtext("PMID", default="").strip()
    title = "".join(article.find("ArticleTitle").itertext()).strip() if article.find("ArticleTitle") is not None else ""

    abstract_parts = [
        "".join(a.itertext()).strip()
        for a in article.findall("Abstract/AbstractText")
    ]
    abstract = " ".join(p for p in abstract_parts if p).strip()
    if not abstract:
        return None  # loader skips empty abstracts anyway

    authors = []
    for author in article.findall("AuthorList/Author"):
        last = author.findtext("LastName", default="").strip()
        initials = author.findtext("Initials", default="").strip()
        full = f"{last} {initials}".strip()
        if full:
            authors.append(full)

    journal = article.findtext("Journal/Title", default="").strip()
    year = article.findtext("Journal/JournalIssue/PubDate/Year", default="").strip()
    month = article.findtext("Journal/JournalIssue/PubDate/Month", default="").strip()
    pub_date = f"{year}-{month}" if year and month else (year or None)

    mesh_terms = [
        m.text.strip()
        for m in citation.findall("MeshHeadingList/MeshHeading/DescriptorName")
        if m.text
    ]
    pub_types = [pt.text.strip() for pt in article.findall("PublicationTypeList/PublicationType") if pt.text]

    return {
        "pmid": pmid,
        "title": title,
        "abstract": abstract,
        "authors": authors,
        "journal": journal or "Unknown",
        "pub_date": pub_date,
        "mesh_terms": mesh_terms,
        "drugs_mentioned": [],
        "adverse_events_mentioned": [],
        "study_type": _study_type(pub_types),
        "key_findings": "",
    }


def _fetch_batch(pmids: list[str], api_key: str) -> list[dict]:
    """efetch a batch of PMIDs and parse them into article dicts."""
    params = {"db": "pubmed", "id": ",".join(pmids), "rettype": "abstract", "retmode": "xml"}
    if api_key:
        params["api_key"] = api_key
    response = requests.get(_EFETCH, params=params, timeout=90)
    response.raise_for_status()
    root = etree.fromstring(response.content)
    articles = []
    for node in root.findall("PubmedArticle"):
        parsed = _parse_article(node)
        if parsed:
            articles.append(parsed)
    return articles


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out_dir = Path(args.out) if args.out else settings.data_path / "raw" / "pubmed"
    out_dir.mkdir(parents=True, exist_ok=True)

    pmids = _search_pmids(args.term, args.limit, args.api_key)
    logger.info("esearch returned {} PMIDs", len(pmids))

    articles: list[dict] = []
    for start in range(0, len(pmids), _BATCH):
        chunk = pmids[start : start + _BATCH]
        articles.extend(_fetch_batch(chunk, args.api_key))
        logger.info("Fetched {}/{} abstracts", len(articles), len(pmids))
        time.sleep(0.34)  # E-utilities rate limit (~3 req/s without a key)

    out_path = out_dir / "pubmed_articles.json"
    out_path.write_text(json.dumps(articles, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved {} PubMed articles to {}", len(articles), out_path)
    print(f"Saved {len(articles)} PubMed articles to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
