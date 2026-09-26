"""Loader for PubMed abstracts (local JSON files only).

Parses both our simplified sample format (already shaped like
:class:`PubMedArticle`) and PubMed E-utilities JSON (a ``PubmedArticleSet`` of
``MedlineCitation`` records) into validated :class:`PubMedArticle` models.
Articles with empty abstracts are skipped, and malformed records are logged and
skipped.

For real data, ``drugs_mentioned`` / ``adverse_events_mentioned`` are populated
by a separate LLM extraction step; in sample data they are pre-populated.
"""

from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import ValidationError

from src.ingestion._common import extract_items, read_json_documents
from src.models.enums import StudyType
from src.models.schemas import PubMedArticle


def _text(value: Any) -> str:
    """Coerce an XML-to-JSON scalar (str or ``{"#text": ...}``) to a string."""
    if isinstance(value, dict):
        return str(value.get("#text", "")).strip()
    if isinstance(value, list):
        return " ".join(_text(v) for v in value).strip()
    return str(value).strip() if value is not None else ""


def _from_sample(item: dict) -> PubMedArticle:
    """Validate an item already shaped like :class:`PubMedArticle`."""
    return PubMedArticle.model_validate(item)


def _from_eutils(article: dict) -> PubMedArticle:
    """Map a raw E-utilities ``PubmedArticle`` into a :class:`PubMedArticle`."""
    citation = article.get("MedlineCitation", {}) or {}
    art = citation.get("Article", {}) or {}

    abstract_node = (art.get("Abstract", {}) or {}).get("AbstractText", "")
    abstract = _text(abstract_node)

    authors: list[str] = []
    author_list = (art.get("AuthorList", {}) or {}).get("Author", [])
    if isinstance(author_list, dict):
        author_list = [author_list]
    for author in author_list or []:
        last = _text(author.get("LastName", ""))
        initials = _text(author.get("Initials", ""))
        full = f"{last} {initials}".strip()
        if full:
            authors.append(full)

    mesh_terms: list[str] = []
    mesh_list = (citation.get("MeshHeadingList", {}) or {}).get("MeshHeading", [])
    if isinstance(mesh_list, dict):
        mesh_list = [mesh_list]
    for heading in mesh_list or []:
        term = _text(heading.get("DescriptorName", ""))
        if term:
            mesh_terms.append(term)

    return PubMedArticle(
        pmid=_text(citation.get("PMID", "")),
        title=_text(art.get("ArticleTitle", "")),
        abstract=abstract,
        authors=authors,
        journal=_text((art.get("Journal", {}) or {}).get("Title", "")),
        pub_date=None,
        mesh_terms=mesh_terms,
        study_type=StudyType.OTHER,
    )


def load_pubmed_articles(data_dir: Path) -> list[PubMedArticle]:
    """Load and validate all PubMed articles from a directory of JSON files.

    Args:
        data_dir: Directory containing PubMed ``*.json`` files (sample or
            E-utilities format).

    Returns:
        A list of validated :class:`PubMedArticle` objects with non-empty
        abstracts (empty list if none).
    """
    articles: list[PubMedArticle] = []
    for filename, document in read_json_documents(Path(data_dir)):
        is_eutils = isinstance(document, dict) and "PubmedArticleSet" in document
        if is_eutils:
            raw_set = document["PubmedArticleSet"]
            items = raw_set.get("PubmedArticle", []) if isinstance(raw_set, dict) else raw_set
            if isinstance(items, dict):
                items = [items]
        else:
            items = extract_items(document, "PubmedArticleSet")

        for item in items or []:
            if not isinstance(item, dict):
                continue
            try:
                article = _from_eutils(item) if is_eutils else _from_sample(item)
            except (ValidationError, ValueError, KeyError, TypeError) as exc:
                logger.warning("Skipping malformed PubMed article in {}: {}", filename, exc)
                continue
            if not article.abstract.strip():
                logger.debug("Skipping PubMed article {} with empty abstract", article.pmid)
                continue
            articles.append(article)

    logger.info("Loaded {} PubMed articles from {}", len(articles), data_dir)
    return articles
