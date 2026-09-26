"""LLM-based entity extraction for enriching records.

Uses :class:`~src.utils.llm_client.GroqClient` to pull drug names and
adverse-event terms out of free text (e.g. PubMed abstracts). Enrichment is
best-effort: on any failure the original record is returned unchanged and a
warning is logged, so a bad extraction never breaks ingestion.

Requires the Groq API and therefore only runs on the personal laptop; on the
dev machine this module is syntax-checked only.
"""

from loguru import logger

from src.models.schemas import PubMedArticle
from src.utils.llm_client import GroqClient

# System prompt instructing the model to extract entities as JSON.
ENTITY_EXTRACTION_PROMPT_V1 = (  # V1
    "You are a biomedical entity extraction system. Given a passage of medical "
    "text, extract two lists:\n"
    "  1. drugs — generic or brand drug names explicitly mentioned.\n"
    "  2. adverse_events — adverse reactions, side effects, or symptoms mentioned.\n\n"
    "Return ONLY a JSON object with exactly these keys and no commentary:\n"
    '{"drugs": ["..."], "adverse_events": ["..."]}\n'
    "Use lowercase. If none are present, return empty lists."
)


class MedicalExtractor:
    """Extracts drug and adverse-event entities from text via an LLM."""

    def __init__(self, client: GroqClient | None = None) -> None:
        """Initialize the extractor.

        Args:
            client: An existing :class:`GroqClient`, or ``None`` to construct one.
        """
        self.client = client or GroqClient()

    def extract_entities(self, text: str) -> dict[str, list[str]]:
        """Extract drug and adverse-event entities from ``text``.

        Args:
            text: The passage to analyze.

        Returns:
            A dict with keys ``"drugs"`` and ``"adverse_events"`` mapping to
            lists of strings (empty lists on empty input or parse issues).
        """
        if not text or not text.strip():
            return {"drugs": [], "adverse_events": []}

        result = self.client.generate_json(
            prompt=text,
            system_prompt=ENTITY_EXTRACTION_PROMPT_V1,
            temperature=0.0,
        )
        drugs = result.get("drugs", []) or []
        adverse_events = result.get("adverse_events", []) or []
        return {
            "drugs": [str(d).strip().lower() for d in drugs if str(d).strip()],
            "adverse_events": [
                str(a).strip().lower() for a in adverse_events if str(a).strip()
            ],
        }

    def enrich_pubmed_article(self, article: PubMedArticle) -> PubMedArticle:
        """Populate an article's entity mentions from its abstract if missing.

        Args:
            article: The article to enrich.

        Returns:
            An enriched copy, or the original unchanged if extraction fails or
            the mentions are already populated.
        """
        if article.drugs_mentioned and article.adverse_events_mentioned:
            return article
        try:
            entities = self.extract_entities(article.abstract)
        except Exception as exc:  # noqa: BLE001 - enrichment is best-effort
            logger.warning("Entity extraction failed for PMID {}: {}", article.pmid, exc)
            return article

        return article.model_copy(
            update={
                "drugs_mentioned": article.drugs_mentioned or entities["drugs"],
                "adverse_events_mentioned": (
                    article.adverse_events_mentioned or entities["adverse_events"]
                ),
            }
        )

    def enrich_batch(self, articles: list[PubMedArticle]) -> list[PubMedArticle]:
        """Enrich a batch of articles, logging progress.

        Args:
            articles: Articles to enrich.

        Returns:
            The list of (possibly) enriched articles, order preserved.
        """
        total = len(articles)
        enriched: list[PubMedArticle] = []
        for i, article in enumerate(articles, start=1):
            enriched.append(self.enrich_pubmed_article(article))
            if i % 10 == 0 or i == total:
                logger.info("Enriched {}/{} PubMed articles", i, total)
        return enriched
