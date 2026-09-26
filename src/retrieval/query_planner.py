"""Query planning — decompose a drug-safety question into a retrieval plan.

Provides an LLM-backed planner (:meth:`QueryPlanner.plan`) and a fully offline,
keyword-based planner (:meth:`QueryPlanner.plan_without_llm`) used both as the
no-API path and as the fallback whenever the LLM call fails or returns
unusable JSON.
"""

from __future__ import annotations

from loguru import logger

from src.models.enums import DataSource, QueryIntent
from src.models.schemas import QueryPlan, SubQuery
from src.utils.llm_client import GroqClient
from src.utils.medical_terms import (
    BRAND_TO_GENERIC,
    COMMON_ADVERSE_EVENTS,
    DRUG_CLASSES,
    extract_ae_terms_from_text,
    extract_drug_names_from_text,
)

# Known drug vocabulary: every class member generic (medical_terms.DRUG_CLASSES
# now covers NSAIDs, antiplatelets, gabapentinoids, etc.), plus known brands and
# their generics, so both "atorvastatin" and "Lipitor" are recognised.
_KNOWN_DRUGS: list[str] = sorted(
    {drug for drugs in DRUG_CLASSES.values() for drug in drugs}
    | set(BRAND_TO_GENERIC.values())
    | set(BRAND_TO_GENERIC.keys())
)

# Adverse-event vocabulary comes straight from medical_terms (now enriched with
# the MedDRA / lay terms used across the sources) — single source of truth.
_KNOWN_ADVERSE_EVENTS: list[str] = list(dict.fromkeys(COMMON_ADVERSE_EVENTS))

# Intent keyword rules, evaluated most-specific first; first match wins.
_INTENT_KEYWORDS: list[tuple[QueryIntent, tuple[str, ...]]] = [
    (QueryIntent.DRUG_INTERACTION, ("interact", "interaction", "combination", "co-prescribe", "coprescribe", "together with", "concomitant", "use with", "used with", "taken with", "safe with", "along with", "combined with")),
    (QueryIntent.DRUG_COMPARISON, ("compare", "comparison", "versus", " vs ", " vs.", "better", "safer")),
    (QueryIntent.SIGNAL_DETECTION, ("signal", "emerging", "new evidence", "trend", "increasing", "disproportionate", "disproportionately")),
    (QueryIntent.SAFETY_PROFILE, ("safety", "profile", "overview", "how safe", "side effect", "side effects", "contraindication", "contraindications", "boxed warning", "black box", "black-box")),
    (QueryIntent.ADVERSE_EVENT_LOOKUP, ("adverse", "reaction", "cause", "associated with", "linked to", "report of", "reports of")),
]

_ALL_SOURCES = [
    DataSource.DRUG_LABELS,
    DataSource.FAERS,
    DataSource.PUBMED,
    DataSource.CLINICAL_TRIALS,
]

# System prompt (V1) instructing the LLM to emit a QueryPlan-shaped JSON object.
PLANNING_PROMPT_V1 = (  # V1
    "You are a drug-safety query planner for a pharmacovigilance RAG system.\n"
    "Given a user's question, analyze it and produce a structured retrieval plan.\n\n"
    "Do ALL of the following:\n"
    "1. Classify the query intent as exactly one of: "
    "safety_profile, drug_interaction, adverse_event_lookup, signal_detection, "
    "drug_comparison, general.\n"
    "2. Identify the drugs mentioned (lowercase generic names where possible).\n"
    "3. Identify the adverse events / reactions mentioned (lowercase).\n"
    "4. Decompose the question into 1-4 sub-queries. Each sub-query targets one "
    "or more data sources drawn from exactly these values: "
    '"drug_labels", "faers", "pubmed", "clinical_trials".\n'
    "   - Prefer drug_labels + faers for a drug's safety/adverse events.\n"
    "   - Add pubmed for supporting literature.\n"
    "   - Add clinical_trials when trials or studies are relevant.\n\n"
    "Return ONLY a JSON object with this exact shape and no commentary:\n"
    "{\n"
    '  "intent": "adverse_event_lookup",\n'
    '  "drugs_mentioned": ["..."],\n'
    '  "adverse_events_mentioned": ["..."],\n'
    '  "sub_queries": [\n'
    '    {"query": "...", "target_sources": ["drug_labels", "faers"], "reasoning": "..."}\n'
    "  ]\n"
    "}"
)


class QueryPlanner:
    """Builds a :class:`QueryPlan` from a natural-language query."""

    def __init__(self, llm_client: GroqClient | None = None) -> None:
        """Store the LLM client (``None`` forces the offline keyword planner)."""
        self.client = llm_client

    # -- LLM planner ------------------------------------------------------- #
    def plan(self, query: str) -> QueryPlan:
        """Plan via the LLM, falling back to the keyword planner on any failure.

        Args:
            query: The user's drug-safety question.

        Returns:
            A :class:`QueryPlan`.
        """
        if self.client is None:
            return self.plan_without_llm(query)
        try:
            data = self.client.generate_json(
                prompt=query, system_prompt=PLANNING_PROMPT_V1, temperature=0.0
            )
            plan = self._parse_llm_plan(query, data)
            if not plan.sub_queries:
                logger.warning("LLM plan had no sub-queries; using keyword fallback")
                return self.plan_without_llm(query)
            logger.info("LLM planned {} sub-queries (intent={})", len(plan.sub_queries), plan.intent.value)
            return plan
        except Exception as exc:  # noqa: BLE001 - any LLM/parse failure falls back
            logger.warning("LLM planning failed ({}); using keyword fallback", exc)
            return self.plan_without_llm(query)

    @staticmethod
    def _coerce_source(value: str) -> DataSource | None:
        """Coerce a source string (value or member name) to a ``DataSource``."""
        text = str(value).strip()
        try:
            return DataSource(text.lower())
        except ValueError:
            try:
                return DataSource[text.upper()]
            except KeyError:
                return None

    def _parse_llm_plan(self, query: str, data: dict) -> QueryPlan:
        """Map raw LLM JSON into a validated :class:`QueryPlan`."""
        try:
            intent = QueryIntent(str(data.get("intent", "general")).lower())
        except ValueError:
            intent = QueryIntent.GENERAL

        sub_queries: list[SubQuery] = []
        for raw in data.get("sub_queries", []) or []:
            if not isinstance(raw, dict):
                continue
            sources = [
                src
                for src in (self._coerce_source(s) for s in raw.get("target_sources", []) or [])
                if src is not None
            ]
            if not sources:
                sources = list(_ALL_SOURCES)
            sub_queries.append(
                SubQuery(
                    query=str(raw.get("query", query)),
                    target_sources=sources,
                    reasoning=str(raw.get("reasoning", "")),
                )
            )

        return QueryPlan(
            original_query=query,
            intent=intent,
            sub_queries=sub_queries,
            drugs_mentioned=[str(d).lower() for d in data.get("drugs_mentioned", []) or []],
            adverse_events_mentioned=[
                str(a).lower() for a in data.get("adverse_events_mentioned", []) or []
            ],
        )

    # -- offline keyword planner ------------------------------------------- #
    def _classify_intent(
        self, query_lower: str, drugs: list[str], adverse_events: list[str]
    ) -> QueryIntent:
        """Classify intent by keyword, with a drug+AE heuristic fallback."""
        for intent, keywords in _INTENT_KEYWORDS:
            if any(keyword in query_lower for keyword in keywords):
                return intent
        # Heuristic fallbacks when no keyword matched:
        # - two or more drugs named usually implies an interaction question;
        # - one drug plus a named adverse event implies an adverse-event lookup.
        if len(drugs) >= 2:
            return QueryIntent.DRUG_INTERACTION
        if drugs and adverse_events:
            return QueryIntent.ADVERSE_EVENT_LOOKUP
        return QueryIntent.GENERAL

    def plan_without_llm(self, query: str) -> QueryPlan:
        """Build a plan using only offline keyword/entity heuristics.

        Args:
            query: The user's drug-safety question.

        Returns:
            A :class:`QueryPlan` requiring no API access.
        """
        query_lower = query.lower()
        drugs = extract_drug_names_from_text(query, _KNOWN_DRUGS)
        adverse_events = extract_ae_terms_from_text(query, _KNOWN_ADVERSE_EVENTS)
        intent = self._classify_intent(query_lower, drugs, adverse_events)
        wants_trials = "trial" in query_lower or "study" in query_lower or "studies" in query_lower

        sub_queries: list[SubQuery] = []

        if intent == QueryIntent.DRUG_INTERACTION and len(drugs) >= 2:
            sub_queries.append(
                SubQuery(
                    query=f"interaction between {drugs[0]} and {drugs[1]}",
                    target_sources=[DataSource.DRUG_LABELS, DataSource.FAERS],
                    reasoning="Label interactions and co-reported adverse events for the pair.",
                )
            )
            sub_queries.append(
                SubQuery(
                    query=query,
                    target_sources=[DataSource.PUBMED],
                    reasoning="Literature describing the drug-drug interaction.",
                )
            )
        elif drugs:
            sub_queries.append(
                SubQuery(
                    query=query,
                    target_sources=[DataSource.DRUG_LABELS, DataSource.FAERS],
                    reasoning="Authoritative label plus spontaneous adverse-event reports.",
                )
            )
            sub_queries.append(
                SubQuery(
                    query=query,
                    target_sources=[DataSource.PUBMED],
                    reasoning="Supporting literature for the safety question.",
                )
            )
            if intent == QueryIntent.DRUG_COMPARISON or wants_trials:
                sub_queries.append(
                    SubQuery(
                        query=query,
                        target_sources=[DataSource.CLINICAL_TRIALS],
                        reasoning="Trial evidence for comparison / study-based claims.",
                    )
                )
        else:
            # No specific drug detected — search everything with the raw query.
            sub_queries.append(
                SubQuery(
                    query=query,
                    target_sources=list(_ALL_SOURCES),
                    reasoning="No specific drug detected; broad multi-source search.",
                )
            )

        return QueryPlan(
            original_query=query,
            intent=intent,
            sub_queries=sub_queries[:4],
            drugs_mentioned=drugs,
            adverse_events_mentioned=adverse_events,
        )
