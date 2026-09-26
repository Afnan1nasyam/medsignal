"""Analytics routes — aggregate views over FAERS, the graph, and the query log."""

from __future__ import annotations

from collections import Counter

from fastapi import APIRouter, Depends, Query
from loguru import logger

from api.dependencies import get_graph_store_dep, get_sql_store
from src.models.enums import NodeType
from src.storage.graph_store import GraphStoreProtocol
from src.storage.sql_store import SQLStore
from src.utils.medical_terms import DRUG_CLASSES, normalize_drug_name

router = APIRouter(prefix="/api/analytics", tags=["analytics"])


@router.get("/top-drugs")
async def top_drugs(
    limit: int = Query(default=15, ge=1, le=100),
    sql: SQLStore = Depends(get_sql_store),
) -> dict:
    """Return the most frequently reported drugs across FAERS reports."""
    counts: Counter[str] = Counter()
    try:
        for row in sql.search_faers():
            for drug in row.get("drugs") or []:
                name = (drug.get("generic_name") or drug.get("name") or "").strip().lower()
                if name:
                    counts[name] += 1
    except Exception as exc:  # noqa: BLE001
        logger.error("top-drugs aggregation failed: {}", exc)
    return {
        "top_drugs": [
            {"drug": name, "report_count": count} for name, count in counts.most_common(limit)
        ]
    }


@router.get("/top-adverse-events")
async def top_adverse_events(
    limit: int = Query(default=15, ge=1, le=100),
    sql: SQLStore = Depends(get_sql_store),
) -> dict:
    """Return the most frequently reported adverse events across FAERS reports."""
    counts: Counter[str] = Counter()
    try:
        for row in sql.search_faers():
            for reaction in row.get("reactions") or []:
                term = (reaction.get("preferred_term") or "").strip()
                if term:
                    counts[term] += 1
    except Exception as exc:  # noqa: BLE001
        logger.error("top-adverse-events aggregation failed: {}", exc)
    return {
        "top_adverse_events": [
            {"reaction": term, "report_count": count}
            for term, count in counts.most_common(limit)
        ]
    }


@router.get("/outcome-distribution")
async def outcome_distribution(sql: SQLStore = Depends(get_sql_store)) -> dict:
    """Return FAERS seriousness split and reaction-outcome distribution."""
    serious = non_serious = 0
    outcomes: Counter[str] = Counter()
    try:
        for row in sql.search_faers():
            if row.get("serious"):
                serious += 1
            else:
                non_serious += 1
            for reaction in row.get("reactions") or []:
                outcome = (reaction.get("outcome") or "other").strip()
                outcomes[outcome] += 1
    except Exception as exc:  # noqa: BLE001
        logger.error("outcome-distribution aggregation failed: {}", exc)
    return {
        "serious": serious,
        "non_serious": non_serious,
        "by_outcome": dict(outcomes),
    }


@router.get("/drug-class-profile")
async def drug_class_profile(
    graph: GraphStoreProtocol = Depends(get_graph_store_dep),
) -> dict:
    """Return total adverse-event report counts aggregated per drug class."""
    profile: list[dict] = []
    for drug_class, members in DRUG_CLASSES.items():
        total_reports = 0
        present = 0
        for member in members:
            if graph.get_node(f"drug:{member}") is None:
                continue
            present += 1
            try:
                for ae in graph.get_drug_adverse_events(member):
                    total_reports += int(ae.get("report_count", 0) or 0)
            except Exception as exc:  # noqa: BLE001
                logger.error("class aggregation failed for {}: {}", member, exc)
        if present:
            profile.append(
                {"drug_class": drug_class, "drug_count": present, "total_reports": total_reports}
            )
    profile.sort(key=lambda p: p["total_reports"], reverse=True)
    return {"drug_class_profile": profile}


@router.get("/signals")
async def emerging_signals(
    graph: GraphStoreProtocol = Depends(get_graph_store_dep),
    sql: SQLStore = Depends(get_sql_store),
) -> dict:
    """Surface potential emerging signals: high FAERS counts not (yet) on the label.

    A signal is a drug -> adverse-event pair carrying a FAERS ``report_count``.
    ``in_label`` is True when the drug's SQL label mentions the event in its
    adverse-reactions or warnings sections.
    """
    inner = getattr(graph, "graph", None)  # NetworkX DiGraph
    if inner is None:
        return {"signals": [], "note": "signal scan requires the NetworkX backend"}

    # Pre-index label AE/warning terms by normalized generic name.
    label_terms: dict[str, set[str]] = {}
    for label in sql.search_labels():
        generic = normalize_drug_name(label.get("generic_name") or label.get("drug_name") or "")
        terms: set[str] = set()
        for reaction in label.get("adverse_reactions") or []:
            terms.add(str(reaction.get("reaction", "")).lower())
        for warning in label.get("warnings") or []:
            terms.add(str(warning).lower())
        label_terms[generic] = terms

    signals: list[dict] = []
    for source_id, target_id, data in inner.edges(data=True):
        if data.get("type") != "REPORTED_WITH":
            continue
        if inner.nodes[source_id].get("type") != NodeType.DRUG.value:
            continue
        if inner.nodes[target_id].get("type") != NodeType.ADVERSE_EVENT.value:
            continue
        faers_count = data.get("report_count")
        if not faers_count:
            continue  # only FAERS-derived edges carry a report count

        drug = inner.nodes[source_id].get("name", source_id.split(":", 1)[-1])
        ae = inner.nodes[target_id].get("name", target_id.split(":", 1)[-1])
        generic = normalize_drug_name(drug)
        ae_lower = ae.lower()
        terms = label_terms.get(generic, set())
        in_label = any(ae_lower in t or (t and t in ae_lower) for t in terms)

        try:
            publications = len(graph.get_drug_publications(generic))
        except Exception:  # noqa: BLE001
            publications = 0

        signals.append(
            {
                "drug": drug,
                "adverse_event": ae,
                "faers_count": int(faers_count),
                "in_label": in_label,
                "publications": publications,
            }
        )

    signals.sort(key=lambda s: s["faers_count"], reverse=True)
    return {"signals": signals}


@router.get("/query-stats")
async def query_stats(sql: SQLStore = Depends(get_sql_store)) -> dict:
    """Return query-log analytics (totals, by-intent, by-grade, avg time)."""
    return sql.get_query_stats()
