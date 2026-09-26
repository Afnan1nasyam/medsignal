"""Knowledge-graph retrieval — structured, multi-hop drug-safety queries.

Wraps a :class:`~src.storage.graph_store.GraphStoreProtocol` backend to answer
relationship questions the vector stores cannot: aggregated safety profiles,
same-class comparisons, adverse-event landscapes, and pairwise interaction risk.

Drug names are normalized (brand -> generic, lower-cased) before every graph
lookup so callers may pass either a brand or a generic name.
"""

from __future__ import annotations

from loguru import logger

from src.models.enums import NodeType
from src.models.schemas import DrugSafetyProfile
from src.storage.graph_store import GraphStoreProtocol, get_graph_store
from src.utils.medical_terms import (
    DRUG_CLASSES,
    normalize_adverse_event,
    normalize_drug_name,
)

# Reverse lookup: generic drug name -> its drug class.
_DRUG_TO_CLASS: dict[str, str] = {
    drug: cls for cls, drugs in DRUG_CLASSES.items() for drug in drugs
}


class GraphRetriever:
    """Answers structured drug-safety questions against the knowledge graph."""

    def __init__(self, graph_store: GraphStoreProtocol | None = None) -> None:
        """Store the graph backend (defaults to the configured one).

        Args:
            graph_store: A graph store, or ``None`` to use :func:`get_graph_store`.
        """
        self.graph: GraphStoreProtocol = graph_store or get_graph_store()

    # -- helpers ----------------------------------------------------------- #
    @staticmethod
    def _drug_id(generic: str) -> str:
        """Build the canonical drug node id for a normalized generic name."""
        return f"drug:{generic}"

    def _class_of(self, generic: str) -> str | None:
        """Resolve a drug's class from its graph node, else the static map."""
        node = self.graph.get_node(self._drug_id(generic))
        if node and node.get("drug_class"):
            return node["drug_class"]
        return _DRUG_TO_CLASS.get(generic)

    # -- safety profile ---------------------------------------------------- #
    def get_safety_profile(self, drug_name: str) -> DrugSafetyProfile:
        """Aggregate a drug's full safety profile from the graph.

        Args:
            drug_name: Brand or generic drug name.

        Returns:
            A :class:`DrugSafetyProfile` populated from graph aggregates.
        """
        generic = normalize_drug_name(drug_name)
        profile = self.graph.get_drug_profile(generic)

        top_aes = profile.get("top_adverse_events", []) or []
        interactions = profile.get("known_interactions", []) or []
        contraindications = profile.get("contraindications", []) or []
        active_trials = int(profile.get("active_trials", 0) or 0)
        publications = int(profile.get("related_publications", 0) or 0)
        total_faers = int(profile.get("total_faers_reports", 0) or 0)

        ae_names = [ae.get("preferred_term", "") for ae in top_aes[:5] if ae.get("preferred_term")]
        summary = (
            f"{generic}: {total_faers} FAERS report(s) across {len(top_aes)} "
            f"adverse event(s); {len(interactions)} known interaction(s); "
            f"{active_trials} trial(s); {publications} publication(s)."
        )
        if ae_names:
            summary += " Most reported: " + ", ".join(ae_names) + "."

        return DrugSafetyProfile(
            drug_name=generic,
            total_faers_reports=total_faers,
            top_adverse_events=top_aes,
            known_interactions=interactions,
            contraindications=contraindications,
            active_trials=active_trials,
            related_publications=publications,
            evidence_summary=summary,
        )

    # -- related drugs ----------------------------------------------------- #
    def get_related_drugs(self, drug_name: str) -> list[dict]:
        """Return same-class and interacting drugs (multi-hop graph BFS)."""
        generic = normalize_drug_name(drug_name)
        try:
            return self.graph.get_related_drugs(generic)
        except Exception as exc:  # noqa: BLE001
            logger.error("get_related_drugs failed for {}: {}", generic, exc)
            return []

    # -- class comparison -------------------------------------------------- #
    def get_drug_class_comparison(self, drug_name: str) -> dict:
        """Compare adverse-event profiles across drugs in the same class.

        Args:
            drug_name: Brand or generic drug name.

        Returns:
            A dict ``{drug_class, members: {generic: [top adverse events]}}``.
            ``members`` includes only class drugs present in the graph.
        """
        generic = normalize_drug_name(drug_name)
        drug_class = self._class_of(generic)
        if drug_class is None:
            logger.info("No known drug class for {}", generic)
            return {"drug_class": None, "query_drug": generic, "members": {}}

        members: dict[str, list[dict]] = {}
        for candidate in DRUG_CLASSES.get(drug_class, []):
            if self.graph.get_node(self._drug_id(candidate)) is None:
                continue  # not present in the graph
            try:
                aes = self.graph.get_drug_adverse_events(candidate)
            except Exception as exc:  # noqa: BLE001
                logger.error("AE lookup failed for {}: {}", candidate, exc)
                aes = []
            members[candidate] = [
                {"preferred_term": ae.get("preferred_term"), "report_count": ae.get("report_count", 0)}
                for ae in aes[:10]
            ]

        return {"drug_class": drug_class, "query_drug": generic, "members": members}

    # -- adverse-event landscape ------------------------------------------- #
    def get_ae_drug_landscape(self, ae_term: str) -> dict:
        """Return the drugs associated with an adverse event, with report counts.

        Args:
            ae_term: Adverse-event preferred term.

        Returns:
            A dict with the drug list (sorted by report count) and totals.
        """
        try:
            drugs = self.graph.get_adverse_event_drugs(ae_term)
        except Exception as exc:  # noqa: BLE001
            logger.error("AE landscape lookup failed for {}: {}", ae_term, exc)
            drugs = []

        entries = [
            {"drug_name": d.get("drug_name"), "report_count": d.get("report_count", 0)}
            for d in drugs
        ]
        entries.sort(key=lambda d: d.get("report_count", 0), reverse=True)
        return {
            "adverse_event": normalize_adverse_event(ae_term),
            "drugs": entries,
            "total_drugs": len(entries),
            "total_reports": sum(e.get("report_count", 0) for e in entries),
        }

    # -- pairwise interaction risk ----------------------------------------- #
    def find_interaction_risk(self, drug1: str, drug2: str) -> dict:
        """Assess interaction risk between two drugs from the graph.

        Checks for a direct ``INTERACTS_WITH`` edge and computes shared adverse
        events, then derives a qualitative risk level.

        Args:
            drug1: First drug (brand or generic).
            drug2: Second drug (brand or generic).

        Returns:
            A dict describing the direct interaction (if any), its severity,
            shared adverse events, and an overall risk level.
        """
        g1 = normalize_drug_name(drug1)
        g2 = normalize_drug_name(drug2)

        direct = False
        severity: str | None = None
        try:
            for interaction in self.graph.get_drug_interactions(g1):
                other = interaction.get("drug_name") or ""
                if normalize_drug_name(other) == g2 or interaction.get("node_id") == self._drug_id(g2):
                    direct = True
                    severity = (interaction.get("edge") or {}).get("severity") or interaction.get("severity")
                    break
        except Exception as exc:  # noqa: BLE001
            logger.error("Interaction lookup failed for {}: {}", g1, exc)

        def _ae_terms(generic: str) -> set[str]:
            try:
                return {
                    ae.get("preferred_term", "").lower()
                    for ae in self.graph.get_drug_adverse_events(generic)
                    if ae.get("preferred_term")
                }
            except Exception as exc:  # noqa: BLE001
                logger.error("AE set lookup failed for {}: {}", generic, exc)
                return set()

        shared = sorted(_ae_terms(g1) & _ae_terms(g2))

        if direct and severity and severity.lower() == "major":
            risk_level = "high"
        elif direct:
            risk_level = "moderate"
        elif shared:
            risk_level = "low"
        else:
            risk_level = "none"

        return {
            "drug1": g1,
            "drug2": g2,
            "direct_interaction": direct,
            "severity": severity,
            "shared_adverse_events": shared,
            "risk_level": risk_level,
        }
