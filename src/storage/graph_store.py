"""Biomedical knowledge-graph storage.

Defines a backend-agnostic :class:`GraphStoreProtocol` and two implementations:

* :class:`NetworkXGraphStore` — an in-process ``DiGraph`` with JSON persistence,
  used during development (no server required). Fully offline.
* :class:`Neo4jGraphStore` — a Cypher-backed store for demo/prod. Its driver
  import and connection are lazy so this module still imports on a dev machine
  that has neither the ``neo4j`` package nor a running server.

Use :func:`get_graph_store` to obtain the backend selected by
``settings.GRAPH_BACKEND``.

Node IDs are normalized as ``"{prefix}:{key}"`` (e.g. ``"drug:metformin"``,
``"ae:nausea"``, ``"trial:NCT12345"``). Name-like nodes (drug, adverse event,
condition, drug class) are lower-cased; identifier-like nodes (trials by NCT id,
publications by PMID) keep their original case.
"""

from __future__ import annotations

import json
from collections import Counter, deque
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import networkx as nx
from loguru import logger

from src.config import settings
from src.models.enums import EdgeType, NodeType


# --------------------------------------------------------------------------- #
# ID normalization
# --------------------------------------------------------------------------- #
#: Node-ID prefix per node type.
_PREFIX: dict[NodeType, str] = {
    NodeType.DRUG: "drug",
    NodeType.ADVERSE_EVENT: "ae",
    NodeType.CONDITION: "condition",
    NodeType.CLINICAL_TRIAL: "trial",
    NodeType.PUBLICATION: "pub",
    NodeType.DRUG_CLASS: "drug_class",
}

#: Node types whose key is a free-text name and should be lower-cased.
_LOWERCASE_TYPES = {
    NodeType.DRUG,
    NodeType.ADVERSE_EVENT,
    NodeType.CONDITION,
    NodeType.DRUG_CLASS,
}


def _node_id(node_type: NodeType, key: str) -> str:
    """Build a normalized ``"{prefix}:{key}"`` node id for a type and key."""
    key = key.strip()
    if node_type in _LOWERCASE_TYPES:
        key = key.lower()
    return f"{_PREFIX[node_type]}:{key}"


def _coerce_edge_type(edge_type: EdgeType | str) -> EdgeType:
    """Coerce an ``EdgeType`` or its name/value string into an ``EdgeType``."""
    if isinstance(edge_type, EdgeType):
        return edge_type
    text = str(edge_type)
    try:
        return EdgeType(text)  # match by value, e.g. "REPORTED_WITH"
    except ValueError:
        return EdgeType[text.upper()]  # match by member name


# --------------------------------------------------------------------------- #
# Protocol
# --------------------------------------------------------------------------- #
@runtime_checkable
class GraphStoreProtocol(Protocol):
    """Backend-agnostic interface for the biomedical knowledge graph."""

    # -- node writers (each returns the normalized node id) ---------------- #
    def add_drug(self, name: str, properties: dict) -> str: ...
    def add_adverse_event(self, preferred_term: str, properties: dict) -> str: ...
    def add_condition(self, name: str, properties: dict) -> str: ...
    def add_clinical_trial(self, nct_id: str, properties: dict) -> str: ...
    def add_publication(self, pmid: str, properties: dict) -> str: ...
    def add_drug_class(self, name: str, properties: dict) -> str: ...

    # -- edge writer ------------------------------------------------------- #
    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType | str,
        properties: dict | None = None,
    ) -> None: ...

    # -- readers ----------------------------------------------------------- #
    def get_node(self, node_id: str) -> dict | None: ...
    def get_drug_profile(self, drug_name: str) -> dict: ...
    def get_drug_adverse_events(self, drug_name: str) -> list[dict]: ...
    def get_drug_interactions(self, drug_name: str) -> list[dict]: ...
    def get_adverse_event_drugs(self, ae_term: str) -> list[dict]: ...
    def get_related_drugs(self, drug_name: str, max_hops: int = 2) -> list[dict]: ...
    def get_drug_trials(self, drug_name: str) -> list[dict]: ...
    def get_drug_publications(self, drug_name: str) -> list[dict]: ...
    def get_stats(self) -> dict: ...

    # -- maintenance ------------------------------------------------------- #
    def reset(self) -> None: ...


# --------------------------------------------------------------------------- #
# NetworkX implementation
# --------------------------------------------------------------------------- #
class NetworkXGraphStore:
    """An in-process knowledge graph backed by a :class:`networkx.DiGraph`.

    Nodes carry a ``type`` attribute (a :class:`NodeType` value) plus a ``name``
    and any caller-supplied properties. Edges carry a ``type`` attribute (an
    :class:`EdgeType` value) plus properties. The graph is persisted to
    ``settings.GRAPH_PATH`` as node-link JSON.
    """

    def __init__(self, auto_load: bool = True) -> None:
        """Create the store, loading any persisted graph from disk.

        Args:
            auto_load: If True (default), load ``settings.GRAPH_PATH`` when present.
        """
        self.graph: nx.DiGraph = nx.DiGraph()
        self.path = Path(settings.GRAPH_PATH)
        if auto_load and self.path.exists():
            self.load()
        logger.info(
            "NetworkXGraphStore ready ({} nodes, {} edges)",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
        )

    # -- internal node writer --------------------------------------------- #
    def _add_node(self, node_type: NodeType, key: str, properties: dict) -> str:
        """MERGE-style upsert of a typed node; returns its normalized id."""
        node_id = _node_id(node_type, key)
        attrs = {"type": node_type.value, "name": key.strip(), **(properties or {})}
        if self.graph.has_node(node_id):
            self.graph.nodes[node_id].update(attrs)
        else:
            self.graph.add_node(node_id, **attrs)
        return node_id

    def add_drug(self, name: str, properties: dict | None = None) -> str:
        """Upsert a drug node. Returns its node id (``"drug:{name}"``)."""
        return self._add_node(NodeType.DRUG, name, properties or {})

    def add_adverse_event(
        self, preferred_term: str, properties: dict | None = None
    ) -> str:
        """Upsert an adverse-event node. Returns its node id (``"ae:{term}"``)."""
        return self._add_node(NodeType.ADVERSE_EVENT, preferred_term, properties or {})

    def add_condition(self, name: str, properties: dict | None = None) -> str:
        """Upsert a condition node. Returns its node id (``"condition:{name}"``)."""
        return self._add_node(NodeType.CONDITION, name, properties or {})

    def add_clinical_trial(
        self, nct_id: str, properties: dict | None = None
    ) -> str:
        """Upsert a clinical-trial node. Returns its node id (``"trial:{nct}"``)."""
        return self._add_node(NodeType.CLINICAL_TRIAL, nct_id, properties or {})

    def add_publication(self, pmid: str, properties: dict | None = None) -> str:
        """Upsert a publication node. Returns its node id (``"pub:{pmid}"``)."""
        return self._add_node(NodeType.PUBLICATION, pmid, properties or {})

    def add_drug_class(self, name: str, properties: dict | None = None) -> str:
        """Upsert a drug-class node. Returns its node id (``"drug_class:{name}"``)."""
        return self._add_node(NodeType.DRUG_CLASS, name, properties or {})

    # -- edge writer ------------------------------------------------------- #
    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType | str,
        properties: dict | None = None,
    ) -> None:
        """Add (or update) a typed, directed edge between two existing nodes.

        Args:
            source_id: Normalized id of the source node.
            target_id: Normalized id of the target node.
            edge_type: An :class:`EdgeType` or its name/value string.
            properties: Optional edge attributes.
        """
        etype = _coerce_edge_type(edge_type)
        for node_id in (source_id, target_id):
            if not self.graph.has_node(node_id):
                logger.warning("add_edge references unknown node: {}", node_id)
        self.graph.add_edge(
            source_id, target_id, type=etype.value, **(properties or {})
        )

    # -- generic neighborhood helper -------------------------------------- #
    def _incident(
        self,
        node_id: str,
        edge_types: set[str] | None = None,
        neighbor_types: set[str] | None = None,
        direction: str = "both",
    ) -> list[tuple[str, dict]]:
        """Return ``(neighbor_id, edge_attrs)`` for edges incident to a node.

        Args:
            node_id: Center node.
            edge_types: If given, keep only these edge ``type`` values.
            neighbor_types: If given, keep only neighbors of these node ``type`` values.
            direction: ``"out"``, ``"in"`` or ``"both"``.
        """
        if not self.graph.has_node(node_id):
            return []
        results: list[tuple[str, dict]] = []
        if direction in ("out", "both"):
            for _, nbr, data in self.graph.out_edges(node_id, data=True):
                results.append((nbr, data))
        if direction in ("in", "both"):
            for nbr, _, data in self.graph.in_edges(node_id, data=True):
                results.append((nbr, data))

        filtered: list[tuple[str, dict]] = []
        for nbr, data in results:
            if edge_types is not None and data.get("type") not in edge_types:
                continue
            if (
                neighbor_types is not None
                and self.graph.nodes.get(nbr, {}).get("type") not in neighbor_types
            ):
                continue
            filtered.append((nbr, data))
        return filtered

    def _neighbor_record(self, node_id: str, edge_data: dict) -> dict:
        """Assemble a flat result dict for a neighbor node + connecting edge."""
        node_attrs = dict(self.graph.nodes.get(node_id, {}))
        return {
            "node_id": node_id,
            "name": node_attrs.get("name", node_id.split(":", 1)[-1]),
            "node_type": node_attrs.get("type"),
            "properties": node_attrs,
            "edge": dict(edge_data),
        }

    # -- readers ----------------------------------------------------------- #
    def get_node(self, node_id: str) -> dict | None:
        """Return a node's attributes (with its id), or ``None`` if absent."""
        if not self.graph.has_node(node_id):
            return None
        return {"id": node_id, **dict(self.graph.nodes[node_id])}

    def get_drug_adverse_events(self, drug_name: str) -> list[dict]:
        """Adverse events reported with a drug, most-reported first.

        Follows ``REPORTED_WITH`` edges from the drug to adverse-event nodes.
        """
        drug_id = _node_id(NodeType.DRUG, drug_name)
        rows = []
        for nbr, data in self._incident(
            drug_id,
            edge_types={EdgeType.REPORTED_WITH.value},
            neighbor_types={NodeType.ADVERSE_EVENT.value},
            direction="out",
        ):
            attrs = dict(self.graph.nodes.get(nbr, {}))
            rows.append(
                {
                    "preferred_term": attrs.get("name", nbr.split(":", 1)[-1]),
                    "node_id": nbr,
                    "report_count": data.get("report_count", 0),
                    "properties": attrs,
                    "edge": dict(data),
                }
            )
        rows.sort(key=lambda r: r.get("report_count", 0), reverse=True)
        return rows

    def get_adverse_event_drugs(self, ae_term: str) -> list[dict]:
        """Drugs reported with an adverse event, most-reported first.

        Follows ``REPORTED_WITH`` edges into the adverse-event node.
        """
        ae_id = _node_id(NodeType.ADVERSE_EVENT, ae_term)
        rows = []
        for nbr, data in self._incident(
            ae_id,
            edge_types={EdgeType.REPORTED_WITH.value},
            neighbor_types={NodeType.DRUG.value},
            direction="in",
        ):
            attrs = dict(self.graph.nodes.get(nbr, {}))
            rows.append(
                {
                    "drug_name": attrs.get("name", nbr.split(":", 1)[-1]),
                    "node_id": nbr,
                    "report_count": data.get("report_count", 0),
                    "properties": attrs,
                    "edge": dict(data),
                }
            )
        rows.sort(key=lambda r: r.get("report_count", 0), reverse=True)
        return rows

    def get_drug_interactions(self, drug_name: str) -> list[dict]:
        """Drugs interacting with a drug (``INTERACTS_WITH``, either direction)."""
        drug_id = _node_id(NodeType.DRUG, drug_name)
        seen: set[str] = set()
        rows: list[dict] = []
        for nbr, data in self._incident(
            drug_id,
            edge_types={EdgeType.INTERACTS_WITH.value},
            neighbor_types={NodeType.DRUG.value},
            direction="both",
        ):
            if nbr in seen:
                continue
            seen.add(nbr)
            attrs = dict(self.graph.nodes.get(nbr, {}))
            rows.append(
                {
                    "drug_name": attrs.get("name", nbr.split(":", 1)[-1]),
                    "node_id": nbr,
                    "severity": data.get("severity"),
                    "properties": attrs,
                    "edge": dict(data),
                }
            )
        return rows

    def get_related_drugs(self, drug_name: str, max_hops: int = 2) -> list[dict]:
        """Drugs reachable within ``max_hops`` via class/interaction edges.

        Performs an undirected BFS following ``SAME_CLASS_AS`` and
        ``INTERACTS_WITH`` edges, returning each reachable drug once with the
        hop distance at which it was first found.
        """
        start = _node_id(NodeType.DRUG, drug_name)
        if not self.graph.has_node(start):
            return []
        follow = {EdgeType.SAME_CLASS_AS.value, EdgeType.INTERACTS_WITH.value}
        visited = {start: 0}
        queue: deque[str] = deque([start])
        results: list[dict] = []
        while queue:
            current = queue.popleft()
            hops = visited[current]
            if hops >= max_hops:
                continue
            for nbr, _ in self._incident(
                current, edge_types=follow, direction="both"
            ):
                if nbr in visited:
                    continue
                visited[nbr] = hops + 1
                queue.append(nbr)
                attrs = dict(self.graph.nodes.get(nbr, {}))
                if attrs.get("type") == NodeType.DRUG.value:
                    results.append(
                        {
                            "drug_name": attrs.get("name", nbr.split(":", 1)[-1]),
                            "node_id": nbr,
                            "hops": hops + 1,
                            "properties": attrs,
                        }
                    )
        results.sort(key=lambda r: (r["hops"], r["drug_name"]))
        return results

    def get_drug_trials(self, drug_name: str) -> list[dict]:
        """Clinical trials a drug was studied in (``STUDIED_IN`` -> trial)."""
        drug_id = _node_id(NodeType.DRUG, drug_name)
        return [
            self._neighbor_record(nbr, data)
            for nbr, data in self._incident(
                drug_id,
                edge_types={EdgeType.STUDIED_IN.value},
                neighbor_types={NodeType.CLINICAL_TRIAL.value},
                direction="out",
            )
        ]

    def get_drug_publications(self, drug_name: str) -> list[dict]:
        """Publications about a drug (any edge to a publication node)."""
        drug_id = _node_id(NodeType.DRUG, drug_name)
        return [
            self._neighbor_record(nbr, data)
            for nbr, data in self._incident(
                drug_id,
                neighbor_types={NodeType.PUBLICATION.value},
                direction="both",
            )
        ]

    def get_drug_profile(self, drug_name: str) -> dict:
        """Aggregate a drug's connected evidence into a safety-profile dict.

        Sums FAERS report counts across ``REPORTED_WITH`` edges, and lists
        interactions, trials, contraindications and publications.
        """
        drug_id = _node_id(NodeType.DRUG, drug_name)
        node = self.get_node(drug_id)
        adverse_events = self.get_drug_adverse_events(drug_name)
        interactions = self.get_drug_interactions(drug_name)
        trials = self.get_drug_trials(drug_name)
        publications = self.get_drug_publications(drug_name)

        contraindications = [
            self._neighbor_record(nbr, data)["name"]
            for nbr, data in self._incident(
                drug_id,
                edge_types={EdgeType.CONTRAINDICATED_FOR.value},
                direction="out",
            )
        ]
        total_faers = sum(ae.get("report_count", 0) for ae in adverse_events)

        return {
            "drug_name": node.get("name", drug_name) if node else drug_name,
            "exists": node is not None,
            "total_faers_reports": total_faers,
            "top_adverse_events": adverse_events[:10],
            "known_interactions": interactions,
            "contraindications": contraindications,
            "active_trials": len(trials),
            "trials": trials,
            "related_publications": len(publications),
            "publications": publications,
        }

    def get_stats(self) -> dict:
        """Node counts by type, edge counts by type, and top-10 connected drugs."""
        node_counts: Counter[str] = Counter(
            data.get("type", "unknown") for _, data in self.graph.nodes(data=True)
        )
        edge_counts: Counter[str] = Counter(
            data.get("type", "unknown") for _, _, data in self.graph.edges(data=True)
        )
        drug_degrees = [
            (self.graph.nodes[n].get("name", n), n, self.graph.degree(n))
            for n, data in self.graph.nodes(data=True)
            if data.get("type") == NodeType.DRUG.value
        ]
        drug_degrees.sort(key=lambda t: t[2], reverse=True)
        top_drugs = [
            {"drug_name": name, "node_id": nid, "connections": deg}
            for name, nid, deg in drug_degrees[:10]
        ]
        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "node_counts": dict(node_counts),
            "edge_counts": dict(edge_counts),
            "top_connected_drugs": top_drugs,
        }

    # -- persistence ------------------------------------------------------- #
    def save(self) -> None:
        """Persist the graph as node-link JSON at ``settings.GRAPH_PATH``."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            data = nx.node_link_data(self.graph, edges="edges")
        except TypeError:  # older networkx without the ``edges`` keyword
            data = nx.node_link_data(self.graph)
        with self.path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str)
        logger.info("Saved knowledge graph to {}", self.path)

    def load(self) -> None:
        """Load the graph from node-link JSON at ``settings.GRAPH_PATH``."""
        with self.path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        try:
            self.graph = nx.node_link_graph(
                data, directed=True, edges="edges"
            )
        except TypeError:  # older networkx without the ``edges`` keyword
            self.graph = nx.node_link_graph(data, directed=True)
        logger.info("Loaded knowledge graph from {}", self.path)

    def reset(self) -> None:
        """Clear the in-memory graph and remove the persisted JSON file."""
        self.graph = nx.DiGraph()
        if self.path.exists():
            self.path.unlink()
        logger.info("Knowledge graph reset (in-memory cleared, {} removed)", self.path)


# --------------------------------------------------------------------------- #
# Neo4j implementation (Cypher; exercised on the personal laptop)
# --------------------------------------------------------------------------- #
# Tested on personal laptop with Neo4j Desktop
class Neo4jGraphStore:
    """A Cypher-backed knowledge graph for demo/prod (Neo4j Community).

    The ``neo4j`` driver is imported lazily and the connection is wrapped in
    ``try/except`` so that constructing this store on a machine without Neo4j
    (or the driver) degrades gracefully to a disconnected no-op rather than
    raising at import time. Every method carries its full Cypher query; when the
    driver is unavailable the method logs and returns an empty result.
    """

    def __init__(self) -> None:
        """Open a Neo4j driver from ``settings``; stay disconnected on failure."""
        self.driver = None
        try:
            from neo4j import GraphDatabase

            self.driver = GraphDatabase.driver(
                settings.NEO4J_URI,
                auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
            )
            self.driver.verify_connectivity()
            logger.info("Neo4jGraphStore connected to {}", settings.NEO4J_URI)
        except Exception as exc:  # ImportError, ServiceUnavailable, auth, ...
            self.driver = None
            logger.warning("Neo4j unavailable ({}); store is disconnected", exc)

    # -- helpers ----------------------------------------------------------- #
    def _run(self, cypher: str, **params: Any) -> list[dict]:
        """Run a Cypher statement, returning row dicts (``[]`` on any failure)."""
        if self.driver is None:
            logger.warning("Neo4j driver unavailable; skipping query")
            return []
        try:
            with self.driver.session() as session:
                result = session.run(cypher, **params)
                return [record.data() for record in result]
        except Exception as exc:
            logger.error("Neo4j query failed: {}", exc)
            return []

    def close(self) -> None:
        """Close the underlying driver if connected."""
        if self.driver is not None:
            self.driver.close()

    # -- node writers ------------------------------------------------------ #
    def add_drug(self, name: str, properties: dict | None = None) -> str:
        cypher = "MERGE (d:Drug {name: $name}) SET d += $properties RETURN d"
        self._run(cypher, name=name, properties=properties or {})
        return _node_id(NodeType.DRUG, name)

    def add_adverse_event(
        self, preferred_term: str, properties: dict | None = None
    ) -> str:
        cypher = (
            "MERGE (ae:AdverseEvent {preferred_term: $term}) "
            "SET ae += $properties RETURN ae"
        )
        self._run(cypher, term=preferred_term, properties=properties or {})
        return _node_id(NodeType.ADVERSE_EVENT, preferred_term)

    def add_condition(self, name: str, properties: dict | None = None) -> str:
        cypher = "MERGE (c:Condition {name: $name}) SET c += $properties RETURN c"
        self._run(cypher, name=name, properties=properties or {})
        return _node_id(NodeType.CONDITION, name)

    def add_clinical_trial(
        self, nct_id: str, properties: dict | None = None
    ) -> str:
        cypher = (
            "MERGE (t:ClinicalTrial {nct_id: $nct_id}) "
            "SET t += $properties RETURN t"
        )
        self._run(cypher, nct_id=nct_id, properties=properties or {})
        return _node_id(NodeType.CLINICAL_TRIAL, nct_id)

    def add_publication(self, pmid: str, properties: dict | None = None) -> str:
        cypher = (
            "MERGE (p:Publication {pmid: $pmid}) SET p += $properties RETURN p"
        )
        self._run(cypher, pmid=pmid, properties=properties or {})
        return _node_id(NodeType.PUBLICATION, pmid)

    def add_drug_class(self, name: str, properties: dict | None = None) -> str:
        cypher = "MERGE (dc:DrugClass {name: $name}) SET dc += $properties RETURN dc"
        self._run(cypher, name=name, properties=properties or {})
        return _node_id(NodeType.DRUG_CLASS, name)

    # -- edge writer ------------------------------------------------------- #
    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType | str,
        properties: dict | None = None,
    ) -> None:
        """Create a typed relationship, matching nodes by their normalized ids.

        Node ids are stored on every node as an ``id`` property so that edges
        can be created without knowing each node's label.
        """
        etype = _coerce_edge_type(edge_type)
        # Relationship type cannot be parameterized in Cypher, so it is
        # interpolated from the validated EdgeType enum value (safe input).
        cypher = (
            "MATCH (a {id: $source_id}), (b {id: $target_id}) "
            f"MERGE (a)-[r:{etype.value}]->(b) "
            "SET r += $properties RETURN r"
        )
        self._run(
            cypher,
            source_id=source_id,
            target_id=target_id,
            properties=properties or {},
        )

    # -- readers ----------------------------------------------------------- #
    def get_node(self, node_id: str) -> dict | None:
        cypher = "MATCH (n {id: $node_id}) RETURN n LIMIT 1"
        rows = self._run(cypher, node_id=node_id)
        return rows[0].get("n") if rows else None

    def get_drug_adverse_events(self, drug_name: str) -> list[dict]:
        cypher = (
            "MATCH (d:Drug {name: $name})-[r:REPORTED_WITH]->(ae:AdverseEvent) "
            "RETURN ae AS adverse_event, r AS edge "
            "ORDER BY r.report_count DESC"
        )
        return self._run(cypher, name=drug_name)

    def get_adverse_event_drugs(self, ae_term: str) -> list[dict]:
        cypher = (
            "MATCH (d:Drug)-[r:REPORTED_WITH]->(ae:AdverseEvent {preferred_term: $term}) "
            "RETURN d AS drug, r AS edge "
            "ORDER BY r.report_count DESC"
        )
        return self._run(cypher, term=ae_term)

    def get_drug_interactions(self, drug_name: str) -> list[dict]:
        cypher = (
            "MATCH (d:Drug {name: $name})-[r:INTERACTS_WITH]-(other:Drug) "
            "RETURN DISTINCT other AS drug, r AS edge"
        )
        return self._run(cypher, name=drug_name)

    def get_related_drugs(self, drug_name: str, max_hops: int = 2) -> list[dict]:
        # Variable-length pattern bound is interpolated from a validated int.
        hops = int(max_hops)
        cypher = (
            "MATCH (d:Drug {name: $name})"
            f"-[:SAME_CLASS_AS|INTERACTS_WITH*1..{hops}]-(related:Drug) "
            "RETURN DISTINCT related"
        )
        return self._run(cypher, name=drug_name)

    def get_drug_trials(self, drug_name: str) -> list[dict]:
        cypher = (
            "MATCH (d:Drug {name: $name})-[:STUDIED_IN]->(t:ClinicalTrial) "
            "RETURN t AS trial"
        )
        return self._run(cypher, name=drug_name)

    def get_drug_publications(self, drug_name: str) -> list[dict]:
        cypher = (
            "MATCH (d:Drug {name: $name})-[:DESCRIBES|STUDIES_DRUG]-(p:Publication) "
            "RETURN DISTINCT p AS publication"
        )
        return self._run(cypher, name=drug_name)

    def get_drug_profile(self, drug_name: str) -> dict:
        cypher = (
            "MATCH (d:Drug {name: $name}) "
            "OPTIONAL MATCH (d)-[r:REPORTED_WITH]->(ae:AdverseEvent) "
            "OPTIONAL MATCH (d)-[:INTERACTS_WITH]-(other:Drug) "
            "OPTIONAL MATCH (d)-[:STUDIED_IN]->(t:ClinicalTrial) "
            "OPTIONAL MATCH (d)-[:CONTRAINDICATED_FOR]->(c:Condition) "
            "RETURN d AS drug, "
            "sum(coalesce(r.report_count, 0)) AS total_faers_reports, "
            "collect(DISTINCT ae) AS adverse_events, "
            "collect(DISTINCT other) AS interactions, "
            "count(DISTINCT t) AS active_trials, "
            "collect(DISTINCT c.name) AS contraindications"
        )
        rows = self._run(cypher, name=drug_name)
        if not rows:
            return {"drug_name": drug_name, "exists": False}
        row = rows[0]
        return {"drug_name": drug_name, "exists": True, **row}

    def get_stats(self) -> dict:
        node_cypher = (
            "MATCH (n) RETURN labels(n)[0] AS label, count(*) AS count"
        )
        edge_cypher = (
            "MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS count"
        )
        top_cypher = (
            "MATCH (d:Drug)-[r]-() "
            "RETURN d.name AS drug_name, count(r) AS connections "
            "ORDER BY connections DESC LIMIT 10"
        )
        return {
            "node_counts": {
                row["label"]: row["count"] for row in self._run(node_cypher)
            },
            "edge_counts": {
                row["type"]: row["count"] for row in self._run(edge_cypher)
            },
            "top_connected_drugs": self._run(top_cypher),
        }

    def reset(self) -> None:
        """Delete all nodes and relationships in the database."""
        self._run("MATCH (n) DETACH DELETE n")
        logger.info("Neo4j knowledge graph reset")


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def get_graph_store() -> GraphStoreProtocol:
    """Return the graph-store backend selected by ``settings.GRAPH_BACKEND``."""
    if settings.GRAPH_BACKEND == "neo4j":
        return Neo4jGraphStore()
    return NetworkXGraphStore()
