"""Tests for the NetworkX knowledge-graph store (offline)."""

from __future__ import annotations

from src.storage.graph_store import NetworkXGraphStore


def test_add_drug_and_ae(tmp_data_dir):
    """Adding a drug, an AE, and an edge is retrievable."""
    graph = NetworkXGraphStore(auto_load=False)
    drug_id = graph.add_drug("metformin", {"drug_class": "biguanides"})
    ae_id = graph.add_adverse_event("Lactic acidosis", {})
    graph.add_edge(drug_id, ae_id, "REPORTED_WITH", {"report_count": 9})

    assert graph.get_node(drug_id)["name"] == "metformin"
    aes = graph.get_drug_adverse_events("metformin")
    assert len(aes) == 1
    assert aes[0]["preferred_term"] == "Lactic acidosis"
    assert aes[0]["report_count"] == 9


def test_get_drug_profile(sample_graph):
    """Profile aggregation sums FAERS counts and lists related evidence."""
    profile = sample_graph.get_drug_profile("metformin")
    assert profile["total_faers_reports"] == 14  # 9 + 5
    assert profile["active_trials"] == 1
    assert profile["related_publications"] == 1
    assert len(profile["top_adverse_events"]) == 2
    assert "Severe renal impairment" in profile["contraindications"]
    interacting = {i["drug_name"] for i in profile["known_interactions"]}
    assert "simvastatin" in interacting


def test_get_drug_interactions(sample_graph):
    """Interaction edges are retrievable in either direction."""
    interactions = sample_graph.get_drug_interactions("metformin")
    names = {i["drug_name"] for i in interactions}
    assert "simvastatin" in names


def test_save_load_roundtrip(sample_graph, tmp_data_dir):
    """Saving then reloading preserves node and edge counts."""
    sample_graph.save()
    reloaded = NetworkXGraphStore()  # auto-loads from the temp GRAPH_PATH
    before = sample_graph.get_stats()
    after = reloaded.get_stats()
    assert after["total_nodes"] == before["total_nodes"]
    assert after["total_edges"] == before["total_edges"]
    assert reloaded.get_node("drug:metformin") is not None


def test_related_drugs(sample_graph):
    """Related-drug BFS follows SAME_CLASS_AS and INTERACTS_WITH edges."""
    related = {r["drug_name"] for r in sample_graph.get_related_drugs("simvastatin", max_hops=2)}
    assert "atorvastatin" in related  # same class (1 hop)
    assert "metformin" in related  # interacts (1 hop)


def test_stats(sample_graph):
    """Stats dict exposes the expected structure."""
    stats = sample_graph.get_stats()
    for key in ["total_nodes", "total_edges", "node_counts", "edge_counts", "top_connected_drugs"]:
        assert key in stats
    assert stats["node_counts"].get("drug") == 3
    assert isinstance(stats["top_connected_drugs"], list)
