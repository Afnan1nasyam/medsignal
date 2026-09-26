"""Page: Knowledge Graph — interactive ego-network exploration via Pyvis.

The backend exposes per-node neighbors (not a full-graph dump), so this page
builds an ego-network around a chosen drug: its neighbors, and optionally one
more hop. Node-type filters and a minimum edge-weight slider refine the view.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
import streamlit.components.v1 as components

from components.api import api_call
from components.graph_viz import build_network_html, legend_html
from components.theme import NODE_LABELS

st.set_page_config(page_title="Knowledge Graph", page_icon="🕸️", layout="wide")

st.title("🕸️ Knowledge Graph")
st.caption("Interactive exploration of the biomedical knowledge graph (ego-network around a drug).")

drugs_result = api_call("/api/drugs")
known = drugs_result.get("data", {}).get("drugs", []) if drugs_result.get("ok") else []
if not drugs_result.get("ok"):
    st.warning(f"{drugs_result.get('error')}: {drugs_result.get('detail', '')}")
    st.stop()
if not known:
    st.info("The graph is empty. Ingest data first: `python scripts/ingest_all.py`.")
    st.stop()

# Controls
ctrl = st.columns([2, 1, 1])
with ctrl[0]:
    focus_drug = st.selectbox("Center on drug", options=known)
with ctrl[1]:
    expand = st.checkbox("Expand 2 hops", value=False)
with ctrl[2]:
    min_weight = st.slider("Min edge weight", 0, 100, 0)

visible_types = st.multiselect(
    "Show node types",
    options=list(NODE_LABELS.keys()),
    default=list(NODE_LABELS.keys()),
    format_func=lambda t: NODE_LABELS[t],
)

center_id = f"drug:{focus_drug.lower()}"


def _collect(node_id: str, nodes: dict, edges: list, visited: set, depth: int) -> None:
    """BFS-collect neighbors from the API up to ``depth`` hops."""
    if depth < 0 or node_id in visited:
        return
    visited.add(node_id)
    resp = api_call(f"/api/graph/neighbors/{node_id}")
    if not resp.get("ok"):
        return
    data = resp["data"]
    center = data.get("node", {})
    nodes[node_id] = {"id": node_id, "label": center.get("name", node_id.split(":")[-1]), "type": center.get("type", "")}
    for nb in data.get("neighbors", []):
        nid = nb["node_id"]
        ndata = nb.get("node", {})
        nodes[nid] = {"id": nid, "label": ndata.get("name", nid.split(":")[-1]), "type": ndata.get("type", "")}
        weight = 0
        # weight lives on the edge; neighbors endpoint returns edge_type only,
        # so treat REPORTED_WITH as weighted via the node's own data when present.
        edges.append({
            "from": node_id if nb.get("direction") == "out" else nid,
            "to": nid if nb.get("direction") == "out" else node_id,
            "label": nb.get("edge_type", ""),
            "weight": weight,
        })
        if depth - 1 >= 0:
            _collect(nid, nodes, edges, visited, depth - 1)


nodes: dict = {}
edges: list = []
_collect(center_id, nodes, edges, set(), 1 if expand else 0)

# Apply node-type filter (and drop edges to filtered-out nodes).
kept = {nid: n for nid, n in nodes.items() if n["type"] in visible_types or nid == center_id}
kept_edges = [
    e for e in edges
    if e["from"] in kept and e["to"] in kept and e.get("weight", 0) >= min_weight
]

st.markdown(legend_html(), unsafe_allow_html=True)
if kept:
    html = build_network_html(list(kept.values()), kept_edges, height=600, highlight=center_id)
    components.html(html, height=640, scrolling=False)
else:
    st.info("No nodes match the current filters.")

# Stats table
st.subheader("Graph statistics")
stats = api_call("/api/graph/stats")
if stats.get("ok"):
    data = stats["data"]
    s1, s2 = st.columns(2)
    s1.metric("Total nodes", data.get("total_nodes", 0))
    s2.metric("Total edges", data.get("total_edges", 0))
    s1.write("**Nodes by type**")
    s1.json(data.get("node_counts", {}))
    s2.write("**Edges by type**")
    s2.json(data.get("edge_counts", {}))
