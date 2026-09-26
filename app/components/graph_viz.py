"""Pyvis interactive knowledge-graph rendering for Streamlit."""

from __future__ import annotations

from pyvis.network import Network

from components.theme import NODE_COLORS, NODE_LABELS


def _node_color(node_type: str) -> str:
    """Color for a node type (falls back to a neutral gray)."""
    return NODE_COLORS.get(node_type, "#898781")


def build_network_html(
    nodes: list[dict],
    edges: list[dict],
    height: int = 620,
    highlight: str | None = None,
) -> str:
    """Build an interactive Pyvis network and return its HTML string.

    Args:
        nodes: ``[{"id", "label", "type"}]``.
        edges: ``[{"from", "to", "label"?, "weight"?}]``.
        height: Canvas height in pixels.
        highlight: Optional node id to emphasize (larger, ringed).

    Returns:
        Self-contained HTML for embedding via ``st.components.v1.html``.
    """
    net = Network(
        height=f"{height}px",
        width="100%",
        directed=True,
        bgcolor="#fcfcfb",
        font_color="#0b0b0b",
        cdn_resources="in_line",
    )
    net.barnes_hut(spring_length=140, gravity=-8000)

    for node in nodes:
        node_type = node.get("type", "")
        is_focus = highlight is not None and node.get("id") == highlight
        net.add_node(
            node["id"],
            label=node.get("label", node["id"]),
            color=_node_color(node_type),
            shape="dot",
            size=26 if is_focus else 16,
            borderWidth=4 if is_focus else 1,
            title=f"{NODE_LABELS.get(node_type, node_type)}: {node.get('label', node['id'])}",
        )

    for edge in edges:
        weight = edge.get("weight")
        net.add_edge(
            edge["from"],
            edge["to"],
            title=edge.get("label", ""),
            label=str(weight) if weight else "",
            value=weight if weight else 1,
            color="#c3c2b7",
        )

    net.set_options(
        """
        var options = {
          "interaction": {"hover": true, "tooltipDelay": 120},
          "physics": {"stabilization": {"iterations": 150}},
          "nodes": {"font": {"size": 14, "face": "system-ui"}}
        }
        """
    )
    try:
        return net.generate_html(notebook=False)
    except TypeError:
        return net.generate_html()


def legend_html() -> str:
    """Return a small HTML legend mapping node colors to types."""
    items = "".join(
        f'<span style="display:inline-flex;align-items:center;gap:6px;margin-right:14px;">'
        f'<span style="width:12px;height:12px;border-radius:50%;background:{color};'
        f'display:inline-block;"></span>{NODE_LABELS.get(t, t)}</span>'
        for t, color in NODE_COLORS.items()
    )
    return f'<div style="font-family:system-ui;font-size:13px;color:#52514e;">{items}</div>'
