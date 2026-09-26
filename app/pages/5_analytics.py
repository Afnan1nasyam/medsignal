"""Page: Analytics — Plotly dashboards over FAERS, the graph, and the query log."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plotly.graph_objects as go
import streamlit as st

from components.api import api_call
from components.theme import BAR_BLUE, GRADE_COLORS, GRADE_LABELS, STATUS, style_fig

st.set_page_config(page_title="Analytics", page_icon="📊", layout="wide")

st.title("📊 Analytics")
st.caption("Aggregate views across FAERS reports, the knowledge graph, and the query log.")


def _hbar(pairs: list[tuple[str, int]], title: str, xlabel: str) -> go.Figure:
    """Horizontal magnitude bar (single blue hue), largest on top, direct labels."""
    pairs = pairs[::-1]  # plotly draws bottom-up; reverse so largest is on top
    labels = [p[0] for p in pairs]
    values = [p[1] for p in pairs]
    fig = go.Figure(
        go.Bar(
            x=values, y=labels, orientation="h",
            marker_color=BAR_BLUE, marker_line_width=0,
            text=values, textposition="outside", cliponaxis=False,
            hovertemplate="%{y}: %{x}<extra></extra>",
        )
    )
    fig.update_layout(title=title)
    fig.update_xaxes(title=xlabel)
    return style_fig(fig, height=460)


# -- metric cards (totals) --
status = api_call("/api/ingest/status")
if status.get("ok"):
    sql = status["data"].get("sql", {})
    m = st.columns(4)
    m[0].metric("FAERS reports", sql.get("faers_reports", 0))
    m[1].metric("PubMed articles", sql.get("pubmed_articles", 0))
    m[2].metric("Clinical trials", sql.get("clinical_trials", 0))
    m[3].metric("Drug labels", sql.get("drug_labels", 0))
else:
    st.warning(f"{status.get('error')}: {status.get('detail', '')}")

st.divider()

# -- top drugs & adverse events --
left, right = st.columns(2)
with left:
    res = api_call("/api/analytics/top-drugs", data={"limit": 15})
    if res.get("ok") and res["data"].get("top_drugs"):
        pairs = [(d["drug"], d["report_count"]) for d in res["data"]["top_drugs"]]
        st.plotly_chart(_hbar(pairs, "Top drugs by FAERS reports", "reports"), use_container_width=True)
    else:
        st.info("No FAERS drug data yet.")

with right:
    res = api_call("/api/analytics/top-adverse-events", data={"limit": 15})
    if res.get("ok") and res["data"].get("top_adverse_events"):
        pairs = [(a["reaction"], a["report_count"]) for a in res["data"]["top_adverse_events"]]
        st.plotly_chart(_hbar(pairs, "Top adverse events", "reports"), use_container_width=True)
    else:
        st.info("No FAERS adverse-event data yet.")

st.divider()

# -- outcome seriousness pie & drug-class bar --
left2, right2 = st.columns(2)
with left2:
    res = api_call("/api/analytics/outcome-distribution")
    if res.get("ok") and (res["data"].get("serious", 0) or res["data"].get("non_serious", 0)):
        data = res["data"]
        fig = go.Figure(
            go.Pie(
                labels=["Serious", "Non-serious"],
                values=[data.get("serious", 0), data.get("non_serious", 0)],
                marker_colors=[STATUS["critical"], STATUS["good"]],
                hole=0.0, textinfo="label+percent", sort=False,
            )
        )
        fig.update_layout(title="Reports by seriousness")
        st.plotly_chart(style_fig(fig, height=420, showlegend=True), use_container_width=True)
    else:
        st.info("No outcome data yet.")

with right2:
    res = api_call("/api/analytics/drug-class-profile")
    profile = res["data"].get("drug_class_profile", []) if res.get("ok") else []
    if profile:
        pairs = [(p["drug_class"].replace("_", " "), p["total_reports"]) for p in profile]
        st.plotly_chart(_hbar(pairs, "Adverse-event reports by drug class", "reports"), use_container_width=True)
    else:
        st.info("No drug-class graph data yet.")

st.divider()

# -- evidence grade distribution (donut) from the query log --
res = api_call("/api/analytics/query-stats")
if res.get("ok") and res["data"].get("by_grade"):
    by_grade = res["data"]["by_grade"]
    order = [g for g in ["A", "B", "C", "D", "E"] if g in by_grade]
    fig = go.Figure(
        go.Pie(
            labels=[f"{g} · {GRADE_LABELS[g]}" for g in order],
            values=[by_grade[g] for g in order],
            marker_colors=[GRADE_COLORS[g] for g in order],
            hole=0.55, textinfo="label+value", sort=False,
        )
    )
    fig.update_layout(title="Evidence grade distribution (from query log)")
    st.plotly_chart(style_fig(fig, height=420, showlegend=True), use_container_width=True)
else:
    st.info("No queries logged yet — run some questions on the Safety Query page.")
