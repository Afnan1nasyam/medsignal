"""Streamlit component for rendering a drug safety profile."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from components.theme import SEVERITY_COLORS


def _severity_chip(severity: str) -> str:
    """Return an HTML chip for an interaction severity."""
    sev = str(severity or "").lower()
    color = SEVERITY_COLORS.get(sev, "#898781")
    return (
        f'<span style="background:{color};color:#fff;border-radius:8px;'
        f'padding:2px 10px;font-size:12px;font-weight:600;">{sev or "unknown"}</span>'
    )


def render_profile(profile: dict, neighbors: list[dict]) -> None:
    """Render a full drug safety profile: metrics, AEs, interactions, related drugs.

    Args:
        profile: The ``DrugSafetyProfile`` dict from ``/api/drugs/{name}``.
        neighbors: The ``graph_neighbors`` list (related/same-class drugs).
    """
    st.subheader(f"Safety profile — {profile.get('drug_name', 'unknown')}")
    if profile.get("evidence_summary"):
        st.caption(profile["evidence_summary"])

    # Top row: metric cards.
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("FAERS reports", profile.get("total_faers_reports", 0))
    c2.metric("Known interactions", len(profile.get("known_interactions", []) or []))
    c3.metric("Active trials", profile.get("active_trials", 0))
    c4.metric("Publications", profile.get("related_publications", 0))

    st.divider()
    left, right = st.columns(2)

    # Adverse events table.
    with left:
        st.markdown("**Adverse events (from graph)**")
        aes = profile.get("top_adverse_events", []) or []
        if aes:
            df = pd.DataFrame(
                [
                    {"Adverse event": a.get("preferred_term", "?"), "Reports": a.get("report_count", 0)}
                    for a in aes
                ]
            )
            st.dataframe(df, hide_index=True, use_container_width=True)
        else:
            st.info("No adverse-event edges in the graph for this drug.")

    # Interactions with severity colors.
    with right:
        st.markdown("**Drug interactions**")
        interactions = profile.get("known_interactions", []) or []
        if interactions:
            for it in interactions:
                name = it.get("drug_name", "?")
                st.markdown(
                    f"- {name} &nbsp; {_severity_chip(it.get('severity'))}",
                    unsafe_allow_html=True,
                )
        else:
            st.info("No interaction edges recorded.")

    st.divider()

    # Contraindications.
    st.markdown("**Contraindications**")
    contraindications = profile.get("contraindications", []) or []
    if contraindications:
        for contra in contraindications:
            st.markdown(f"- {contra}")
    else:
        st.caption("None recorded.")

    # Related drugs as chips.
    st.markdown("**Related drugs (same class / interacting)**")
    if neighbors:
        chips = " ".join(
            f'<span style="background:#eef3fb;color:#1c5cab;border:1px solid #c9def7;'
            f'border-radius:14px;padding:3px 12px;margin:2px;display:inline-block;font-size:13px;">'
            f'{n.get("drug_name", n.get("name", "?"))}'
            f'<span style="opacity:0.6;"> · {n.get("hops", "?")} hop</span></span>'
            for n in neighbors
        )
        st.markdown(chips, unsafe_allow_html=True)
    else:
        st.caption("No related drugs found in the graph.")
