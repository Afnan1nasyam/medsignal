"""Streamlit components for rendering evidence grades and citations."""

from __future__ import annotations

import streamlit as st

from components.theme import GRADE_COLORS, GRADE_LABELS, SOURCE_ICONS


def grade_badge(grade: str) -> None:
    """Render a large colored evidence-grade badge (letter + label, not color alone)."""
    grade = (grade or "E").upper()
    color = GRADE_COLORS.get(grade, GRADE_COLORS["E"])
    label = GRADE_LABELS.get(grade, "Insufficient")
    st.markdown(
        f"""
        <div style="display:inline-flex;align-items:center;gap:14px;
                    background:{color};color:#ffffff;border-radius:12px;
                    padding:14px 22px;font-family:system-ui,-apple-system,'Segoe UI',sans-serif;">
            <span style="font-size:40px;font-weight:700;line-height:1;">{grade}</span>
            <span style="display:flex;flex-direction:column;line-height:1.2;">
                <span style="font-size:12px;opacity:0.85;letter-spacing:0.05em;">EVIDENCE GRADE</span>
                <span style="font-size:20px;font-weight:600;">{label}</span>
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_citations(citations: list[dict]) -> None:
    """Render each citation as an expandable panel with source icon + snippet."""
    if not citations:
        st.info("No citations were attached to this answer.")
        return
    st.caption(f"{len(citations)} citation(s)")
    for i, citation in enumerate(citations, start=1):
        source = str(citation.get("source", "")).lower()
        icon = SOURCE_ICONS.get(source, "🔗")
        ref = citation.get("reference_id", "?")
        title = citation.get("title") or source or "citation"
        grade = str(citation.get("evidence_grade", "")).upper()
        with st.expander(f"{icon} [{i}] {title} — {source}:{ref}  ·  grade {grade}"):
            if citation.get("snippet"):
                st.write(citation["snippet"])
            cols = st.columns(3)
            cols[0].caption(f"Source: {source}")
            cols[1].caption(f"Reference: {ref}")
            if citation.get("url"):
                cols[2].caption(f"[link]({citation['url']})")
