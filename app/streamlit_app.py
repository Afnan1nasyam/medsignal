"""MedSignal Streamlit front-end (home page).

Calls the FastAPI backend for all data. Heavy work lives server-side; this app
only renders. If the API is down, every page degrades to a clear warning.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the app package dir is importable for `components.*` (pages too).
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from components.api import API_BASE, api_call

st.set_page_config(page_title="MedSignal", page_icon="💊", layout="wide")


def render_sidebar() -> None:
    """Render API connection status and live system stats in the sidebar."""
    st.sidebar.title("💊 MedSignal")
    st.sidebar.caption("Drug Safety Intelligence")

    health = api_call("/", timeout=5.0)
    if health.get("ok"):
        st.sidebar.success(f"API connected · {API_BASE}")
        stats = api_call("/api/ingest/status", timeout=10.0)
        if stats.get("ok"):
            data = stats["data"]
            sql = data.get("sql", {})
            graph = data.get("graph", {})
            st.sidebar.markdown("**System stats**")
            st.sidebar.metric("FAERS reports", sql.get("faers_reports", 0))
            st.sidebar.metric("Graph nodes", graph.get("total_nodes", 0))
            st.sidebar.metric("Graph edges", graph.get("total_edges", 0))
            st.sidebar.caption(
                f"PubMed {sql.get('pubmed_articles', 0)} · "
                f"Trials {sql.get('clinical_trials', 0)} · "
                f"Labels {sql.get('drug_labels', 0)}"
            )
    else:
        st.sidebar.error("API not connected")
        st.sidebar.caption(health.get("detail", "Start the backend to load data."))
        st.sidebar.code("uvicorn api.main:app --port 8000", language="bash")


def main() -> None:
    """Render the home page."""
    render_sidebar()

    st.title("MedSignal — Drug Safety Intelligence")
    st.markdown(
        "A multi-source **agentic RAG** system for pharmacovigilance. MedSignal "
        "ingests FDA FAERS, PubMed, ClinicalTrials.gov, and DailyMed labels into a "
        "vector store, a SQL store, and a biomedical **knowledge graph**, then a "
        "LangGraph agent plans retrieval across sources, evaluates evidence, and "
        "answers with an **A–E evidence grade**."
    )

    st.subheader("Explore")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### 🔍 Safety Query")
        st.caption("Ask a natural-language drug-safety question and get a graded, cited answer.")
        st.markdown("#### 💊 Drug Explorer")
        st.caption("Aggregated safety profile, interactions, and related drugs for any drug.")
    with c2:
        st.markdown("#### 🚨 Signal Detection")
        st.caption("Potential emerging signals: high FAERS counts not yet on the label.")
        st.markdown("#### 🕸️ Knowledge Graph")
        st.caption("Interactive exploration of the drug/adverse-event/trial graph.")
    with c3:
        st.markdown("#### 📊 Analytics")
        st.caption("Top drugs and adverse events, outcome distribution, and query stats.")

    st.info("Use the sidebar to navigate between pages. Data requires the API to be running and populated (`python scripts/seed_sample_data.py` then `python scripts/ingest_all.py`).")


if __name__ == "__main__":
    main()
else:
    main()
