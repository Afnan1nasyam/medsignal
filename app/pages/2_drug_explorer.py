"""Page: Drug Explorer — per-drug safety profile from the backend."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from components.api import api_call
from components.drug_profile import render_profile

st.set_page_config(page_title="Drug Explorer", page_icon="💊", layout="wide")

st.title("💊 Drug Explorer")
st.caption("Aggregated safety intelligence for a single drug, from the knowledge graph and labels.")

# Populate a selector from the graph's known drugs; allow free text too.
drugs_result = api_call("/api/drugs")
known = drugs_result.get("data", {}).get("drugs", []) if drugs_result.get("ok") else []

if not drugs_result.get("ok"):
    st.warning(f"{drugs_result.get('error')}: {drugs_result.get('detail', '')}")

col1, col2 = st.columns([2, 1])
with col1:
    selected = st.selectbox(
        "Select a drug", options=["—"] + known, index=0,
        help="Drugs currently present in the knowledge graph.",
    )
with col2:
    typed = st.text_input("...or type a drug name", value="")

drug_name = typed.strip() or (selected if selected != "—" else "")

if drug_name:
    result = api_call(f"/api/drugs/{drug_name}")
    if not result.get("ok"):
        st.error(f"{result.get('error')}: {result.get('detail', '')}")
    else:
        data = result["data"]
        render_profile(data.get("profile", {}), data.get("graph_neighbors", []))
else:
    st.info("Choose or type a drug to view its safety profile.")
