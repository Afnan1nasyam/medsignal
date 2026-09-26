"""Page: Safety Query — run the agentic RAG engine over a question."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from components.api import api_call
from components.evidence_card import grade_badge, render_citations

st.set_page_config(page_title="Safety Query", page_icon="🔍", layout="wide")

_EXAMPLES = [
    "Is metformin linked to lactic acidosis?",
    "Drug interactions for warfarin in elderly",
    "Compare safety of atorvastatin vs simvastatin",
    "Emerging signals for GLP-1 agonists",
]

st.title("🔍 Safety Query")
st.caption("Natural-language question → multi-source agentic retrieval → graded, cited answer.")

if "query_text" not in st.session_state:
    st.session_state.query_text = ""

st.markdown("**Example queries**")
cols = st.columns(len(_EXAMPLES))
for col, example in zip(cols, _EXAMPLES):
    if col.button(example, use_container_width=True):
        st.session_state.query_text = example

query = st.text_area("Your question", value=st.session_state.query_text, height=90, key="query_input")
run = st.button("Search", type="primary")

if run and query.strip():
    with st.spinner("Running the agent across FAERS, PubMed, trials, labels, and the graph..."):
        result = api_call("/api/query", method="POST", data={"query": query.strip()})

    if not result.get("ok"):
        st.error(f"{result.get('error')}: {result.get('detail', '')}")
    else:
        payload = result["data"]
        agent = payload.get("result", {})
        proc_time = payload.get("processing_time_seconds", 0.0)

        top = st.columns([1, 3])
        with top[0]:
            grade_badge(agent.get("evidence_grade", "E"))
        with top[1]:
            st.metric("Processing time", f"{proc_time:.2f} s")
            st.caption(f"Sources consulted: {', '.join(agent.get('sources_consulted', [])) or 'none'}")

        st.subheader("Answer")
        st.markdown(agent.get("answer", "_No answer produced._"))

        st.subheader("Citations")
        render_citations(agent.get("citations", []))

        with st.expander("🔬 Agent trace (plan, sub-queries, iterations)"):
            plan = agent.get("query_plan") or {}
            st.write(f"**Intent:** {plan.get('intent', 'n/a')}")
            st.write(f"**Iterations used:** {agent.get('iterations_used', 0)}")
            st.write(f"**Drugs detected:** {', '.join(plan.get('drugs_mentioned', [])) or 'none'}")
            st.write(f"**Adverse events detected:** {', '.join(plan.get('adverse_events_mentioned', [])) or 'none'}")
            st.markdown("**Sub-queries**")
            for i, sub in enumerate(plan.get("sub_queries", []), start=1):
                st.markdown(
                    f"{i}. `{', '.join(sub.get('target_sources', []))}` — {sub.get('query', '')}  \n"
                    f"   _{sub.get('reasoning', '')}_"
                )
elif run:
    st.warning("Please enter a question first.")
