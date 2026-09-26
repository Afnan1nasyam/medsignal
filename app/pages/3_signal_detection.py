"""Page: Signal Detection — emerging safety signals not yet on labels."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

from components.api import api_call

st.set_page_config(page_title="Signal Detection", page_icon="🚨", layout="wide")

st.title("🚨 Signal Detection")
st.caption(
    "Potential emerging signals: drug → adverse-event pairs with high FAERS report "
    "counts that are **not yet reflected in the drug label**."
)

_HIGH = 50  # FAERS count considered a strong signal

result = api_call("/api/analytics/signals")
if not result.get("ok"):
    st.warning(f"{result.get('error')}: {result.get('detail', '')}")
    st.stop()

signals = result["data"].get("signals", [])
if not signals:
    st.info("No signals found. Ingest data first: `python scripts/ingest_all.py`.")
    st.stop()

# Controls
c1, c2 = st.columns([1, 3])
with c1:
    only_unlabeled = st.checkbox("Only not-in-label", value=True)
    min_count = st.slider("Min FAERS count", 1, 200, 1)

rows = [
    s for s in signals
    if s["faers_count"] >= min_count and (not only_unlabeled or not s["in_label"])
]

# Summary metrics
m1, m2, m3 = st.columns(3)
m1.metric("Signals shown", len(rows))
m2.metric("High + unlabeled", sum(1 for s in rows if s["faers_count"] >= _HIGH and not s["in_label"]))
m3.metric("Total pairs scanned", len(signals))


def _row_style(row: pd.Series):
    """Red for high FAERS + not in label; amber for moderate unlabeled."""
    high_unlabeled = row["FAERS count"] >= _HIGH and row["In label?"] == "No"
    moderate_unlabeled = row["FAERS count"] >= 10 and row["In label?"] == "No"
    if high_unlabeled:
        return ["background-color: #fdecea"] * len(row)
    if moderate_unlabeled:
        return ["background-color: #fef7e6"] * len(row)
    return [""] * len(row)


df = pd.DataFrame(
    [
        {
            "Drug": s["drug"],
            "Adverse event": s["adverse_event"],
            "FAERS count": s["faers_count"],
            "In label?": "Yes" if s["in_label"] else "No",
            "Publications": s["publications"],
        }
        for s in rows
    ]
)

if df.empty:
    st.info("No signals match the current filters.")
else:
    st.dataframe(
        df.style.apply(_row_style, axis=1),
        hide_index=True,
        use_container_width=True,
    )
    st.caption("🔴 high FAERS count & not in label · 🟡 moderate (≥10) & not in label")
