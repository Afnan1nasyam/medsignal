"""Shared visual theme — validated colorblind-safe palette and Plotly styling.

Palette values come from the data-viz reference palette (validated for CVD and
contrast). Categorical hues are used in fixed order and never cycled; magnitude
charts use a single blue hue; evidence grades and interaction severities use the
reserved status palette and always ship with a text label (never color alone).
Knowledge-graph node colors follow the project's specified scheme.
"""

from __future__ import annotations

# Categorical palette (fixed order: blue, orange, aqua, yellow, magenta, green, violet, red).
CATEGORICAL = [
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#4a3aa7", "#e34948",
]
# Single-hue sequential ramp (light -> dark) for magnitude encodings.
SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
BAR_BLUE = "#2a78d6"

# Reserved status palette (never reused as a series color).
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}

# Evidence grade -> (color, label). Ordered severity; always shown with the letter.
GRADE_COLORS = {"A": "#0ca30c", "B": "#2a78d6", "C": "#eda100", "D": "#ec835a", "E": "#d03b3b"}
GRADE_LABELS = {"A": "Strong", "B": "Moderate", "C": "Suggestive", "D": "Weak", "E": "Insufficient"}

# Interaction severity -> color (status palette).
SEVERITY_COLORS = {"major": "#d03b3b", "moderate": "#fab219", "minor": "#0ca30c"}

# Knowledge-graph node type -> color (project scheme).
NODE_COLORS = {
    "drug": "#4A90D9",
    "adverse_event": "#E74C3C",
    "condition": "#27AE60",
    "clinical_trial": "#9B59B6",
    "publication": "#F39C12",
    "drug_class": "#1ABC9C",
}
NODE_LABELS = {
    "drug": "Drug",
    "adverse_event": "Adverse Event",
    "condition": "Condition",
    "clinical_trial": "Clinical Trial",
    "publication": "Publication",
    "drug_class": "Drug Class",
}

# Source type -> emoji icon (for citations).
SOURCE_ICONS = {
    "drug_labels": "🏷️",
    "clinical_trials": "🧪",
    "pubmed": "📄",
    "faers": "⚠️",
}

# Chart chrome / ink (light surface).
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"


def style_fig(fig, height: int = 380, showlegend: bool = False):
    """Apply the shared, recessive Plotly layout to a figure and return it.

    Single-axis, hairline grid, muted ticks, tight margins. Set
    ``showlegend=True`` only for charts with two or more series.
    """
    fig.update_layout(
        font=dict(family=FONT, color=INK_PRIMARY, size=13),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        margin=dict(l=10, r=24, t=44, b=10),
        height=height,
        showlegend=showlegend,
        title=dict(font=dict(size=15, color=INK_PRIMARY)),
        legend=dict(font=dict(color=INK_PRIMARY), bgcolor="rgba(0,0,0,0)"),
    )
    axis = dict(
        gridcolor=GRID,
        linecolor=BASELINE,
        zerolinecolor=BASELINE,
        tickfont=dict(color=INK_MUTED),
        title=dict(font=dict(color=INK_MUTED, size=12)),
    )
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig
