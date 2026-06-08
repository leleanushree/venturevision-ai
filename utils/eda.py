"""
eda.py
All EDA / visualisation helpers (Plotly-based for Streamlit embedding).
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Consistent colour palette
PALETTE = {
    "IPO": "#2ecc71",
    "Acquisition": "#3498db",
    "Failure": "#e74c3c",
}
SECTOR_COLOR = px.colors.qualitative.Bold


# ── Outcome distribution ─────────────────────────────────────────────────────

def plot_outcome_distribution(df: pd.DataFrame) -> go.Figure:
    counts = df["outcome"].value_counts().reset_index()
    counts.columns = ["Outcome", "Count"]
    fig = px.pie(
        counts,
        names="Outcome",
        values="Count",
        title="Startup Outcome Distribution",
        color="Outcome",
        color_discrete_map=PALETTE,
        hole=0.45,
    )
    fig.update_traces(textposition="outside", textinfo="percent+label")
    fig.update_layout(showlegend=True, title_font_size=18)
    return fig


# ── Sector vs Outcome ────────────────────────────────────────────────────────

def plot_sector_vs_outcome(df: pd.DataFrame) -> go.Figure:
    pivot = (
        df.groupby(["sector", "outcome"])
        .size()
        .reset_index(name="count")
    )
    fig = px.bar(
        pivot,
        x="sector",
        y="count",
        color="outcome",
        barmode="group",
        title="Sector vs Startup Outcome",
        color_discrete_map=PALETTE,
        labels={"sector": "Sector", "count": "Number of Startups", "outcome": "Outcome"},
    )
    fig.update_layout(xaxis_tickangle=-30, title_font_size=18)
    return fig


# ── Funding rounds vs Outcome ────────────────────────────────────────────────

def plot_funding_vs_outcome(df: pd.DataFrame) -> go.Figure:
    fig = px.box(
        df,
        x="outcome",
        y="funding_rounds",
        color="outcome",
        title="Funding Rounds by Outcome",
        color_discrete_map=PALETTE,
        labels={"outcome": "Outcome", "funding_rounds": "Funding Rounds"},
        points="outliers",
    )
    fig.update_layout(showlegend=False, title_font_size=18)
    return fig


# ── Revenue vs Outcome ───────────────────────────────────────────────────────

def plot_revenue_vs_outcome(df: pd.DataFrame) -> go.Figure:
    df2 = df.copy()
    df2["revenue_log"] = np.log1p(df2["revenue_million"])
    fig = px.violin(
        df2,
        x="outcome",
        y="revenue_log",
        color="outcome",
        box=True,
        title="Revenue Distribution by Outcome (log scale)",
        color_discrete_map=PALETTE,
        labels={"outcome": "Outcome", "revenue_log": "log(Revenue $M + 1)"},
    )
    fig.update_layout(showlegend=False, title_font_size=18)
    return fig


# ── Correlation heatmap ───────────────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    numeric_cols = [
        "funding_rounds", "founder_experience_years", "team_size",
        "market_size_billion", "product_traction_users",
        "burn_rate_million", "revenue_million",
    ]
    corr = df[numeric_cols].corr()
    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        title="Feature Correlation Heatmap",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
    )
    fig.update_layout(title_font_size=18)
    return fig


# ── Sector success rates (for analytics dashboard) ───────────────────────────

def plot_sector_success_rate(df: pd.DataFrame) -> go.Figure:
    total = df.groupby("sector").size().rename("total")
    ipo = df[df["outcome"] == "IPO"].groupby("sector").size().rename("ipo")
    acq = df[df["outcome"] == "Acquisition"].groupby("sector").size().rename("acq")
    tbl = pd.concat([total, ipo, acq], axis=1).fillna(0).reset_index()
    tbl["IPO Rate %"] = (tbl["ipo"] / tbl["total"] * 100).round(1)
    tbl["Acquisition Rate %"] = (tbl["acq"] / tbl["total"] * 100).round(1)
    tbl["Failure Rate %"] = (100 - tbl["IPO Rate %"] - tbl["Acquisition Rate %"]).round(1)

    fig = go.Figure()
    for col, colour in [("IPO Rate %", PALETTE["IPO"]),
                         ("Acquisition Rate %", PALETTE["Acquisition"]),
                         ("Failure Rate %", PALETTE["Failure"])]:
        fig.add_trace(go.Bar(name=col, x=tbl["sector"], y=tbl[col], marker_color=colour))

    fig.update_layout(
        barmode="stack",
        title="Success / Failure Rates by Sector (%)",
        xaxis_title="Sector",
        yaxis_title="Percentage",
        title_font_size=18,
    )
    return fig


# ── Funding rounds distribution ──────────────────────────────────────────────

def plot_funding_distribution(df: pd.DataFrame) -> go.Figure:
    fig = px.histogram(
        df,
        x="funding_rounds",
        color="outcome",
        barmode="overlay",
        title="Funding Rounds Distribution by Outcome",
        color_discrete_map=PALETTE,
        nbins=9,
        opacity=0.75,
        labels={"funding_rounds": "Funding Rounds", "outcome": "Outcome"},
    )
    fig.update_layout(title_font_size=18)
    return fig


# ── Revenue by sector ────────────────────────────────────────────────────────

def plot_revenue_by_sector(df: pd.DataFrame) -> go.Figure:
    grp = df.groupby("sector")["revenue_million"].median().reset_index()
    grp.columns = ["Sector", "Median Revenue ($M)"]
    grp = grp.sort_values("Median Revenue ($M)", ascending=False)
    fig = px.bar(
        grp,
        x="Sector",
        y="Median Revenue ($M)",
        title="Median Revenue by Sector",
        color="Sector",
        color_discrete_sequence=SECTOR_COLOR,
    )
    fig.update_layout(showlegend=False, title_font_size=18)
    return fig


# ── Burn rate vs outcome ─────────────────────────────────────────────────────

def plot_burn_rate(df: pd.DataFrame) -> go.Figure:
    fig = px.box(
        df,
        x="outcome",
        y="burn_rate_million",
        color="outcome",
        title="Burn Rate by Outcome ($M / month)",
        color_discrete_map=PALETTE,
        labels={"outcome": "Outcome", "burn_rate_million": "Burn Rate ($M)"},
        points="outliers",
    )
    fig.update_layout(showlegend=False, title_font_size=18)
    return fig
