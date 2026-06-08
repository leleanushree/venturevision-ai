"""
Page 4 – Analytics Dashboard
Interactive Plotly dashboards: outcome trends, sector success, revenue, funding.
"""

import streamlit as st
import pandas as pd
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.eda import (
    plot_outcome_distribution,
    plot_sector_vs_outcome,
    plot_funding_vs_outcome,
    plot_revenue_vs_outcome,
    plot_correlation_heatmap,
    plot_sector_success_rate,
    plot_funding_distribution,
    plot_revenue_by_sector,
    plot_burn_rate,
)

st.set_page_config(page_title="Analytics Dashboard", page_icon="📈", layout="wide")

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "startup_success_dataset.csv")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

try:
    df = load_data()
except FileNotFoundError:
    st.error("⚠️ Dataset file not found.")
    st.stop()

st.title("📈 Analytics Dashboard")
st.markdown("Explore patterns across **100,000 startups** with interactive visualisations.")
st.markdown("---")

# ── Filters sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔧 Filters")
    sectors = st.multiselect("Sector", df["sector"].unique().tolist(), default=df["sector"].unique().tolist())
    outcomes = st.multiselect("Outcome", ["IPO", "Acquisition", "Failure"], default=["IPO", "Acquisition", "Failure"])
    inv_types = st.multiselect("Investor Type", df["investor_type"].unique().tolist(), default=df["investor_type"].unique().tolist())

dff = df[
    df["sector"].isin(sectors) &
    df["outcome"].isin(outcomes) &
    df["investor_type"].isin(inv_types)
]

st.markdown(f"**Showing {len(dff):,} startups** matching filters.")

# ── Row 1: KPIs ───────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Startups", f"{len(dff):,}")
k2.metric("IPO Rate",         f"{(dff['outcome']=='IPO').mean():.1%}")
k3.metric("Acquisition Rate", f"{(dff['outcome']=='Acquisition').mean():.1%}")
k4.metric("Failure Rate",     f"{(dff['outcome']=='Failure').mean():.1%}")
k5.metric("Avg Revenue ($M)", f"{dff['revenue_million'].median():,.0f}")

st.markdown("---")

# ── Row 2: Outcome dist + Sector vs outcome ───────────────────────────────────
r2c1, r2c2 = st.columns(2)
with r2c1:
    st.plotly_chart(plot_outcome_distribution(dff), use_container_width=True)
with r2c2:
    st.plotly_chart(plot_sector_vs_outcome(dff), use_container_width=True)

# ── Row 3: Sector success rates + Funding dist ───────────────────────────────
r3c1, r3c2 = st.columns(2)
with r3c1:
    st.plotly_chart(plot_sector_success_rate(dff), use_container_width=True)
with r3c2:
    st.plotly_chart(plot_funding_distribution(dff), use_container_width=True)

# ── Row 4: Revenue + Burn rate ────────────────────────────────────────────────
r4c1, r4c2 = st.columns(2)
with r4c1:
    st.plotly_chart(plot_revenue_vs_outcome(dff), use_container_width=True)
with r4c2:
    st.plotly_chart(plot_burn_rate(dff), use_container_width=True)

# ── Row 5: Revenue by sector + Funding by outcome ────────────────────────────
r5c1, r5c2 = st.columns(2)
with r5c1:
    st.plotly_chart(plot_revenue_by_sector(dff), use_container_width=True)
with r5c2:
    st.plotly_chart(plot_funding_vs_outcome(dff), use_container_width=True)

# ── Row 6: Correlation heatmap ────────────────────────────────────────────────
st.plotly_chart(plot_correlation_heatmap(dff), use_container_width=True)

# ── Raw data explorer ─────────────────────────────────────────────────────────
with st.expander("🔍 Raw Data Explorer"):
    st.dataframe(dff.sample(min(200, len(dff))).reset_index(drop=True), use_container_width=True)
    st.download_button(
        "⬇️ Download Filtered Data",
        dff.to_csv(index=False),
        file_name="filtered_startups.csv",
        mime="text/csv",
    )