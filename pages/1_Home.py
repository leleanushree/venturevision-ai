"""
Page 1 – Home
Project overview, business problem, dataset information.
"""

import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="VentureVision AI", page_icon="🚀", layout="wide")

# ── Shared CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        color: white;
    }
    .metric-card h2 { color: #e94560; font-size: 2rem; margin: 0; }
    .metric-card p  { color: #a8b2d8; margin: 4px 0 0 0; }
    .hero {
        background: linear-gradient(135deg, #0f3460 0%, #16213e 50%, #1a1a2e 100%);
        border-radius: 16px;
        padding: 40px;
        text-align: center;
        color: white;
        margin-bottom: 24px;
    }
    .hero h1 { font-size: 2.8rem; margin-bottom: 8px; }
    .hero p  { font-size: 1.1rem; color: #a8b2d8; }
    .pill {
        display: inline-block;
        background: #e94560;
        color: white;
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.85rem;
        margin: 3px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Hero banner ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
        <h1>🚀 VentureVision AI</h1>
        <p>Startup Outcome Prediction &amp; Decision Support System</p>
        <p>
            <span class="pill">IPO Prediction</span>
            <span class="pill">Acquisition Analysis</span>
            <span class="pill">Failure Risk</span>
            <span class="pill">AI Insights</span>
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Metrics row ───────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="metric-card"><h2>100K</h2><p>Training Samples</p></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="metric-card"><h2>3</h2><p>Outcome Classes</p></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="metric-card"><h2>4</h2><p>ML Models Trained</p></div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="metric-card"><h2>73.2%</h2><p>Best Accuracy</p></div>', unsafe_allow_html=True)

st.markdown("---")

# ── Business problem ──────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("📋 Business Problem")
    st.markdown(
        """
        Venture capital and startup ecosystems suffer from **high uncertainty**.
        Investors, founders, and analysts struggle to objectively assess which
        startups will IPO, get acquired, or fail.

        **VentureVision AI** tackles this by training ML models on historical
        startup data to predict:
        - 🟢 **IPO** — company goes public
        - 🔵 **Acquisition** — acquired by a larger company
        - 🔴 **Failure** — company shuts down

        The system also provides **explainable AI insights**, risk flags, and
        interactive analytics to support better decision-making.
        """
    )

    st.subheader("🎯 Value Proposition")
    st.markdown(
        """
        | Stakeholder | How it helps |
        |---|---|
        | 💼 VC Investor | Prioritise deal flow with data-driven risk scoring |
        | 🧑‍💼 Founder | Identify gaps & improve growth levers |
        | 🏦 Analyst | Automate startup screening and benchmarking |
        | 🎓 Researcher | Explore patterns in startup success |
        """
    )

with col_right:
    st.subheader("📊 Dataset Overview")
    st.markdown(
        """
        **Source**: Startup Funding & Outcome Dataset (Kaggle)

        | Feature | Description |
        |---|---|
        | `funding_rounds` | Number of funding rounds completed |
        | `founder_experience_years` | Years of professional experience |
        | `team_size` | Total headcount |
        | `market_size_billion` | Addressable market ($B) |
        | `product_traction_users` | Active user base |
        | `burn_rate_million` | Monthly cash burn ($M) |
        | `revenue_million` | Annual revenue ($M) |
        | `investor_type` | none / angel / tier2_vc / tier1_vc |
        | `sector` | Industry vertical |
        | `founder_background` | first_time / academic / ex_bigtech / serial |
        | `outcome` | **Target**: IPO / Acquisition / Failure |
        """
    )

st.markdown("---")

# ── Navigation guide ──────────────────────────────────────────────────────────
st.subheader("🗺️ Navigation Guide")
n1, n2, n3, n4, n5 = st.columns(5)
pages = [
    ("🏠", "Home", "You are here"),
    ("🔮", "Predictor", "Input startup details for prediction"),
    ("💡", "AI Insights", "Strengths, risks & improvement plan"),
    ("📈", "Analytics", "EDA & interactive dashboards"),
    ("🏆", "Model Performance", "Metrics, ROC, confusion matrix"),
]
for col, (icon, title, desc) in zip([n1, n2, n3, n4, n5], pages):
    with col:
        st.markdown(
            f"""
            <div style="background:#16213e;border-radius:10px;padding:14px;text-align:center;color:white;height:110px">
                <div style="font-size:1.8rem">{icon}</div>
                <div style="font-weight:bold;font-size:0.9rem">{title}</div>
                <div style="font-size:0.75rem;color:#a8b2d8;margin-top:4px">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ── Dataset preview ───────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🔍 Dataset Preview")

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "startup_success_dataset.csv")
try:
    df = pd.read_csv(DATA_PATH, nrows=500)
    st.dataframe(df.head(10), use_container_width=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Records", f"{len(pd.read_csv(DATA_PATH)):,}")
    c2.metric("Features", str(df.shape[1] - 1))
    c3.metric("Target Classes", "3")
except FileNotFoundError:
    st.warning("Dataset file not found. Please ensure `data/startup_success_dataset.csv` exists.")