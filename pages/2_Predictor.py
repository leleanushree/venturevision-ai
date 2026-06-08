"""
Page 2 – Startup Predictor
Input startup details, get IPO / Acquisition / Failure probabilities.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.preprocessing import load_transformers, preprocess_single_input
from utils.model_utils import load_model, predict_proba

st.set_page_config(page_title="Startup Predictor", page_icon="🔮", layout="wide")

st.markdown(
    """
    <style>
    .prob-card {
        border-radius: 14px;
        padding: 20px;
        text-align: center;
        color: white;
        font-size: 1.1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Load model & transformers (cached) ───────────────────────────────────────
@st.cache_resource
def get_model_and_transformers():
    model = load_model()
    transformers = load_transformers()
    return model, transformers

try:
    model, transformers = get_model_and_transformers()
except FileNotFoundError:
    st.error("⚠️ Model artefacts not found. Run `python train_model.py` first.")
    st.stop()

# ── Page header ───────────────────────────────────────────────────────────────
st.title("🔮 Startup Outcome Predictor")
st.markdown("Fill in the startup profile below to receive an AI-powered outcome prediction.")

st.markdown("---")

# ── Input form ────────────────────────────────────────────────────────────────
with st.form("predictor_form"):
    st.subheader("📋 Startup Profile")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**🔢 Numeric Factors**")
        founder_exp = st.slider("Founder Experience (years)", 0, 24, 10)
        team_size = st.slider("Team Size", 2, 300, 50)
        funding_rounds = st.slider("Funding Rounds", 0, 8, 2)

    with col2:
        st.markdown("**💰 Financial Factors**")
        revenue = st.number_input("Annual Revenue ($M)", min_value=0.0, max_value=5000.0, value=500.0, step=10.0)
        burn_rate = st.number_input("Monthly Burn Rate ($M)", min_value=0.0, max_value=400.0, value=15.0, step=0.5)
        market_size = st.number_input("Market Size ($B)", min_value=0.0, max_value=1100.0, value=20.0, step=1.0)

    with col3:
        st.markdown("**📊 Traction & Background**")
        traction = st.number_input("Product Traction (users)", min_value=0, max_value=1_000_000, value=100_000, step=5_000)
        investor_type = st.selectbox("Investor Type", ["none", "angel", "tier2_vc", "tier1_vc"])
        sector = st.selectbox("Sector", ["AI", "Climate", "Crypto", "Ecommerce", "Fintech", "Health", "SaaS"])
        founder_bg = st.selectbox("Founder Background", ["first_time", "academic", "ex_bigtech", "serial_founder"])

    submitted = st.form_submit_button("🚀 Predict Outcome", use_container_width=True)

# ── Prediction ────────────────────────────────────────────────────────────────
if submitted:
    input_dict = {
        "funding_rounds": funding_rounds,
        "founder_experience_years": founder_exp,
        "team_size": team_size,
        "market_size_billion": market_size,
        "product_traction_users": traction,
        "burn_rate_million": burn_rate,
        "revenue_million": revenue,
        "investor_type": investor_type,
        "sector": sector,
        "founder_background": founder_bg,
    }

    X = preprocess_single_input(input_dict, transformers)
    proba = predict_proba(model, X, transformers["label_encoder"])

    # Sort by probability descending
    sorted_proba = sorted(proba.items(), key=lambda x: x[1], reverse=True)
    top_class = sorted_proba[0][0]

    st.markdown("---")
    st.subheader("🎯 Prediction Results")

    # ── Top outcome banner ────────────────────────────────────────────────────
    colour_map = {"IPO": "#2ecc71", "Acquisition": "#3498db", "Failure": "#e74c3c"}
    emoji_map  = {"IPO": "🟢", "Acquisition": "🔵", "Failure": "🔴"}

    st.markdown(
        f"""
        <div style="background:{colour_map[top_class]};border-radius:14px;
                    padding:20px;text-align:center;color:white;margin-bottom:20px">
            <h2 style="margin:0">{emoji_map[top_class]} Predicted Outcome: {top_class}</h2>
            <p style="margin:4px 0 0 0;font-size:1.1rem">
                Confidence: {proba[top_class]:.1%}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Probability cards ─────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    for col, (cls, prob) in zip([c1, c2, c3], [
        ("IPO", proba.get("IPO", 0)),
        ("Acquisition", proba.get("Acquisition", 0)),
        ("Failure", proba.get("Failure", 0)),
    ]):
        with col:
            colour = colour_map[cls]
            st.markdown(
                f"""
                <div class="prob-card" style="background:{colour}22;border:2px solid {colour}">
                    <div style="font-size:2rem">{emoji_map[cls]}</div>
                    <div style="font-weight:bold;color:{colour}">{cls}</div>
                    <div style="font-size:2rem;font-weight:bold;color:{colour}">{prob:.1%}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(float(prob))

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Gauge-style Plotly chart ──────────────────────────────────────────────
    col_chart, col_bar = st.columns([1, 1])

    with col_chart:
        labels = list(proba.keys())
        values = list(proba.values())
        colours = [colour_map[l] for l in labels]

        fig_pie = go.Figure(
            go.Pie(
                labels=labels,
                values=values,
                hole=0.55,
                marker_colors=colours,
                textinfo="label+percent",
            )
        )
        fig_pie.update_layout(
            title="Probability Distribution",
            annotations=[
                dict(text=f"{top_class}<br>{proba[top_class]:.0%}", x=0.5, y=0.5,
                     font_size=15, showarrow=False)
            ],
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_bar:
        fig_bar = go.Figure(
            go.Bar(
                x=list(proba.values()),
                y=list(proba.keys()),
                orientation="h",
                marker_color=[colour_map[l] for l in proba.keys()],
                text=[f"{v:.1%}" for v in proba.values()],
                textposition="outside",
            )
        )
        fig_bar.update_layout(
            title="Outcome Probabilities",
            xaxis=dict(tickformat=".0%", range=[0, 1]),
            yaxis_title=None,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Save to session state for AI Insights page ────────────────────────────
    st.session_state["last_input"] = input_dict
    st.session_state["last_proba"] = proba
    st.info("💡 Head to the **AI Insights** page to see strengths, risks, and recommendations!")