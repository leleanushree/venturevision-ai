"""
Page 3 – AI Insights
Feature-importance-driven strengths, risks, and improvement recommendations.
"""

import streamlit as st
import os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.preprocessing import load_transformers, preprocess_single_input
from utils.model_utils import (
    load_model, predict_proba, generate_insights,
    get_feature_importance, plot_feature_importance,
)

st.set_page_config(page_title="AI Insights", page_icon="💡", layout="wide")

# ── Load resources ─────────────────────────────────────────────────────────────
@st.cache_resource
def get_resources():
    model = load_model()
    transformers = load_transformers()
    return model, transformers

try:
    model, transformers = get_resources()
except FileNotFoundError:
    st.error("⚠️ Model artefacts not found. Run `python train_model.py` first.")
    st.stop()

st.title("💡 AI Insights")
st.markdown("Understand **why** a prediction was made and **what to improve**.")
st.markdown("---")

# ── Require predictor to be run first, or allow re-entry ─────────────────────
last_input = st.session_state.get("last_input", None)

if last_input is None:
    st.info("ℹ️ Run the **Predictor** first to auto-fill your last startup profile — or fill one in below.")

with st.expander("📋 Enter / Review Startup Profile", expanded=(last_input is None)):
    col1, col2, col3 = st.columns(3)
    defaults = last_input or {
        "funding_rounds": 2, "founder_experience_years": 10, "team_size": 50,
        "market_size_billion": 20.0, "product_traction_users": 100_000,
        "burn_rate_million": 15.0, "revenue_million": 500.0,
        "investor_type": "tier2_vc", "sector": "SaaS", "founder_background": "ex_bigtech",
    }
    with col1:
        founder_exp   = st.slider("Founder Experience (yrs)", 0, 24, int(defaults["founder_experience_years"]), key="ins_exp")
        team_size     = st.slider("Team Size", 2, 300, int(defaults["team_size"]), key="ins_ts")
        funding_rounds = st.slider("Funding Rounds", 0, 8, int(defaults["funding_rounds"]), key="ins_fr")
    with col2:
        revenue    = st.number_input("Revenue ($M)", 0.0, 5000.0, float(defaults["revenue_million"]), 10.0, key="ins_rev")
        burn_rate  = st.number_input("Burn Rate ($M/mo)", 0.0, 400.0, float(defaults["burn_rate_million"]), 0.5, key="ins_br")
        market_size = st.number_input("Market Size ($B)", 0.0, 1100.0, float(defaults["market_size_billion"]), 1.0, key="ins_ms")
    with col3:
        traction    = st.number_input("Traction (users)", 0, 1_000_000, int(defaults["product_traction_users"]), 5_000, key="ins_tr")
        investor    = st.selectbox("Investor Type", ["none","angel","tier2_vc","tier1_vc"],
                                   index=["none","angel","tier2_vc","tier1_vc"].index(defaults["investor_type"]), key="ins_inv")
        sector      = st.selectbox("Sector", ["AI","Climate","Crypto","Ecommerce","Fintech","Health","SaaS"],
                                   index=["AI","Climate","Crypto","Ecommerce","Fintech","Health","SaaS"].index(defaults["sector"]), key="ins_sec")
        bg          = st.selectbox("Founder Background", ["first_time","academic","ex_bigtech","serial_founder"],
                                   index=["first_time","academic","ex_bigtech","serial_founder"].index(defaults["founder_background"]), key="ins_bg")

input_dict = {
    "funding_rounds": funding_rounds, "founder_experience_years": founder_exp,
    "team_size": team_size, "market_size_billion": market_size,
    "product_traction_users": traction, "burn_rate_million": burn_rate,
    "revenue_million": revenue, "investor_type": investor,
    "sector": sector, "founder_background": bg,
}

# ── Predict & generate insights ───────────────────────────────────────────────
X = preprocess_single_input(input_dict, transformers)
proba = predict_proba(model, X, transformers["label_encoder"])
insights = generate_insights(input_dict, proba, model, transformers["feature_names"])

top_class = max(proba, key=proba.get)
colour_map = {"IPO": "#2ecc71", "Acquisition": "#3498db", "Failure": "#e74c3c"}

st.markdown(
    f"""
    <div style="background:{colour_map[top_class]}22;border:2px solid {colour_map[top_class]};
                border-radius:12px;padding:16px;text-align:center;margin-bottom:20px">
        <b style="color:{colour_map[top_class]};font-size:1.2rem">
            Current Prediction: {top_class} ({proba[top_class]:.1%} confidence)
        </b>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── 3-column insight cards ────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("### ✅ Strengths")
    if insights["strengths"]:
        for s in insights["strengths"]:
            st.success(f"✓ {s}")
    else:
        st.info("No notable strengths detected.")

with c2:
    st.markdown("### ⚠️ Risks")
    if insights["risks"]:
        for r in insights["risks"]:
            st.error(f"⚠ {r}")
    else:
        st.success("No major risks identified!")

with c3:
    st.markdown("### 🔧 Improvements")
    if insights["improvements"]:
        for i, tip in enumerate(insights["improvements"], 1):
            st.warning(f"{i}. {tip}")
    else:
        st.success("Profile looks strong!")

st.markdown("---")

# ── Feature importance chart ──────────────────────────────────────────────────
st.subheader("📊 Feature Importance (Model-Level)")
st.markdown("The chart below shows which features **globally** drive the model's decisions.")
fig = plot_feature_importance(model, transformers["feature_names"])
st.plotly_chart(fig, use_container_width=True)

# ── Probability breakdown table ───────────────────────────────────────────────
st.subheader("📋 Probability Breakdown")
import pandas as pd
prob_df = pd.DataFrame(
    [{"Outcome": k, "Probability": f"{v:.2%}", "Bar": v} for k, v in sorted(proba.items(), key=lambda x: -x[1])]
)
st.dataframe(
    prob_df[["Outcome", "Probability"]],
    use_container_width=True,
    hide_index=True,
)
