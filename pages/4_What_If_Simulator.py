"""
Page 4 – What-If Simulator
Modify startup parameters and compare current vs modified predictions.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.preprocessing import load_transformers, preprocess_single_input
from utils.model_utils import load_model, predict_proba

st.set_page_config(page_title="What-If Simulator", page_icon="🧪", layout="wide")

@st.cache_resource
def get_resources():
    return load_model(), load_transformers()

try:
    model, transformers = get_resources()
except FileNotFoundError:
    st.error("⚠️ Model artefacts not found. Run `python train_model.py` first.")
    st.stop()

st.title("🧪 What-If Simulator")
st.markdown(
    "Compare your **baseline** startup profile against a **modified** scenario "
    "to see how changes affect the predicted outcome."
)
st.markdown("---")

# ── Default profile ───────────────────────────────────────────────────────────
last_input = st.session_state.get("last_input", None)
defaults = last_input or {
    "funding_rounds": 2, "founder_experience_years": 10, "team_size": 50,
    "market_size_billion": 20.0, "product_traction_users": 100_000,
    "burn_rate_million": 15.0, "revenue_million": 500.0,
    "investor_type": "tier2_vc", "sector": "SaaS", "founder_background": "ex_bigtech",
}

# ── Two-column layout: baseline | modified ────────────────────────────────────
col_base, col_mod = st.columns(2, gap="large")

with col_base:
    st.subheader("📌 Baseline Profile")
    b_fr   = st.slider("Funding Rounds",         0,  8,   int(defaults["funding_rounds"]),            key="b_fr")
    b_exp  = st.slider("Founder Experience (yrs)",0, 24,  int(defaults["founder_experience_years"]),  key="b_exp")
    b_ts   = st.slider("Team Size",               2, 300, int(defaults["team_size"]),                 key="b_ts")
    b_ms   = st.number_input("Market Size ($B)",      0.0, 1100.0, float(defaults["market_size_billion"]),  1.0, key="b_ms")
    b_tr   = st.number_input("Traction (users)",       0, 1_000_000, int(defaults["product_traction_users"]), 5_000, key="b_tr")
    b_br   = st.number_input("Burn Rate ($M/mo)",     0.0, 400.0, float(defaults["burn_rate_million"]),    0.5, key="b_br")
    b_rev  = st.number_input("Revenue ($M)",           0.0, 5000.0, float(defaults["revenue_million"]),    10.0, key="b_rev")
    b_inv  = st.selectbox("Investor Type",["none","angel","tier2_vc","tier1_vc"],
                          index=["none","angel","tier2_vc","tier1_vc"].index(defaults["investor_type"]), key="b_inv")
    b_sec  = st.selectbox("Sector",["AI","Climate","Crypto","Ecommerce","Fintech","Health","SaaS"],
                          index=["AI","Climate","Crypto","Ecommerce","Fintech","Health","SaaS"].index(defaults["sector"]), key="b_sec")
    b_bg   = st.selectbox("Founder Background",["first_time","academic","ex_bigtech","serial_founder"],
                          index=["first_time","academic","ex_bigtech","serial_founder"].index(defaults["founder_background"]), key="b_bg")

with col_mod:
    st.subheader("✏️ Modified Profile")
    m_fr   = st.slider("Funding Rounds",         0,  8,   min(int(defaults["funding_rounds"]) + 1, 8), key="m_fr")
    m_exp  = st.slider("Founder Experience (yrs)",0, 24,  min(int(defaults["founder_experience_years"]) + 3, 24), key="m_exp")
    m_ts   = st.slider("Team Size",               2, 300, min(int(defaults["team_size"]) + 20, 300),  key="m_ts")
    m_ms   = st.number_input("Market Size ($B)",      0.0, 1100.0, float(defaults["market_size_billion"]),  1.0, key="m_ms")
    m_tr   = st.number_input("Traction (users)",       0, 1_000_000, min(int(defaults["product_traction_users"]) + 50_000, 1_000_000), 5_000, key="m_tr")
    m_br   = st.number_input("Burn Rate ($M/mo)",     0.0, 400.0, max(float(defaults["burn_rate_million"]) - 3.0, 0.0), 0.5, key="m_br")
    m_rev  = st.number_input("Revenue ($M)",           0.0, 5000.0, float(defaults["revenue_million"]) * 1.5, 10.0, key="m_rev")
    m_inv  = st.selectbox("Investor Type",["none","angel","tier2_vc","tier1_vc"], index=2, key="m_inv")
    m_sec  = st.selectbox("Sector",["AI","Climate","Crypto","Ecommerce","Fintech","Health","SaaS"],
                          index=["AI","Climate","Crypto","Ecommerce","Fintech","Health","SaaS"].index(defaults["sector"]), key="m_sec")
    m_bg   = st.selectbox("Founder Background",["first_time","academic","ex_bigtech","serial_founder"], index=2, key="m_bg")

# ── Build input dicts ─────────────────────────────────────────────────────────
def build_dict(fr, exp, ts, ms, tr, br, rev, inv, sec, bg):
    return {
        "funding_rounds": fr, "founder_experience_years": exp, "team_size": ts,
        "market_size_billion": ms, "product_traction_users": tr,
        "burn_rate_million": br, "revenue_million": rev,
        "investor_type": inv, "sector": sec, "founder_background": bg,
    }

base_dict = build_dict(b_fr, b_exp, b_ts, b_ms, b_tr, b_br, b_rev, b_inv, b_sec, b_bg)
mod_dict  = build_dict(m_fr, m_exp, m_ts, m_ms, m_tr, m_br, m_rev, m_inv, m_sec, m_bg)

X_base = preprocess_single_input(base_dict, transformers)
X_mod  = preprocess_single_input(mod_dict,  transformers)

proba_base = predict_proba(model, X_base, transformers["label_encoder"])
proba_mod  = predict_proba(model, X_mod,  transformers["label_encoder"])

# ── Results ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📊 Comparison Results")

colour_map = {"IPO": "#2ecc71", "Acquisition": "#3498db", "Failure": "#e74c3c"}

c1, c2 = st.columns(2)

def render_result(col, label, proba):
    top = max(proba, key=proba.get)
    with col:
        st.markdown(
            f"""
            <div style="background:{colour_map[top]}22;border:2px solid {colour_map[top]};
                        border-radius:12px;padding:14px;text-align:center">
                <b style="color:{colour_map[top]};font-size:1.1rem">{label}: {top}</b><br>
                <span style="color:#ccc">Confidence: {proba[top]:.1%}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        for cls in ["IPO", "Acquisition", "Failure"]:
            p = proba.get(cls, 0)
            st.markdown(f"**{cls}**")
            st.progress(float(p), text=f"{p:.1%}")

render_result(c1, "📌 Baseline Prediction", proba_base)
render_result(c2, "✏️ Modified Prediction",  proba_mod)

# ── Delta table ───────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📈 Probability Change")

rows = []
for cls in ["IPO", "Acquisition", "Failure"]:
    base_p = proba_base.get(cls, 0)
    mod_p  = proba_mod.get(cls, 0)
    delta  = mod_p - base_p
    pct    = (delta / base_p * 100) if base_p > 0 else 0
    rows.append({
        "Outcome":         cls,
        "Baseline":        f"{base_p:.2%}",
        "Modified":        f"{mod_p:.2%}",
        "Δ Change":        f"{delta:+.2%}",
        "% Improvement":   f"{pct:+.1f}%",
    })

df_delta = pd.DataFrame(rows)
st.dataframe(df_delta, use_container_width=True, hide_index=True)

# ── Grouped bar comparison chart ─────────────────────────────────────────────
classes = list(proba_base.keys())
fig = go.Figure()
fig.add_trace(go.Bar(
    name="Baseline", x=classes,
    y=[proba_base[c] for c in classes],
    marker_color="#636EFA", opacity=0.8,
))
fig.add_trace(go.Bar(
    name="Modified", x=classes,
    y=[proba_mod[c] for c in classes],
    marker_color="#EF553B", opacity=0.8,
))
fig.update_layout(
    barmode="group",
    title="Baseline vs Modified – Probability Comparison",
    yaxis=dict(tickformat=".0%", title="Probability"),
    xaxis_title="Outcome",
    title_font_size=18,
)
st.plotly_chart(fig, use_container_width=True)

# ── Save modified to session ──────────────────────────────────────────────────
st.session_state["last_input"] = base_dict
st.session_state["last_proba"] = proba_base
