"""
app.py
────────────────────────────────────────────────────────────────────────────────
VentureVision AI – Streamlit Application Entry Point

Run with:
    streamlit run app.py
────────────────────────────────────────────────────────────────────────────────
"""

import streamlit as st

st.set_page_config(
    page_title="VentureVision AI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/yourusername/venturevision-ai",
        "About": "VentureVision AI – Startup Outcome Prediction & Decision Support",
    },
)

# ── Global sidebar branding ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center;padding:10px 0 20px 0">
            <div style="font-size:2.5rem">🚀</div>
            <div style="font-weight:bold;font-size:1.2rem;color:#e94560">VentureVision AI</div>
            <div style="font-size:0.8rem;color:#a8b2d8">Startup Outcome Prediction</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("**Navigation**")
    st.markdown(
        """
        - 🏠 **Home** – Overview & dataset
        - 🔮 **Predictor** – Get a prediction
        - 💡 **AI Insights** – Strengths & risks
        - 📈 **Analytics** – EDA dashboard
        - 🏆 **Model Performance** – Metrics & ROC
        """
    )
    st.markdown("---")
    st.caption("Built with Streamlit · Scikit-learn · XGBoost · Plotly")

# ── Redirect to Home page ────────────────────────────────────────────────────
st.switch_page("pages/1_Home.py")