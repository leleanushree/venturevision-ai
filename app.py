"""
app.py
────────────────────────────────────────────────────────────────────────────────
VentureVision AI – Streamlit Application Entry Point

Run with:
    streamlit run app.py
────────────────────────────────────────────────────────────────────────────────
"""

import streamlit as st
import os, sys, subprocess

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

# ── Auto-train model if model.pkl is missing (for Streamlit Cloud) ────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")

if not os.path.exists(MODEL_PATH):
    st.info("⏳ First launch: Training ML models... This takes 3-4 minutes. Please wait.")
    progress = st.progress(0, text="Starting training pipeline...")
    progress.progress(10, text="Loading and preprocessing data...")
    result = subprocess.run(
        [sys.executable, os.path.join(BASE_DIR, "train_model.py")],
        capture_output=True, text=True, cwd=BASE_DIR
    )
    progress.progress(100, text="Done!")
    if result.returncode != 0:
        st.error("❌ Training failed. Error details:")
        st.code(result.stderr[-3000:])
        st.stop()
    else:
        st.success("✅ Model trained! Loading app now...")
        st.rerun()

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

# ── Redirect to Home page ─────────────────────────────────────────────────────
