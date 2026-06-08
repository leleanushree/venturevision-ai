"""
Page 5 – Model Performance
Accuracy, confusion matrix, classification report, ROC curves, model comparison.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.model_utils import (
    load_model, load_model_results,
    plot_confusion_matrix, plot_roc_curves,
    plot_model_comparison, plot_feature_importance,
)
from utils.preprocessing import load_transformers
from sklearn.metrics import classification_report

st.set_page_config(page_title="Model Performance", page_icon="🏆", layout="wide")

@st.cache_resource
def get_everything():
    model       = load_model()
    transformers = load_transformers()
    results     = load_model_results()
    return model, transformers, results

try:
    model, transformers, results = get_everything()
except FileNotFoundError:
    st.error("⚠️ Model artefacts not found. Run `python train_model.py` first.")
    st.stop()

best_name  = results.get("_best", "Unknown")
y_test     = results.get("_y_test")
y_pred     = results.get("_y_pred")
X_test     = results.get("_X_test")
label_enc  = transformers["label_encoder"]
class_names = list(label_enc.classes_)

# Clean results (remove internal keys)
model_metrics = {k: v for k, v in results.items() if not k.startswith("_")}

st.title("🏆 Model Performance")
st.markdown(f"**Best Model:** `{best_name}`")
st.markdown("---")

# ── KPI cards ─────────────────────────────────────────────────────────────────
best_metrics = model_metrics.get(best_name, {})
k1, k2, k3, k4 = st.columns(4)
k1.metric("Accuracy",        f"{best_metrics.get('accuracy', 0):.4f}")
k2.metric("Precision (macro)", f"{best_metrics.get('precision', 0):.4f}")
k3.metric("Recall (macro)",   f"{best_metrics.get('recall', 0):.4f}")
k4.metric("F1 Score (macro)", f"{best_metrics.get('f1', 0):.4f}")

st.markdown("---")

# ── Model comparison ──────────────────────────────────────────────────────────
st.subheader("📊 Model Comparison")
fig_comp = plot_model_comparison(model_metrics)
st.plotly_chart(fig_comp, use_container_width=True)

# ── Metrics table ─────────────────────────────────────────────────────────────
rows = []
for name, m in model_metrics.items():
    rows.append({
        "Model": name,
        "Accuracy": f"{m['accuracy']:.4f}",
        "Precision": f"{m['precision']:.4f}",
        "Recall": f"{m['recall']:.4f}",
        "F1 Score": f"{m['f1']:.4f}",
        "Best": "⭐" if name == best_name else "",
    })
df_metrics = pd.DataFrame(rows).sort_values("F1 Score", ascending=False)
st.dataframe(df_metrics, use_container_width=True, hide_index=True)

st.markdown("---")

# ── Confusion matrix + ROC curves ────────────────────────────────────────────
c1, c2 = st.columns(2)

with c1:
    st.subheader("🗂️ Confusion Matrix")
    if y_test is not None and y_pred is not None:
        fig_cm = plot_confusion_matrix(y_test, y_pred, class_names)
        st.plotly_chart(fig_cm, use_container_width=True)
    else:
        st.warning("Test predictions not available.")

with c2:
    st.subheader("📉 ROC Curves")
    if X_test is not None and y_test is not None:
        try:
            fig_roc = plot_roc_curves(model, X_test, y_test, label_enc)
            st.plotly_chart(fig_roc, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not generate ROC curves: {e}")
    else:
        st.warning("Test data not available.")

st.markdown("---")

# ── Classification report ─────────────────────────────────────────────────────
st.subheader("📋 Classification Report")
if y_test is not None and y_pred is not None:
    report = classification_report(
        y_test, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_rows = []
    for label in class_names + ["macro avg", "weighted avg"]:
        if label in report:
            m = report[label]
            report_rows.append({
                "Class": label,
                "Precision": f"{m['precision']:.4f}",
                "Recall": f"{m['recall']:.4f}",
                "F1-Score": f"{m['f1-score']:.4f}",
                "Support": int(m["support"]) if "support" in m else "-",
            })
    st.dataframe(pd.DataFrame(report_rows), use_container_width=True, hide_index=True)
else:
    st.warning("Test data not available.")

st.markdown("---")

# ── Feature importance ────────────────────────────────────────────────────────
st.subheader("📊 Feature Importance")
fig_fi = plot_feature_importance(model, transformers["feature_names"])
st.plotly_chart(fig_fi, use_container_width=True)