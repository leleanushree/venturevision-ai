"""
model_utils.py
Model loading, prediction, feature importance extraction, and AI insight generation.
"""

import os
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import label_binarize

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
CLASSES = ["Acquisition", "Failure", "IPO"]   # alphabetical (LabelEncoder order)


# ── Model I/O ────────────────────────────────────────────────────────────────

def load_model():
    """Load the best saved model."""
    path = os.path.join(MODELS_DIR, "model.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def load_model_results():
    """Load the saved model comparison results dict."""
    path = os.path.join(MODELS_DIR, "model_results.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


# ── Prediction helpers ────────────────────────────────────────────────────────

def predict_proba(model, X: np.ndarray, label_encoder) -> dict:
    """
    Return a dict: {class_name: probability} for a single sample.
    """
    probs = model.predict_proba(X)[0]
    class_names = label_encoder.classes_
    return {cls: float(prob) for cls, prob in zip(class_names, probs)}


# ── Feature importance ────────────────────────────────────────────────────────

def get_feature_importance(model, feature_names: list) -> pd.DataFrame:
    """
    Extract feature importances from tree-based models.
    Falls back to coefficient magnitude for linear models.
    """
    try:
        importances = model.feature_importances_
    except AttributeError:
        # Logistic Regression – use mean absolute coefficient
        importances = np.mean(np.abs(model.coef_), axis=0)

    df = pd.DataFrame(
        {"Feature": feature_names, "Importance": importances}
    ).sort_values("Importance", ascending=False)
    return df


def plot_feature_importance(model, feature_names: list) -> go.Figure:
    """Bar chart of feature importances."""
    df = get_feature_importance(model, feature_names)
    fig = px.bar(
        df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Feature Importance",
        color="Importance",
        color_continuous_scale="Blues",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, title_font_size=18)
    return fig


# ── AI Insights ───────────────────────────────────────────────────────────────

# Thresholds derived from training-set median values (will be refined at runtime)
_THRESHOLDS = {
    "funding_rounds": 2,
    "founder_experience_years": 12,
    "team_size": 50,
    "market_size_billion": 20,
    "product_traction_users": 100_000,
    "burn_rate_million": 15,
    "revenue_million": 500_000,
}

_LABELS = {
    "funding_rounds": "Funding Rounds",
    "founder_experience_years": "Founder Experience",
    "team_size": "Team Size",
    "market_size_billion": "Market Size",
    "product_traction_users": "Product Traction",
    "burn_rate_million": "Burn Rate",
    "revenue_million": "Revenue",
    "investor_type": "Investor Type",
    "sector": "Sector",
    "founder_background": "Founder Background",
}

_HIGH_IS_GOOD = {
    "funding_rounds": True,
    "founder_experience_years": True,
    "team_size": True,
    "market_size_billion": True,
    "product_traction_users": True,
    "burn_rate_million": False,   # high burn is bad
    "revenue_million": True,
}

_INVESTOR_RANK = {"none": 0, "angel": 1, "tier2_vc": 2, "tier1_vc": 3}
_BACKGROUND_RANK = {"first_time": 0, "academic": 1, "ex_bigtech": 2, "serial_founder": 3}


def generate_insights(input_dict: dict, proba_dict: dict, model, feature_names: list) -> dict:
    """
    Generate strengths, risks, and improvements from the input and prediction.

    Returns
    -------
    dict with keys: strengths (list[str]), risks (list[str]), improvements (list[str])
    """
    strengths, risks, improvements = [], [], []

    # Numeric feature checks
    for col, thresh in _THRESHOLDS.items():
        val = float(input_dict.get(col, thresh))
        label = _LABELS[col]
        high_good = _HIGH_IS_GOOD[col]

        if high_good:
            if val >= thresh:
                strengths.append(f"Strong {label.lower()} ({val:,.0f})")
            else:
                risks.append(f"Low {label.lower()} ({val:,.0f})")
                improvements.append(f"Increase {label.lower()} above {thresh:,.0f}")
        else:
            if val <= thresh:
                strengths.append(f"Controlled {label.lower()} (${val:.1f}M)")
            else:
                risks.append(f"High {label.lower()} (${val:.1f}M)")
                improvements.append(f"Reduce {label.lower()} below ${thresh:.1f}M")

    # Categorical checks
    inv = input_dict.get("investor_type", "none")
    if _INVESTOR_RANK.get(inv, 0) >= 2:
        strengths.append(f"Tier VC backing ({inv})")
    else:
        risks.append(f"Weak investor signal ({inv})")
        improvements.append("Upgrade to Tier 1/2 VC investor")

    bg = input_dict.get("founder_background", "first_time")
    if _BACKGROUND_RANK.get(bg, 0) >= 2:
        strengths.append(f"Strong founder background ({bg.replace('_', ' ')})")
    else:
        risks.append(f"Limited founder experience ({bg.replace('_', ' ')})")
        improvements.append("Bring in ex-bigtech or serial founder co-founder")

    # Probability-based context
    failure_p = proba_dict.get("Failure", 0)
    ipo_p = proba_dict.get("IPO", 0)
    if failure_p > 0.5:
        risks.append(f"High predicted failure probability ({failure_p:.0%})")
    if ipo_p > 0.2:
        strengths.append(f"Notable IPO potential ({ipo_p:.0%})")

    return {
        "strengths": strengths[:6],
        "risks": risks[:6],
        "improvements": improvements[:5],
    }


# ── Confusion Matrix chart ────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, class_names) -> go.Figure:
    cm = confusion_matrix(y_true, y_pred)
    fig = px.imshow(
        cm,
        text_auto=True,
        x=class_names,
        y=class_names,
        title="Confusion Matrix",
        color_continuous_scale="Blues",
        labels=dict(x="Predicted", y="Actual"),
    )
    fig.update_layout(title_font_size=18)
    return fig


# ── ROC Curves ────────────────────────────────────────────────────────────────

def plot_roc_curves(model, X_test, y_test, label_encoder) -> go.Figure:
    class_names = label_encoder.classes_
    n_classes = len(class_names)
    y_bin = label_binarize(y_test, classes=list(range(n_classes)))
    y_score = model.predict_proba(X_test)

    colors = ["#2ecc71", "#e74c3c", "#3498db"]
    fig = go.Figure()

    for i, (cls, color) in enumerate(zip(class_names, colors)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        fig.add_trace(
            go.Scatter(
                x=fpr, y=tpr,
                name=f"{cls} (AUC = {roc_auc:.3f})",
                mode="lines",
                line=dict(color=color, width=2),
            )
        )

    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1],
            name="Random Classifier",
            mode="lines",
            line=dict(color="grey", dash="dash"),
        )
    )
    fig.update_layout(
        title="ROC Curves – One-vs-Rest",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        title_font_size=18,
    )
    return fig


# ── Model comparison bar chart ────────────────────────────────────────────────

def plot_model_comparison(results: dict) -> go.Figure:
    rows = []
    for name, metrics in results.items():
        rows.append({"Model": name, "Metric": "Accuracy", "Score": metrics["accuracy"]})
        rows.append({"Model": name, "Metric": "F1 (macro)", "Score": metrics["f1"]})
        rows.append({"Model": name, "Metric": "Precision", "Score": metrics["precision"]})
        rows.append({"Model": name, "Metric": "Recall", "Score": metrics["recall"]})

    df = pd.DataFrame(rows)
    fig = px.bar(
        df,
        x="Model",
        y="Score",
        color="Metric",
        barmode="group",
        title="Model Comparison",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(yaxis_range=[0, 1], title_font_size=18)
    return fig
