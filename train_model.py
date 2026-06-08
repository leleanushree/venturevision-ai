"""
train_model.py
────────────────────────────────────────────────────────────────────────────────
VentureVision AI – Model Training Pipeline
Trains Logistic Regression, Random Forest, XGBoost, and Gradient Boosting.
Evaluates each model, picks the best by macro F1, and saves artefacts.

Usage:
    python train_model.py
────────────────────────────────────────────────────────────────────────────────
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ── Project paths ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ── Import project utilities ──────────────────────────────────────────────────
import sys
sys.path.insert(0, BASE_DIR)
from utils.preprocessing import preprocess_pipeline


# ── Model definitions ─────────────────────────────────────────────────────────
def get_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, solver="lbfgs",
            class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=50, max_depth=8, min_samples_split=4,
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="mlogloss",
            random_state=42, verbosity=0,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=42
        ),
    }


# ── Evaluation helper ─────────────────────────────────────────────────────────
def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Return accuracy, precision, recall, F1, and raw predictions."""
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "y_pred": y_pred,
        "report": classification_report(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }


# ── Main training function ────────────────────────────────────────────────────
def train():
    print("=" * 65)
    print("  VentureVision AI – Training Pipeline")
    print("=" * 65)

    # 1. Preprocess
    print("\n[1/4] Preprocessing data …")
    X_train, X_test, y_train, y_test, feature_names, label_encoder = preprocess_pipeline()

    # 2. Train all models
    print("\n[2/4] Training models …")
    models = get_models()
    results = {}

    for name, model in models.items():
        print(f"  • Training {name} …", end=" ", flush=True)
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        results[name] = metrics
        print(
            f"Acc={metrics['accuracy']:.4f}  "
            f"Prec={metrics['precision']:.4f}  "
            f"Rec={metrics['recall']:.4f}  "
            f"F1={metrics['f1']:.4f}"
        )

    # 3. Pick best model
    print("\n[3/4] Selecting best model …")
    best_name = max(results, key=lambda k: results[k]["f1"])
    best_model = models[best_name]
    best_metrics = results[best_name]

    print(f"\n  ★ Best Model : {best_name}")
    print(f"    Accuracy   : {best_metrics['accuracy']:.4f}")
    print(f"    Precision  : {best_metrics['precision']:.4f}")
    print(f"    Recall     : {best_metrics['recall']:.4f}")
    print(f"    F1 (macro) : {best_metrics['f1']:.4f}")
    print(f"\n  Classification Report:\n{best_metrics['report']}")

    # 4. Save artefacts
    print("[4/4] Saving artefacts …")

    # best model
    with open(os.path.join(MODELS_DIR, "model.pkl"), "wb") as f:
        pickle.dump(best_model, f)
    print(f"  ✅ model.pkl  ({best_name})")

    # all model results (for the dashboard)
    # strip non-serialisable y_pred arrays to keep file small
    results_save = {
        k: {mk: mv for mk, mv in v.items() if mk not in ("y_pred", "report", "confusion_matrix")}
        for k, v in results.items()
    }
    # add test preds for the best model only (needed for ROC, CM on model page)
    results_save["_best"] = best_name
    results_save["_y_test"] = y_test
    results_save["_y_pred"] = best_metrics["y_pred"]
    results_save["_X_test"] = X_test
    results_save["_feature_names"] = feature_names

    with open(os.path.join(MODELS_DIR, "model_results.pkl"), "wb") as f:
        pickle.dump(results_save, f)
    print("  ✅ model_results.pkl")

    print("\n" + "=" * 65)
    print("  Training complete! All artefacts saved to models/")
    print("=" * 65)


if __name__ == "__main__":
    train()
