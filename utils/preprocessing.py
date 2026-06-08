"""
preprocessing.py
Handles all data loading, cleaning, encoding, scaling, and splitting.
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


# ── Column definitions ────────────────────────────────────────────────────────
NUMERIC_COLS = [
    "funding_rounds",
    "founder_experience_years",
    "team_size",
    "market_size_billion",
    "product_traction_users",
    "burn_rate_million",
    "revenue_million",
]
CATEGORICAL_COLS = ["investor_type", "sector", "founder_background"]
TARGET_COL = "outcome"
CLASSES = ["Acquisition", "Failure", "IPO"]  # alphabetical – matches LabelEncoder

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "startup_success_dataset.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


# ── Public API ────────────────────────────────────────────────────────────────

def load_raw_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV. Returns a DataFrame."""
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicates, strip whitespace from strings,
    handle missing values (none present in this dataset,
    but handled defensively).
    """
    df = df.copy()
    df.drop_duplicates(inplace=True)

    # Strip string columns
    for col in CATEGORICAL_COLS + [TARGET_COL]:
        df[col] = df[col].astype(str).str.strip()

    # Drop rows where target is missing
    df.dropna(subset=[TARGET_COL], inplace=True)

    # Fill numeric NAs with median (defensive)
    for col in NUMERIC_COLS:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)

    return df


def encode_and_scale(
    df: pd.DataFrame,
    fit: bool = True,
    encoders: dict = None,
    scaler: StandardScaler = None,
    label_encoder: LabelEncoder = None,
):
    """
    Encode categorical features and scale numerics.

    Parameters
    ----------
    df         : cleaned DataFrame
    fit        : if True, fit new transformers; if False, use provided ones
    encoders   : dict of {col: LabelEncoder} (used when fit=False)
    scaler     : fitted StandardScaler (used when fit=False)
    label_encoder : fitted LabelEncoder for target (used when fit=False)

    Returns
    -------
    X          : feature array (numpy)
    y          : target array (numpy int)
    encoders   : dict of fitted LabelEncoders
    scaler     : fitted StandardScaler
    label_encoder : fitted LabelEncoder for target
    feature_names : list of feature column names
    """
    df = df.copy()

    # Encode categoricals
    if fit:
        encoders = {}
        for col in CATEGORICAL_COLS:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
        label_encoder = LabelEncoder()
        df[TARGET_COL] = label_encoder.fit_transform(df[TARGET_COL])
    else:
        for col in CATEGORICAL_COLS:
            le = encoders[col]
            df[col] = le.transform(df[col])
        df[TARGET_COL] = label_encoder.transform(df[TARGET_COL])

    feature_names = NUMERIC_COLS + CATEGORICAL_COLS
    X = df[feature_names].values
    y = df[TARGET_COL].values

    # Scale numerics (only scale numeric columns; categoricals already encoded)
    num_idx = list(range(len(NUMERIC_COLS)))   # first len(NUMERIC_COLS) cols
    if fit:
        scaler = StandardScaler()
        X[:, num_idx] = scaler.fit_transform(X[:, num_idx])
    else:
        X[:, num_idx] = scaler.transform(X[:, num_idx])

    return X, y, encoders, scaler, label_encoder, feature_names


def preprocess_pipeline(path: str = DATA_PATH):
    """
    Full preprocessing: load → clean → encode → scale → split.
    Saves transformers to models/ directory.

    Returns
    -------
    X_train, X_test, y_train, y_test, feature_names, label_encoder
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    df = load_raw_data(path)
    df = clean_data(df)
    df = df.sample(n=min(20000, len(df)), random_state=42)  # limit for speed

    X, y, encoders, scaler, label_encoder, feature_names = encode_and_scale(df, fit=True)

    # Train / test split (stratified to handle class imbalance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Persist transformers
    with open(os.path.join(MODELS_DIR, "encoders.pkl"), "wb") as f:
        pickle.dump(encoders, f)
    with open(os.path.join(MODELS_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(MODELS_DIR, "label_encoder.pkl"), "wb") as f:
        pickle.dump(label_encoder, f)
    with open(os.path.join(MODELS_DIR, "feature_names.pkl"), "wb") as f:
        pickle.dump(feature_names, f)

    print(f"✅ Preprocessing complete. Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test, feature_names, label_encoder


def load_transformers():
    """Load saved preprocessing objects from disk."""
    def _load(name):
        path = os.path.join(MODELS_DIR, name)
        with open(path, "rb") as f:
            return pickle.load(f)

    return {
        "encoders": _load("encoders.pkl"),
        "scaler": _load("scaler.pkl"),
        "label_encoder": _load("label_encoder.pkl"),
        "feature_names": _load("feature_names.pkl"),
    }


def preprocess_single_input(input_dict: dict, transformers: dict) -> np.ndarray:
    """
    Transform a single user-supplied input dict into a model-ready array.

    Parameters
    ----------
    input_dict   : {feature_name: value, ...}
    transformers : output of load_transformers()

    Returns
    -------
    X : (1, n_features) numpy array
    """
    encoders = transformers["encoders"]
    scaler = transformers["scaler"]
    feature_names = transformers["feature_names"]

    row = {}
    for col in NUMERIC_COLS:
        row[col] = float(input_dict[col])
    for col in CATEGORICAL_COLS:
        le = encoders[col]
        row[col] = le.transform([input_dict[col]])[0]

    X = np.array([[row[col] for col in feature_names]], dtype=float)
    num_idx = list(range(len(NUMERIC_COLS)))
    X[:, num_idx] = scaler.transform(X[:, num_idx])
    return X
