"""Feature engineering, train/test split, encoding, and scaling."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils import MODELS_DIR, PROCESSED_DIR, RANDOM_STATE, TARGET, ensure_directories, log_phase

SERVICE_COLUMNS = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def add_tenure_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """Add tenure buckets used by the model and tests."""
    out = df.copy()
    out["tenure_bucket"] = pd.cut(
        out["tenure"],
        bins=[-1, 12, 24, 48, np.inf],
        labels=["0-12", "13-24", "25-48", "49+"],
    ).astype(str)
    return out


def add_charges_per_tenure(df: pd.DataFrame) -> pd.DataFrame:
    """Add average observed charges per tenure month with zero-tenure protection."""
    out = df.copy()
    out["charges_per_tenure"] = out["TotalCharges"] / (out["tenure"] + 1)
    return out


def add_num_services(df: pd.DataFrame) -> pd.DataFrame:
    """Count services where the customer has an active subscription."""
    out = df.copy()
    available = [col for col in SERVICE_COLUMNS if col in out.columns]
    out["num_services"] = out[available].eq("Yes").sum(axis=1)
    return out


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create all derived model features."""
    engineered = add_tenure_bucket(df)
    engineered = add_charges_per_tenure(engineered)
    engineered = add_num_services(engineered)
    return engineered


def stratified_split(
    df: pd.DataFrame, test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data while preserving churn class ratios."""
    if pd.api.types.is_numeric_dtype(df[TARGET]):
        y = df[TARGET].astype(int)
    else:
        y = df[TARGET].map({"No": 0, "Yes": 1}).astype(int)
    x = df.drop(columns=[TARGET, "customerID"], errors="ignore")
    return train_test_split(x, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE)


def _make_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(x_train: pd.DataFrame) -> ColumnTransformer:
    """Build a fit-on-train-only preprocessing transformer."""
    numeric_features = x_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = x_train.select_dtypes(exclude=["number"]).columns.tolist()
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("scaler", StandardScaler())]), numeric_features),
            ("cat", _make_encoder(), categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def run_feature_engineering(df: pd.DataFrame):
    """Run phase 3, save splits, and return transformed arrays plus metadata."""
    ensure_directories()
    engineered = engineer_features(df)
    x_train, x_test, y_train, y_test = stratified_split(engineered)
    preprocessor = build_preprocessor(x_train)
    x_train_t = preprocessor.fit_transform(x_train)
    x_test_t = preprocessor.transform(x_test)
    feature_names = preprocessor.get_feature_names_out()

    np.save(PROCESSED_DIR / "X_train.npy", x_train_t)
    np.save(PROCESSED_DIR / "X_test.npy", x_test_t)
    y_train.to_csv(PROCESSED_DIR / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_DIR / "y_test.csv", index=False)
    x_train.to_csv(PROCESSED_DIR / "X_train_raw.csv", index=False)
    x_test.to_csv(PROCESSED_DIR / "X_test_raw.csv", index=False)
    pd.Series(feature_names).to_csv(PROCESSED_DIR / "feature_names.csv", index=False, header=["feature"])
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.pkl")
    log_phase("PHASE 3", f"Train/test rows: {x_train_t.shape[0]}/{x_test_t.shape[0]}; features: {x_train_t.shape[1]}")
    return x_train_t, x_test_t, y_train, y_test, feature_names, preprocessor, engineered
