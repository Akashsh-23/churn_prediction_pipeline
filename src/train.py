"""Model training and persistence."""

from __future__ import annotations

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.utils import MODELS_DIR, RANDOM_STATE, ensure_directories, log_phase, safe_name


def model_searches() -> dict[str, GridSearchCV]:
    """Create small, reproducible tuning searches for required models."""
    return {
        "Logistic Regression": GridSearchCV(
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
            {"C": [0.1, 1.0, 3.0]},
            scoring="recall",
            cv=3,
            n_jobs=-1,
        ),
        "Decision Tree": GridSearchCV(
            DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
            {"max_depth": [4, 6, 10], "min_samples_leaf": [20, 50]},
            scoring="recall",
            cv=3,
            n_jobs=-1,
        ),
        "Random Forest": GridSearchCV(
            RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
            {"n_estimators": [120], "max_depth": [6, 10], "min_samples_leaf": [10, 25]},
            scoring="recall",
            cv=3,
            n_jobs=-1,
        ),
        "XGBoost": GridSearchCV(
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            {"n_estimators": [80, 140], "max_depth": [3, 4], "learning_rate": [0.05, 0.1]},
            scoring="recall",
            cv=3,
            n_jobs=-1,
        ),
    }


def train_models(x_train, y_train) -> dict[str, object]:
    """Train, tune, save, and return all required models."""
    ensure_directories()
    trained = {}
    for name, search in model_searches().items():
        log_phase("PHASE 5", f"Training {name}...")
        search.fit(x_train, y_train)
        best = search.best_estimator_
        trained[name] = best
        joblib.dump(best, MODELS_DIR / f"{safe_name(name)}.pkl")
        if name == "XGBoost":
            best.save_model(MODELS_DIR / "xgboost.json")
        log_phase("PHASE 5", f"{name} best params: {search.best_params_}")
    return trained
