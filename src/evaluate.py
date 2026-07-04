"""Model metrics, diagnostic plots, and best-model selection."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.utils import FIGURES_DIR, MODELS_DIR, REPORTS_DIR, RANDOM_STATE, ensure_directories, log_phase, safe_name


def _probabilities(model, x_test):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_test)[:, 1]
    return model.decision_function(x_test)


def evaluate_models(models: dict[str, object], x_train, y_train, x_test, y_test) -> tuple[pd.DataFrame, str]:
    """Evaluate all models and save metrics and figures."""
    ensure_directories()
    rows = []
    plt.figure(figsize=(8, 6))
    for name, model in models.items():
        y_pred = model.predict(x_test)
        y_prob = _probabilities(model, x_test)
        auc = roc_auc_score(y_test, y_prob)
        rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
                "roc_auc": auc,
            }
        )

        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No churn", "Churn"])
        disp.plot(cmap="Blues", values_format="d")
        plt.title(f"{name} Confusion Matrix")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f"confusion_matrix_{safe_name(name)}.png", dpi=160, bbox_inches="tight")
        plt.close()

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "roc_comparison.png", dpi=160, bbox_inches="tight")
    plt.close()

    metrics = pd.DataFrame(rows).sort_values(["recall", "roc_auc"], ascending=False)
    best_model_name = metrics.iloc[0]["model"]
    best_model = models[best_model_name]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(best_model, x_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    metrics["best_model_cv_roc_auc_mean"] = float("nan")
    metrics["best_model_cv_roc_auc_std"] = float("nan")
    metrics.loc[metrics["model"] == best_model_name, "best_model_cv_roc_auc_mean"] = cv_scores.mean()
    metrics.loc[metrics["model"] == best_model_name, "best_model_cv_roc_auc_std"] = cv_scores.std()
    metrics.to_csv(REPORTS_DIR / "metrics_summary.csv", index=False)
    joblib.dump(best_model, MODELS_DIR / "best_model.pkl")

    log_phase(
        "PHASE 6",
        f"Best model: {best_model_name}. Recommendation prioritizes recall because missing a churner is assumed costlier than a false alarm.",
    )
    print(metrics.to_string(index=False))
    return metrics, best_model_name
