"""SHAP interpretability outputs and business takeaways."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.utils import FIGURES_DIR, REPORTS_DIR, TARGET, ensure_directories, log_phase, safe_name


def _churn_mask(series: pd.Series) -> pd.Series:
    """Return a boolean mask for churned customers from binary or Yes/No labels."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int) == 1
    return series.astype(str).str.lower().eq("yes")


def _sample_frame(x_test, feature_names, max_rows: int = 200) -> pd.DataFrame:
    rows = min(max_rows, x_test.shape[0])
    return pd.DataFrame(x_test[:rows], columns=feature_names)


def save_shap_plots(model, x_test, feature_names, best_model_name: str) -> None:
    """Create global and local SHAP plots for the selected model."""
    sample = _sample_frame(x_test, feature_names)
    try:
        explainer = shap.Explainer(model, sample)
        values = explainer(sample)
        shap.summary_plot(values, sample, show=False, max_display=20)
    except Exception:
        explainer = shap.KernelExplainer(model.predict_proba, shap.sample(sample, min(50, len(sample)), random_state=42))
        values = explainer.shap_values(sample.iloc[:80], nsamples=100)
        class_values = values[1] if isinstance(values, list) else values
        shap.summary_plot(class_values, sample.iloc[:80], show=False, max_display=20)
    plt.title(f"SHAP Summary - {best_model_name}")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "shap_summary.png", dpi=160, bbox_inches="tight")
    plt.close()

    try:
        shap_values = values.values if hasattr(values, "values") else class_values
        base_values = values.base_values if hasattr(values, "base_values") else np.zeros(len(sample))
        for idx in range(min(3, len(sample))):
            shap.plots.waterfall(
                shap.Explanation(
                    values=shap_values[idx],
                    base_values=base_values[idx] if np.ndim(base_values) else base_values,
                    data=sample.iloc[idx].values,
                    feature_names=list(feature_names),
                ),
                show=False,
                max_display=12,
            )
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / f"shap_waterfall_sample_{idx + 1}.png", dpi=160, bbox_inches="tight")
            plt.close()
    except Exception as exc:
        log_phase("PHASE 7", f"Local SHAP waterfall plots skipped: {exc}")


def write_business_insights(df: pd.DataFrame, metrics: pd.DataFrame, best_model_name: str) -> None:
    """Write plain-English churn takeaways derived from the data."""
    contract_rates = df.groupby("Contract")[TARGET].apply(lambda s: _churn_mask(s).mean()).sort_values(ascending=False)
    payment_rates = df.groupby("PaymentMethod")[TARGET].apply(lambda s: _churn_mask(s).mean()).sort_values(ascending=False)
    internet_rates = df.groupby("InternetService")[TARGET].apply(lambda s: _churn_mask(s).mean()).sort_values(ascending=False)
    tenure_gap = df.assign(_churned=_churn_mask(df[TARGET])).groupby("_churned")["tenure"].mean()
    best = metrics.loc[metrics["model"] == best_model_name].iloc[0]
    lines = [
        "# Business Insights",
        "",
        f"- {contract_rates.index[0]} customers churn at {contract_rates.iloc[0] / max(contract_rates.iloc[-1], 0.001):.1f}x the rate of {contract_rates.index[-1]} customers.",
        f"- Customers using {payment_rates.index[0]} have the highest churn rate ({payment_rates.iloc[0]:.1%}), making payment experience a practical retention lever.",
        f"- {internet_rates.index[0]} customers show the highest churn rate by internet service type ({internet_rates.iloc[0]:.1%}).",
        f"- Churned customers average {tenure_gap.get(True, 0):.1f} months of tenure versus {tenure_gap.get(False, 0):.1f} months for retained customers.",
        f"- The recommended model is {best_model_name}, with recall {best['recall']:.1%} and ROC-AUC {best['roc_auc']:.1%} on the test set.",
    ]
    (REPORTS_DIR / "business_insights.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_interpretability(model, x_test, feature_names, best_model_name: str, df: pd.DataFrame, metrics: pd.DataFrame) -> None:
    """Run phase 7 interpretability outputs."""
    ensure_directories()
    save_shap_plots(model, x_test, feature_names, best_model_name)
    write_business_insights(df, metrics, best_model_name)
    log_phase("PHASE 7", "Saved SHAP plots and business insights.")
