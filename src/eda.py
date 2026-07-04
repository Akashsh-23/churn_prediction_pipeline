"""Exploratory plots and statistical hypothesis tests."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

from src.utils import FIGURES_DIR, REPORTS_DIR, TARGET, ensure_directories, log_phase, safe_name

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]


def _churn_mask(series: pd.Series) -> pd.Series:
    """Return a boolean mask for churned customers from binary or Yes/No labels."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int) == 1
    return series.astype(str).str.lower().eq("yes")


def _save_current(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / name, dpi=160, bbox_inches="tight")
    plt.close()


def create_eda_plots(df: pd.DataFrame) -> None:
    """Save target, numeric, and categorical churn plots."""
    sns.set_theme(style="whitegrid")
    plot_df = df.copy()
    plot_df["ChurnLabel"] = _churn_mask(plot_df[TARGET]).map({True: "Churn", False: "No churn"})
    churn_rate = plot_df["ChurnLabel"].value_counts(normalize=True).mul(100).rename("percent").reset_index()
    churn_rate.columns = ["ChurnLabel", "percent"]
    sns.barplot(data=churn_rate, x="ChurnLabel", y="percent", hue="ChurnLabel", legend=False)
    plt.title("Customer Churn Distribution")
    plt.ylabel("Percent of customers")
    _save_current("target_distribution.png")

    for feature in NUMERIC_FEATURES:
        sns.histplot(data=plot_df, x=feature, hue="ChurnLabel", kde=True, bins=30, element="step")
        plt.title(f"{feature} Distribution by Churn")
        _save_current(f"{safe_name(feature)}_by_churn.png")

    for feature in ["Contract", "PaymentMethod", "InternetService", "PaperlessBilling", "TechSupport"]:
        rates = df.groupby(feature)[TARGET].apply(lambda s: _churn_mask(s).mean()).mul(100).reset_index()
        rates.columns = [feature, "churn_rate"]
        sns.barplot(data=rates, x=feature, y="churn_rate", hue=feature, legend=False)
        plt.title(f"Churn Rate by {feature}")
        plt.ylabel("Churn rate (%)")
        plt.xticks(rotation=35, ha="right")
        _save_current(f"churn_rate_by_{safe_name(feature)}.png")


def run_hypothesis_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Run chi-square, t-tests, and ANOVA, returning a tidy result table."""
    results = []
    for feature in CATEGORICAL_FEATURES:
        table = pd.crosstab(df[feature], df[TARGET])
        chi2, p_value, _, _ = stats.chi2_contingency(table)
        results.append({"test": "Chi-square", "feature": feature, "statistic": chi2, "p_value": p_value})

    churned = df[_churn_mask(df[TARGET])]
    retained = df[~_churn_mask(df[TARGET])]
    for feature in NUMERIC_FEATURES:
        statistic, p_value = stats.ttest_ind(churned[feature], retained[feature], equal_var=False)
        results.append({"test": "T-test", "feature": feature, "statistic": statistic, "p_value": p_value})

    groups = [
        _churn_mask(df.loc[df["Contract"] == contract, TARGET]).astype(int)
        for contract in df["Contract"].dropna().unique()
    ]
    statistic, p_value = stats.f_oneway(*groups)
    results.append({"test": "One-way ANOVA", "feature": "Contract churn rate", "statistic": statistic, "p_value": p_value})
    return pd.DataFrame(results).sort_values("p_value")


def write_findings(results: pd.DataFrame) -> None:
    """Write statistically significant EDA findings to markdown."""
    significant = results[results["p_value"] < 0.05].head(12)
    lines = [
        "# EDA Findings",
        "",
        "Statistical significance threshold: p < 0.05.",
        "",
    ]
    for _, row in significant.iterrows():
        lines.append(
            f"- {row['test']} found `{row['feature']}` significantly associated with churn "
            f"(p={row['p_value']:.3g})."
        )
    if significant.empty:
        lines.append("- No statistically significant findings were detected at p < 0.05.")
    (REPORTS_DIR / "eda_findings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_eda(df: pd.DataFrame) -> pd.DataFrame:
    """Run phase 2 and persist figures/findings."""
    ensure_directories()
    create_eda_plots(df)
    results = run_hypothesis_tests(df)
    results.to_csv(REPORTS_DIR / "hypothesis_tests.csv", index=False)
    write_findings(results)
    log_phase("PHASE 2", "Most significant statistical tests:")
    print(results.head(10).to_string(index=False))
    return results
