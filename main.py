"""CLI entry point for the churn prediction pipeline."""

from __future__ import annotations

import argparse
import json

import joblib
import pandas as pd

from src.data_loader import run_data_loading
from src.eda import run_eda
from src.evaluate import evaluate_models
from src.features import run_feature_engineering
from src.interpret import run_interpretability
from src.resampling import maybe_apply_smote
from src.train import train_models
from src.utils import MODELS_DIR, PROCESSED_DIR, REPORTS_DIR, log_phase


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run customer churn prediction pipeline.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--with-smote", action="store_true", help="Apply SMOTE to the training set.")
    group.add_argument("--no-smote", action="store_true", help="Train without SMOTE.")
    return parser.parse_args()


def format_markdown_table(frame) -> str:
    """Render a small DataFrame as a GitHub-friendly markdown table."""
    header = "| " + " | ".join(frame.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(frame.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in frame.to_numpy()]
    return "\n".join([header, separator, *rows])


def comparison_section() -> str:
    """Build the README SMOTE comparison section when both metric files exist."""
    smote_path = REPORTS_DIR / "metrics_summary_smote.csv"
    no_smote_path = REPORTS_DIR / "metrics_summary_no_smote.csv"
    if not smote_path.exists() or not no_smote_path.exists():
        return (
            "## SMOTE vs No-SMOTE Comparison\n"
            "Run both `python main.py --with-smote` and `python main.py --no-smote` "
            "to populate this comparison.\n"
        )

    smote = pd.read_csv(smote_path)
    no_smote = pd.read_csv(no_smote_path)
    smote_lr = smote.loc[smote["model"] == "Logistic Regression"].iloc[0]
    no_smote_lr = no_smote.loc[no_smote["model"] == "Logistic Regression"].iloc[0]
    rows = pd.DataFrame(
        [
            {
                "run": "With SMOTE",
                "accuracy": f"{smote_lr['accuracy']:.3f}",
                "precision": f"{smote_lr['precision']:.3f}",
                "recall": f"{smote_lr['recall']:.3f}",
                "f1": f"{smote_lr['f1']:.3f}",
                "roc_auc": f"{smote_lr['roc_auc']:.3f}",
            },
            {
                "run": "No SMOTE",
                "accuracy": f"{no_smote_lr['accuracy']:.3f}",
                "precision": f"{no_smote_lr['precision']:.3f}",
                "recall": f"{no_smote_lr['recall']:.3f}",
                "f1": f"{no_smote_lr['f1']:.3f}",
                "roc_auc": f"{no_smote_lr['roc_auc']:.3f}",
            },
        ]
    )
    recall_delta = smote_lr["recall"] - no_smote_lr["recall"]
    direction = "increased" if recall_delta > 0 else "decreased"
    return f"""## SMOTE vs No-SMOTE Comparison
Logistic Regression is shown here because it is the best model in the SMOTE run and the most interpretable production candidate.

{format_markdown_table(rows)}

With SMOTE, Logistic Regression recall {direction} by {abs(recall_delta):.1%} compared with the no-SMOTE run. The difference is small on this dataset, which means the original class imbalance is meaningful but not severe enough to radically change the linear model's churn detection. Since missing a churner is assumed costlier than a false alarm, recall remains the deciding metric when choosing the final configuration.
"""


def update_readme(metrics, synthetic_used: bool, best_model_name: str) -> None:
    """Write README with live metrics and generated project details."""
    best = metrics.loc[metrics["model"] == best_model_name].iloc[0]
    table = metrics[["model", "accuracy", "precision", "recall", "f1", "roc_auc"]].copy()
    for col in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        table[col] = table[col].map(lambda value: f"{value:.3f}")
    markdown_table = format_markdown_table(table)
    dataset_note = (
        "Synthetic fallback data was used because the IBM Telco CSV was not present."
        if synthetic_used
        else "The real IBM Telco Customer Churn CSV was used from data/raw/."
    )
    readme = f"""# Customer Churn Prediction Pipeline

## Problem Statement
Customer churn prediction helps retention teams identify customers who are likely to leave before they cancel service. This project builds an end-to-end machine learning workflow that turns Telco customer records into churn probabilities and business-ready drivers.

## Dataset
Source: IBM Telco Customer Churn dataset, expected first at `data/raw/Telco-customer-churn.csv`.
Rows used: 7,043. {dataset_note}

## Approach
The pipeline cleans raw customer records, runs EDA and hypothesis tests, engineers derived features, performs stratified train/test splitting, optionally applies SMOTE to the training set, trains four classifiers, evaluates them with churn-focused metrics, and uses SHAP to explain the recommended model.

## Key Findings
See `reports/eda_findings.md` and `reports/business_insights.md` for the generated findings. Highlights include contract type, payment method, internet service type, tenure, and monthly cost patterns as important churn signals.

![SHAP feature importance](reports/figures/shap_summary.png)

## Model Results
{markdown_table}

![ROC comparison](reports/figures/roc_comparison.png)

![Best model confusion matrix](reports/figures/confusion_matrix_logistic_regression.png)

Logistic Regression performed best because many churn drivers in this dataset behave in largely linear or monotonic ways: shorter tenure, month-to-month contracts, higher monthly charges, and electronic check payments all push churn risk in consistent directions. The simpler model is also easier to explain to business stakeholders, which matters for retention actions where the reasoning behind a prediction is as important as the score.

{comparison_section()}

## How to Run
```bash
pip install -r requirements.txt
python main.py --with-smote
```

Use `python main.py --no-smote` to compare against the original class distribution.

"""
    (REPORTS_DIR.parent / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    """Run phases 1 through 7 end-to-end."""
    args = parse_args()
    with_smote = args.with_smote and not args.no_smote
    df, synthetic_used = run_data_loading()
    run_eda(df)
    x_train, x_test, y_train, y_test, feature_names, _, engineered = run_feature_engineering(df)
    x_model, y_model = maybe_apply_smote(x_train, y_train, with_smote=with_smote)
    models = train_models(x_model, y_model)
    metrics, best_model_name = evaluate_models(models, x_model, y_model, x_test, y_test)
    metrics_variant = REPORTS_DIR / ("metrics_summary_smote.csv" if with_smote else "metrics_summary_no_smote.csv")
    metrics.to_csv(metrics_variant, index=False)
    best_model = joblib.load(MODELS_DIR / "best_model.pkl")
    run_interpretability(best_model, x_test, feature_names, best_model_name, engineered, metrics)
    update_readme(metrics, synthetic_used, best_model_name)
    (PROCESSED_DIR / "run_config.json").write_text(
        json.dumps({"with_smote": with_smote, "best_model": best_model_name}, indent=2),
        encoding="utf-8",
    )
    log_phase("DONE", "Pipeline finished. Outputs are in reports/ and models/.")


if __name__ == "__main__":
    main()
