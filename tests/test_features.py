"""Unit tests for feature engineering helpers."""

from __future__ import annotations

import pandas as pd

from src.data_loader import clean_data
from src.features import add_charges_per_tenure, add_tenure_bucket, stratified_split


def test_tenure_bucket_boundaries():
    df = pd.DataFrame({"tenure": [0, 12, 13, 24, 25, 48, 49, 72]})
    out = add_tenure_bucket(df)
    assert out["tenure_bucket"].tolist() == ["0-12", "0-12", "13-24", "13-24", "25-48", "25-48", "49+", "49+"]


def test_charges_per_tenure_uses_tenure_plus_one():
    df = pd.DataFrame({"TotalCharges": [100.0, 0.0], "tenure": [9, 0]})
    out = add_charges_per_tenure(df)
    assert out["charges_per_tenure"].tolist() == [10.0, 0.0]


def test_stratified_split_preserves_class_ratio():
    df = pd.DataFrame(
        {
            "customerID": [f"C{i}" for i in range(100)],
            "tenure": range(100),
            "MonthlyCharges": [50.0] * 100,
            "TotalCharges": [500.0] * 100,
            "Contract": ["Month-to-month"] * 100,
            "Churn": ["Yes"] * 25 + ["No"] * 75,
        }
    )
    _, _, y_train, y_test = stratified_split(df)
    assert abs(y_train.mean() - y_test.mean()) <= 0.02


def test_clean_data_normalizes_real_ibm_schema():
    raw = pd.DataFrame(
        {
            "CustomerID": ["A", "B", "C"],
            "Gender": ["Female", "Male", "Female"],
            "Senior Citizen": ["No", "Yes", "No"],
            "Partner": ["Yes", "No", "No"],
            "Dependents": ["No", "No", "Yes"],
            "Tenure Months": [1, 2, 3],
            "Phone Service": ["Yes", "Yes", "No"],
            "Multiple Lines": ["No", "Yes", "No phone service"],
            "Internet Service": ["DSL", "Fiber optic", "No"],
            "Online Security": ["No", "No", "No internet service"],
            "Online Backup": ["Yes", "No", "No internet service"],
            "Device Protection": ["No", "Yes", "No internet service"],
            "Tech Support": ["No", "No", "No internet service"],
            "Streaming TV": ["No", "Yes", "No internet service"],
            "Streaming Movies": ["No", "Yes", "No internet service"],
            "Contract": ["Month-to-month", "One year", "Two year"],
            "Paperless Billing": ["Yes", "No", "No"],
            "Payment Method": ["Electronic check", "Mailed check", "Credit card (automatic)"],
            "Monthly Charges": [29.85, 56.95, 20.0],
            "Total Charges": ["29.85", " ", "60.0"],
            "Churn Label": ["No", "Yes", "No"],
            "Churn Value": [0, 1, 0],
            "Churn Score": [20, 80, 10],
            "CLTV": [2000, 3000, 4000],
            "Churn Reason": ["", "Moved", ""],
        }
    )

    cleaned = clean_data(raw)

    assert "customerID" not in cleaned.columns
    assert cleaned["Churn"].tolist() == [0, 1, 0]
    assert pd.api.types.is_numeric_dtype(cleaned["TotalCharges"])
    assert cleaned["TotalCharges"].isna().sum() == 0
    assert str(cleaned["SeniorCitizen"].dtype) == "category"
    assert "Churn Score" not in cleaned.columns
