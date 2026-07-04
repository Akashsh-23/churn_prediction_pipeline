"""Data loading, synthetic fallback generation, and cleaning."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.utils import PROCESSED_DIR, RAW_DIR, RAW_FILE, RAW_FILE_CANDIDATES, RANDOM_STATE, ensure_directories, log_phase


TELCO_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
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
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]

IBM_TELCO_COLUMNS = [
    "CustomerID",
    "Count",
    "Country",
    "State",
    "City",
    "Zip Code",
    "Lat Long",
    "Latitude",
    "Longitude",
    "Gender",
    "Senior Citizen",
    "Partner",
    "Dependents",
    "Tenure Months",
    "Phone Service",
    "Multiple Lines",
    "Internet Service",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Contract",
    "Paperless Billing",
    "Payment Method",
    "Monthly Charges",
    "Total Charges",
    "Churn Label",
    "Churn Value",
    "Churn Score",
    "CLTV",
    "Churn Reason",
]

IBM_TO_CLASSIC_COLUMNS = {
    "CustomerID": "customerID",
    "Gender": "gender",
    "Senior Citizen": "SeniorCitizen",
    "Partner": "Partner",
    "Dependents": "Dependents",
    "Tenure Months": "tenure",
    "Phone Service": "PhoneService",
    "Multiple Lines": "MultipleLines",
    "Internet Service": "InternetService",
    "Online Security": "OnlineSecurity",
    "Online Backup": "OnlineBackup",
    "Device Protection": "DeviceProtection",
    "Tech Support": "TechSupport",
    "Streaming TV": "StreamingTV",
    "Streaming Movies": "StreamingMovies",
    "Contract": "Contract",
    "Paperless Billing": "PaperlessBilling",
    "Payment Method": "PaymentMethod",
    "Monthly Charges": "MonthlyCharges",
    "Total Charges": "TotalCharges",
    "Churn Label": "Churn",
}


def generate_synthetic_telco(n_rows: int = 7043) -> pd.DataFrame:
    """Generate an IBM Telco-like dataset with realistic churn signals."""
    rng = np.random.default_rng(RANDOM_STATE)
    internet = rng.choice(["DSL", "Fiber optic", "No"], n_rows, p=[0.34, 0.44, 0.22])
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n_rows, p=[0.55, 0.21, 0.24])
    tenure = np.clip(rng.gamma(shape=2.2, scale=16, size=n_rows), 0, 72).round().astype(int)
    monthly = (
        20
        + (internet == "DSL") * rng.normal(35, 8, n_rows)
        + (internet == "Fiber optic") * rng.normal(58, 10, n_rows)
        + rng.normal(0, 6, n_rows)
    )
    monthly = np.clip(monthly, 18, 120).round(2)

    def yes_no(prob: float) -> np.ndarray:
        return rng.choice(["Yes", "No"], n_rows, p=[prob, 1 - prob])

    def internet_addon(prob: float) -> np.ndarray:
        vals = yes_no(prob)
        return np.where(internet == "No", "No internet service", vals)

    senior_binary = rng.binomial(1, 0.16, n_rows)
    senior = np.where(senior_binary == 1, "Yes", "No")
    partner = yes_no(0.48)
    dependents = np.where(partner == "Yes", yes_no(0.46), yes_no(0.18))
    paperless = yes_no(0.59)
    payment = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        n_rows,
        p=[0.34, 0.23, 0.22, 0.21],
    )

    logits = (
        -1.0
        + 1.1 * (contract == "Month-to-month")
        - 0.8 * (contract == "Two year")
        + 0.9 * (internet == "Fiber optic")
        + 0.6 * (payment == "Electronic check")
        + 0.35 * (paperless == "Yes")
        + 0.35 * senior_binary
        - 0.025 * tenure
        + 0.012 * (monthly - 65)
    )
    churn_prob = 1 / (1 + np.exp(-logits))
    churn = np.where(rng.random(n_rows) < churn_prob, "Yes", "No")
    churn_value = (churn == "Yes").astype(int)
    total = (monthly * np.maximum(tenure, 1) + rng.normal(0, 50, n_rows)).clip(0).round(2)
    total_as_str = total.astype(str)
    blank_idx = rng.choice(np.arange(n_rows), size=11, replace=False)
    total_as_str[blank_idx] = " "

    cities = rng.choice(["Los Angeles", "San Diego", "San Francisco", "Sacramento", "Fresno"], n_rows)
    zip_codes = rng.integers(90001, 96162, n_rows)
    latitudes = rng.uniform(32.5, 41.9, n_rows).round(6)
    longitudes = rng.uniform(-124.4, -114.1, n_rows).round(6)
    churn_reasons = np.where(
        churn == "Yes",
        rng.choice(
            ["Competitor made better offer", "Moved", "Attitude of support person", "Price too high"],
            n_rows,
        ),
        "",
    )

    return pd.DataFrame(
        {
            "CustomerID": [f"SYN{i:06d}" for i in range(n_rows)],
            "Count": 1,
            "Country": "United States",
            "State": "California",
            "City": cities,
            "Zip Code": zip_codes,
            "Lat Long": [f"{lat}, {lon}" for lat, lon in zip(latitudes, longitudes)],
            "Latitude": latitudes,
            "Longitude": longitudes,
            "Gender": rng.choice(["Female", "Male"], n_rows),
            "Senior Citizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "Tenure Months": tenure,
            "Phone Service": yes_no(0.9),
            "Multiple Lines": rng.choice(["Yes", "No", "No phone service"], n_rows, p=[0.42, 0.48, 0.10]),
            "Internet Service": internet,
            "Online Security": internet_addon(0.36),
            "Online Backup": internet_addon(0.43),
            "Device Protection": internet_addon(0.44),
            "Tech Support": internet_addon(0.34),
            "Streaming TV": internet_addon(0.49),
            "Streaming Movies": internet_addon(0.49),
            "Contract": contract,
            "Paperless Billing": paperless,
            "Payment Method": payment,
            "Monthly Charges": monthly,
            "Total Charges": total_as_str,
            "Churn Label": churn,
            "Churn Value": churn_value,
            "Churn Score": np.clip((churn_prob * 100 + rng.normal(0, 8, n_rows)).round(), 1, 100).astype(int),
            "CLTV": rng.integers(2000, 7000, n_rows),
            "Churn Reason": churn_reasons,
        },
        columns=IBM_TELCO_COLUMNS,
    )


def _find_raw_path():
    """Return the preferred real dataset path if one exists."""
    for file_name in RAW_FILE_CANDIDATES:
        candidate = RAW_DIR / file_name
        if candidate.exists():
            return candidate
    return None


def load_raw_data() -> tuple[pd.DataFrame, bool]:
    """Load the IBM Telco CSV or create the synthetic fallback at the same path."""
    ensure_directories()
    raw_path = _find_raw_path()
    if raw_path is not None:
        df = pd.read_csv(raw_path)
        id_column = "customerID" if "customerID" in df.columns else "CustomerID" if "CustomerID" in df.columns else None
        synthetic_used = bool(id_column and df[id_column].astype(str).str.startswith("SYN").all())
        log_phase("PHASE 1", f"Loaded raw CSV from {raw_path}")
        if synthetic_used:
            log_phase("PHASE 1", "Detected synthetic fallback data by generated customer IDs.")
        return df, synthetic_used

    df = generate_synthetic_telco()
    synthetic_path = RAW_DIR / RAW_FILE
    df.to_csv(synthetic_path, index=False)
    log_phase("PHASE 1", f"Raw CSV not found; generated synthetic fallback at {synthetic_path}")
    return df, True


def normalize_telco_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize IBM and classic Telco schemas into the model-facing schema."""
    normalized = df.rename(columns=IBM_TO_CLASSIC_COLUMNS).copy()
    if "Churn" not in normalized.columns and "Churn Value" in normalized.columns:
        normalized["Churn"] = normalized["Churn Value"]

    available = [column for column in TELCO_COLUMNS if column in normalized.columns]
    normalized = normalized[available].copy()
    return normalized


def normalize_yes_no(value) -> str:
    """Convert common binary encodings to Yes/No strings."""
    if pd.isna(value):
        return value
    text = str(value).strip()
    if text in {"1", "1.0"} or text.lower() == "yes":
        return "Yes"
    if text in {"0", "0.0"} or text.lower() == "no":
        return "No"
    return text


def encode_churn(value) -> int:
    """Convert Churn to binary 1/0."""
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if text in {"1", "1.0"} or text.lower() == "yes":
        return 1
    if text in {"0", "0.0"} or text.lower() == "no":
        return 0
    raise ValueError(f"Unexpected Churn value: {value!r}")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean Telco churn data and return a downstream-ready DataFrame."""
    cleaned = normalize_telco_schema(df)

    duplicate_count = int(cleaned.duplicated().sum())
    if duplicate_count:
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    log_phase("PHASE 1", f"Dropped {duplicate_count} duplicate rows.")

    cleaned = cleaned.drop(columns=["customerID"], errors="ignore")
    if "SeniorCitizen" in cleaned.columns:
        cleaned["SeniorCitizen"] = cleaned["SeniorCitizen"].map(normalize_yes_no).astype("category")
    if "Churn" in cleaned.columns:
        cleaned["Churn"] = cleaned["Churn"].map(encode_churn).astype(int)
    cleaned["TotalCharges"] = pd.to_numeric(cleaned["TotalCharges"], errors="coerce")
    blanks = int(cleaned["TotalCharges"].isna().sum())
    if blanks:
        impute_value = cleaned.groupby("Contract")["TotalCharges"].transform("median")
        cleaned["TotalCharges"] = cleaned["TotalCharges"].fillna(impute_value)
        cleaned["TotalCharges"] = cleaned["TotalCharges"].fillna(cleaned["TotalCharges"].median())
        log_phase("PHASE 1", f"Imputed {blanks} blank TotalCharges values using contract medians.")

    categorical_columns = cleaned.select_dtypes(include=["object", "string"]).columns.difference(["Churn"])
    for column in categorical_columns:
        cleaned[column] = cleaned[column].astype("category")

    missing = cleaned.isna().sum()
    log_phase("PHASE 1", "Missing values after cleaning:")
    print(missing.to_string())
    return cleaned


def save_cleaned_data(df: pd.DataFrame) -> None:
    """Persist cleaned data, preferring parquet and always writing a CSV fallback."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DIR / "cleaned.csv", index=False)
    try:
        df.to_parquet(PROCESSED_DIR / "cleaned.parquet", index=False)
    except Exception as exc:
        log_phase("PHASE 1", f"Parquet write skipped ({exc}); cleaned.csv was saved.")


def run_data_loading() -> tuple[pd.DataFrame, bool]:
    """Run phase 1 and save metadata about whether synthetic data was used."""
    raw, synthetic_used = load_raw_data()
    cleaned = clean_data(raw)
    save_cleaned_data(cleaned)
    (PROCESSED_DIR / "dataset_metadata.json").write_text(
        json.dumps({"synthetic_data_used": synthetic_used, "rows": int(cleaned.shape[0])}, indent=2),
        encoding="utf-8",
    )
    log_phase("PHASE 1", f"Shape: {cleaned.shape}")
    print(cleaned.dtypes.to_string())
    return cleaned, synthetic_used
