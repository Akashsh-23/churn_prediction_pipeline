"""Shared paths, constants, and lightweight helpers."""

from __future__ import annotations

from pathlib import Path

RANDOM_STATE = 42
TARGET = "Churn"
RAW_FILE = "Telco-customer-churn.csv"
RAW_FILE_CANDIDATES = [
    RAW_FILE,
    "Telco_customer_churn.csv",
    "WA_Fn-UseC_-Telco-Customer-Churn.csv",
]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


def ensure_directories() -> None:
    """Create all runtime output folders."""
    for path in [RAW_DIR, PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def log_phase(phase: str, message: str) -> None:
    """Print a consistent phase-prefixed log line."""
    print(f"[{phase}] {message}")


def safe_name(name: str) -> str:
    """Return a filesystem-friendly version of a display name."""
    return (
        name.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
    )
