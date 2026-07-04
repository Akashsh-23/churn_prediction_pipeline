"""Tests for generated pipeline artifacts."""

from __future__ import annotations

from pathlib import Path

from src.utils import MODELS_DIR, REPORTS_DIR


def assert_non_empty(path: Path) -> None:
    """Assert a generated artifact exists and has content."""
    assert path.exists(), f"Missing expected artifact: {path}"
    assert path.stat().st_size > 0, f"Empty expected artifact: {path}"


def test_pipeline_outputs_exist_and_are_non_empty():
    assert_non_empty(REPORTS_DIR / "metrics_summary.csv")
    assert_non_empty(REPORTS_DIR / "eda_findings.md")
    assert_non_empty(REPORTS_DIR / "business_insights.md")
    assert_non_empty(REPORTS_DIR / "figures" / "shap_summary.png")

    model_files = [path for path in MODELS_DIR.iterdir() if path.is_file() and path.stat().st_size > 0]
    assert model_files, "Expected at least one non-empty model artifact."
