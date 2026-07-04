"""Tests for raw-data fallback behavior."""

from __future__ import annotations

from src import data_loader


def test_load_raw_data_generates_synthetic_when_real_csv_missing(tmp_path, monkeypatch):
    raw_dir = tmp_path / "data" / "raw"

    monkeypatch.setattr(data_loader, "RAW_DIR", raw_dir)
    monkeypatch.setattr(data_loader, "RAW_FILE", "Telco-customer-churn.csv")
    monkeypatch.setattr(data_loader, "RAW_FILE_CANDIDATES", ["Telco-customer-churn.csv"])
    monkeypatch.setattr(data_loader, "ensure_directories", lambda: raw_dir.mkdir(parents=True, exist_ok=True))

    df, synthetic_used = data_loader.load_raw_data()

    assert synthetic_used is True
    assert df.shape[0] == 7043
    assert df.columns.tolist() == data_loader.IBM_TELCO_COLUMNS
    assert data_loader.clean_data(df).columns.tolist() == [column for column in data_loader.TELCO_COLUMNS if column != "customerID"]
    assert (raw_dir / "Telco-customer-churn.csv").exists()
