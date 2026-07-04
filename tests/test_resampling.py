"""Tests for class-imbalance handling."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.resampling import maybe_apply_smote


def class_gap(y) -> int:
    """Return absolute difference between majority and minority counts."""
    counts = pd.Series(y).value_counts()
    return int(abs(counts.max() - counts.min()))


def test_smote_changes_training_only_and_balances_classes():
    x_train = np.arange(200).reshape(100, 2)
    y_train = pd.Series([0] * 80 + [1] * 20)
    x_test = np.arange(40).reshape(20, 2)
    y_test = pd.Series([0] * 14 + [1] * 6)

    original_test_shape = x_test.shape
    original_test_distribution = y_test.value_counts().sort_index().to_dict()
    original_train_gap = class_gap(y_train)

    x_resampled, y_resampled = maybe_apply_smote(x_train, y_train, with_smote=True)

    assert x_test.shape == original_test_shape
    assert y_test.value_counts().sort_index().to_dict() == original_test_distribution
    assert class_gap(y_resampled) < original_train_gap
    assert pd.Series(y_resampled).value_counts().nunique() == 1
    assert x_resampled.shape[0] > x_train.shape[0]
