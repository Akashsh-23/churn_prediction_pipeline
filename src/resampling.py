"""Class imbalance reporting and optional SMOTE resampling."""

from __future__ import annotations

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE

from src.utils import RANDOM_STATE, log_phase


def print_class_balance(y, label: str) -> None:
    """Print class counts and percentages."""
    counts = pd.Series(y).value_counts().sort_index()
    percentages = counts / counts.sum()
    summary = {int(k): f"{int(counts[k])} ({percentages[k]:.1%})" for k in counts.index}
    log_phase("PHASE 4", f"{label} class balance: {summary}")


def maybe_apply_smote(x_train: np.ndarray, y_train, with_smote: bool = False):
    """Apply SMOTE to training data only when requested."""
    print_class_balance(y_train, "Original training")
    if not with_smote:
        log_phase("PHASE 4", "SMOTE disabled; using original training distribution.")
        return x_train, y_train

    sampler = SMOTE(random_state=RANDOM_STATE)
    x_resampled, y_resampled = sampler.fit_resample(x_train, y_train)
    print_class_balance(y_resampled, "SMOTE-resampled training")
    return x_resampled, y_resampled
