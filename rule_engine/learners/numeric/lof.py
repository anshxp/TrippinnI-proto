"""Local Outlier Factor numeric anomaly learner."""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import LocalOutlierFactor


def anomaly_scores(values: np.ndarray) -> np.ndarray:
    """Return LOF scores while handling ties common in EHR variables."""
    X = np.asarray(values, dtype=float).reshape(-1, 1)
    n_samples = len(X)
    if n_samples < 10:
        return np.zeros(n_samples)

    unique_count = np.unique(X).size
    if unique_count < 5:
        return np.zeros(n_samples)

    n_neighbors = min(35, max(5, n_samples // 10), n_samples - 1)
    n_neighbors = min(n_neighbors, unique_count - 1)
    if n_neighbors < 2:
        return np.zeros(n_samples)

    model = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination="auto",
    )
    model.fit_predict(X)
    return -model.negative_outlier_factor_
