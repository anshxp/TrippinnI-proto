"""
Isolation Forest wrapper.

Provides anomaly detection using scikit-learn's IsolationForest.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.ensemble import IsolationForest


class IsolationForestDetector:
    """
    Wrapper around sklearn IsolationForest.
    """

    def __init__(
        self,
        contamination: float = 0.05,
        random_state: int = 42,
    ) -> None:

        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
        )

    def fit(self, X: np.ndarray) -> None:
        """
        Train the model.
        """
        self.model.fit(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies.

        Returns
        -------
        ndarray
            1  -> Normal
            -1 -> Outlier
        """
        return self.model.predict(X)

    def decision_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Return anomaly scores.
        """
        return self.model.decision_function(X)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Train and predict in one step.
        """
        return self.model.fit_predict(X)