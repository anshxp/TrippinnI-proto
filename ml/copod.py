"""
COPOD Detector.

Wrapper around PyOD's COPOD algorithm.
"""

from __future__ import annotations

import numpy as np
from pyod.models.copod import COPOD


class COPODDetector:
    """
    Wrapper around PyOD COPOD.
    """

    def __init__(self) -> None:
        self.model = COPOD()

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
            0 -> Normal
            1 -> Outlier
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
        self.fit(X)
        return self.predict(X)