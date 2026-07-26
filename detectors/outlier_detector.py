"""
Outlier Detector.

Coordinates multiple outlier detection methods.
"""

from __future__ import annotations

from typing import Any

from detectors.base_detector import BaseDetector
from models.detector_result import DetectorResult


class OutlierDetector(BaseDetector):
    """
    Detect anomalous records using multiple methods.

    Pipeline:
        1. Rule-Based Validation
        2. Isolation Forest
        3. COPOD
        4. Autoencoder
    """

    def __init__(self) -> None:
        super().__init__("OutlierDetector")

    def detect(self, dataset: Any, profile: Any) -> DetectorResult:

        result = DetectorResult(detector_name=self.name)

        # -----------------------------
        # Rule-Based Detection
        # -----------------------------
        rule_result = self._rule_based(dataset, profile)

        # -----------------------------
        # Isolation Forest
        # -----------------------------
        isolation_result = self._isolation_forest(dataset, profile)

        # -----------------------------
        # COPOD
        # -----------------------------
        copod_result = self._copod(dataset, profile)

        # -----------------------------
        # Autoencoder
        # -----------------------------
        autoencoder_result = self._autoencoder(dataset, profile)

        # Merge results
        for detector_result in (
            rule_result,
            isolation_result,
            copod_result,
            autoencoder_result,
        ):
            result.issues.extend(detector_result.issues)

        result.statistics = {
            "rule_based": rule_result.issue_count,
            "isolation_forest": isolation_result.issue_count,
            "copod": copod_result.issue_count,
            "autoencoder": autoencoder_result.issue_count,
        }

        return result

    def _rule_based(self, dataset: Any, profile: Any) -> DetectorResult:
        return DetectorResult("RuleBased")

    def _isolation_forest(self, dataset: Any, profile: Any) -> DetectorResult:
        return DetectorResult("IsolationForest")

    def _copod(self, dataset: Any, profile: Any) -> DetectorResult:
        return DetectorResult("COPOD")

    def _autoencoder(self, dataset: Any, profile: Any) -> DetectorResult:
        return DetectorResult("Autoencoder")