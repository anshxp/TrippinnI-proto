"""
Quality Detection Engine.

Coordinates all Module 2 detectors and aggregates their results.
"""

from __future__ import annotations

from typing import Any, List

from models.detector_result import DetectorResult
from models.quality_result import QualityResult

from detectors.missing_detector import MissingDetector
from detectors.datatype_detector import DatatypeDetector
from detectors.duplicate_detector import DuplicateDetector
from detectors.outlier_detector import OutlierDetector


class QualityDetector:
    """Main orchestration engine for Module 2."""

    def __init__(self) -> None:
        self.detectors = [
            MissingDetector(),
            DuplicateDetector(),
            DatatypeDetector(),
            OutlierDetector(),
        ]

    def run(self, dataset: Any, profile: Any) -> QualityResult:
        """Execute every detector sequentially."""
        results: List[DetectorResult] = []

        for detector in self.detectors:
            # BaseDetector exposes the public detector interface as `detect`.
            result = detector.detect(dataset, profile)
            results.append(result)

        return QualityResult.from_detector_results(results)
