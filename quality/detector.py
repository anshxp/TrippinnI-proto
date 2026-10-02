"""Quality detection orchestration for Module 2."""

from __future__ import annotations

from typing import Any, List

from detectors.datatype_detector import DatatypeDetector
from detectors.duplicate_detector import DuplicateDetector
from detectors.missing_detector import MissingDetector
from detectors.outlier_detector import OutlierDetector
from models.detector_result import DetectorResult
from models.quality_result import QualityResult
from quality.quality_score import QualityScore
from rule_engine.rule_validator import RuleValidator


class QualityDetector:
    """Coordinate deterministic, statistical, and healthcare validation."""

    def __init__(self) -> None:
        self.detectors = [
            MissingDetector(),
            DuplicateDetector(),
            DatatypeDetector(),
            RuleValidator(),
            OutlierDetector(),
        ]
        self.scorer = QualityScore()

    def run(self, dataset: Any, profile: Any) -> QualityResult:
        results: List[DetectorResult] = []
        for detector in self.detectors:
            try:
                result = detector.detect(dataset, profile)
            except Exception as exc:
                result = DetectorResult(
                    detector_name=getattr(detector, "name", detector.__class__.__name__),
                    success=False, error=str(exc),
                )
            results.append(result)

        quality = QualityResult.from_detector_results(results)
        if isinstance(dataset, dict):
            total_records = sum(len(df) for df in dataset.values() if hasattr(df, "__len__"))
            total_cells = sum(getattr(df, "size", 0) for df in dataset.values())
        else:
            total_records = len(dataset) if hasattr(dataset, "__len__") else 0
            total_cells = getattr(dataset, "size", 0)

        score = self.scorer.calculate(
            quality.issues, total_records=total_records, total_cells=total_cells
        )
        quality.quality_score = score["overall_score"]
        quality.summary = score
        return quality