"""
QualityResult model for Module 2.

Represents the final output of the quality detection pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from models.detector_result import DetectorResult
from models.issue import Issue


@dataclass(slots=True)
class QualityResult:
    """
    Final output produced by the QualityDetector.
    """

    detector_results: List[DetectorResult] = field(default_factory=list)

    issues: List[Issue] = field(default_factory=list)

    quality_score: float = 0.0

    summary: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_issues(self) -> int:
        """Return total number of detected issues."""
        return len(self.issues)

    @classmethod
    def from_detector_results(
        cls,
        detector_results: List[DetectorResult],
    ) -> "QualityResult":
        """
        Build a QualityResult from all detector outputs.
        """

        issues: List[Issue] = []

        for result in detector_results:
            issues.extend(result.issues)

        return cls(
            detector_results=detector_results,
            issues=issues,
        )

    def detector_summary(self) -> Dict[str, int]:
        """
        Return issue count for each detector.
        """

        return {
            result.detector_name: result.issue_count
            for result in self.detector_results
        }

    def severity_summary(self) -> Dict[str, int]:
        """
        Count issues by severity.
        """

        summary: Dict[str, int] = {}

        for issue in self.issues:
            summary[issue.severity] = summary.get(issue.severity, 0) + 1

        return summary

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the result into a serializable dictionary.
        """

        return {
            "quality_score": self.quality_score,
            "total_issues": self.total_issues,
            "summary": self.summary,
            "detectors": [
                detector.to_dict()
                for detector in self.detector_results
            ],
            "severity_summary": self.severity_summary(),
            "detector_summary": self.detector_summary(),
        }