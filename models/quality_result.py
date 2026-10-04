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
        by_fingerprint: Dict[tuple, Issue] = {}
        severity_rank = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}

        for result in detector_results:
            for issue in result.issues:
                fingerprint = (
                    issue.table,
                    issue.row_index,
                    issue.column,
                    issue.issue_type,
                )
                existing = by_fingerprint.get(fingerprint)
                if existing is None:
                    by_fingerprint[fingerprint] = issue
                    issues.append(issue)
                    continue

                # Multiple detectors can independently describe the same
                # defect (for example a missing required field). Keep one
                # canonical finding while preserving detector provenance.
                sources = set(existing.metadata.get("source_detectors", []))
                sources.add(existing.detector)
                sources.add(issue.detector)
                existing.metadata["source_detectors"] = sorted(sources)

                if severity_rank.get(issue.severity, 0) > severity_rank.get(existing.severity, 0):
                    existing.severity = issue.severity
                existing.confidence = max(existing.confidence, issue.confidence)

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
            "issues": [issue.to_dict() for issue in self.issues],
            "detectors": [
                detector.to_dict()
                for detector in self.detector_results
            ],
            "severity_summary": self.severity_summary(),
            "detector_summary": self.detector_summary(),
        }