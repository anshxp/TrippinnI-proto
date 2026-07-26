"""
Report Generator.

Generates the final data quality assessment report.
"""

from __future__ import annotations

from typing import Any, Dict, List

from models.issue import Issue
from models.quality_result import QualityResult


class ReportGenerator:
    """
    Generates a structured report from the quality assessment.
    """

    def generate(
        self,
        quality_result: QualityResult,
        explanations: List[Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        """
        Generate the final report.
        """

        return {
            "summary": self._summary(quality_result),
            "quality_score": quality_result.quality_score,
            "detectors": quality_result.detector_summary(),
            "severity": quality_result.severity_summary(),
            "issues": self._issues(quality_result.issues),
            "explanations": explanations or [],
        }

    def _summary(
        self,
        quality_result: QualityResult,
    ) -> Dict[str, Any]:
        """
        Build report summary.
        """

        return {
            "total_issues": quality_result.total_issues,
            "quality_score": quality_result.quality_score,
            "detectors": len(quality_result.detector_results),
        }

    def _issues(
        self,
        issues: List[Issue],
    ) -> List[Dict[str, Any]]:
        """
        Convert Issue objects into dictionaries.
        """

        return [
            issue.to_dict()
            for issue in issues
        ]