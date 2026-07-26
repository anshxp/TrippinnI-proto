"""
Quality Score Calculator.

Computes the overall dataset quality score.
"""

from __future__ import annotations

from collections import Counter
from typing import List

from models.issue import Issue


class QualityScore:
    """
    Computes an overall quality score for the dataset.
    """

    def __init__(
        self,
        missing_weight: float = 0.20,
        duplicate_weight: float = 0.20,
        datatype_weight: float = 0.20,
        outlier_weight: float = 0.40,
    ):

        self.weights = {
            "missing": missing_weight,
            "duplicate": duplicate_weight,
            "datatype": datatype_weight,
            "outlier": outlier_weight,
        }

    def calculate(
        self,
        issues: List[Issue],
        total_records: int,
    ) -> dict:
        """
        Calculate the overall quality score.
        """

        issue_counts = Counter(
            issue.issue_type.lower()
            for issue in issues
        )

        scores = {}

        for issue_type, weight in self.weights.items():

            count = issue_counts.get(issue_type, 0)

            if total_records == 0:
                quality = 100.0
            else:
                quality = max(
                    0.0,
                    100 - ((count / total_records) * 100),
                )

            scores[issue_type] = {
                "count": count,
                "score": round(quality, 2),
                "weight": weight,
            }

        overall_score = sum(
            scores[name]["score"] * scores[name]["weight"]
            for name in self.weights
        )

        return {
            "overall_score": round(overall_score, 2),
            "category_scores": scores,
            "grade": self._grade(overall_score),
        }

    def _grade(
        self,
        score: float,
    ) -> str:
        """
        Convert numerical score into quality grade.
        """

        if score >= 95:
            return "Excellent"

        if score >= 85:
            return "Good"

        if score >= 70:
            return "Fair"

        if score >= 50:
            return "Poor"

        return "Critical"