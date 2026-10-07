"""Dimension-aware quality scoring for the prototype."""

from __future__ import annotations

from collections import Counter
from typing import List

from models.issue import Issue


class QualityScore:
    """Compute bounded, dimension-specific quality scores.

    Issue counts are normalized against the appropriate denominator:
    missing cells against cells, duplicates against rows, and value/rule
    violations against rows. Penalties are capped so one noisy detector
    cannot produce a negative or otherwise invalid score.
    """

    WEIGHTS = {
        "missing": 0.25,
        "duplicate": 0.15,
        "datatype": 0.20,
        "outlier": 0.15,
        "conformance": 0.10,
        "temporal": 0.05,
        "plausibility": 0.05,
        "referential": 0.05,
        "bias": 0.03,
        "fairness": 0.03,
        "interoperability": 0.03,
        "drift": 0.03,
        "robustness": 0.03,
    }

    def __init__(self, **weights: float) -> None:
        self.weights = dict(self.WEIGHTS)
        for name, value in weights.items():
            if name not in self.weights:
                raise ValueError(f"Unknown quality dimension: {name}")
            self.weights[name] = float(value)
        total = sum(self.weights.values())
        if total <= 0:
            raise ValueError("Quality weights must sum to a positive value")
        self.weights = {name: value / total for name, value in self.weights.items()}

    def calculate(
        self, issues: List[Issue], total_records: int, total_cells: int | None = None
    ) -> dict:
        counts = Counter(issue.issue_type.lower() for issue in issues)

        # Missingness findings are column-level aggregates. Their metadata
        # carries the number of affected cells, which is the correct
        # denominator for completeness scoring.
        missing_affected = sum(
            int(issue.metadata.get("missing_cells", 0) or 0)
            for issue in issues
            if issue.issue_type.lower() == "missing"
        )

        denominators = {
            "missing": max(int(total_cells or total_records), 1),
            "duplicate": max(int(total_records), 1),
            "datatype": max(int(total_cells or total_records), 1),
            "outlier": max(int(total_records), 1),
            "conformance": max(int(total_records), 1),
            "temporal": max(int(total_records), 1),
            "plausibility": max(int(total_records), 1),
            "referential": max(int(total_records), 1),
            "bias": max(int(total_records), 1),
            "fairness": max(int(total_records), 1),
            "interoperability": max(int(total_records), 1),
            "drift": max(int(total_records), 1),
            "robustness": max(int(total_records), 1),
        }

        category_scores = {}
        for name, weight in self.weights.items():
            count = (
                missing_affected
                if name == "missing"
                else int(counts.get(name, 0))
            )
            denominator = denominators[name]
            error_rate = min(count / denominator, 1.0)
            category_scores[name] = {
                "count": count,
                "denominator": denominator,
                "error_rate": round(error_rate, 6),
                "score": round((1.0 - error_rate) * 100.0, 2),
                "weight": round(weight, 6),
            }

        overall = sum(item["score"] * item["weight"] for item in category_scores.values())
        return {
            "overall_score": round(overall, 2),
            "category_scores": category_scores,
            "grade": self._grade(overall),
        }

    @staticmethod
    def _grade(score: float) -> str:
        if score >= 95:
            return "Excellent"
        if score >= 85:
            return "Good"
        if score >= 70:
            return "Fair"
        if score >= 50:
            return "Poor"
        return "Critical"