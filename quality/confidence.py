"""
Confidence Aggregator.

Aggregates confidence scores from multiple detection methods.
"""

from __future__ import annotations

from typing import Dict, List

from models.issue import Issue


class ConfidenceAggregator:
    """
    Aggregate confidence scores from multiple detectors.
    """

    def __init__(self) -> None:
        pass

    def aggregate(self, issues: List[Issue]) -> List[Issue]:
        """
        Aggregate confidence for duplicate issues.

        Parameters
        ----------
        issues
            Issues produced by all detectors.

        Returns
        -------
        List[Issue]
            Issues with updated confidence scores.
        """

        grouped = self._group_issues(issues)

        aggregated: List[Issue] = []

        for _, group in grouped.items():

            confidence = self._weighted_vote(group)

            issue = group[0]
            issue.confidence = confidence

            aggregated.append(issue)

        return aggregated

    def _group_issues(self, issues: List[Issue]) -> Dict[str, List[Issue]]:
        """
        Group issues referring to the same observation.
        """

        grouped: Dict[str, List[Issue]] = {}

        for issue in issues:

            key = (
                f"{issue.table}:"
                f"{issue.row_index}:"
                f"{issue.column}:"
                f"{issue.issue_type}"
            )

            grouped.setdefault(key, []).append(issue)

        return grouped

    def _weighted_vote(self, issues: List[Issue]) -> float:
        """
        Weighted voting confidence.

        Placeholder implementation.
        """

        if not issues:
            return 0.0

        return sum(i.confidence for i in issues) / len(issues)

    def bias_variance(self, issues: List[Issue]) -> float:
        """
        Placeholder for Bias-Variance aggregation.
        """
        raise NotImplementedError

    def dempster_shafer(self, issues: List[Issue]) -> float:
        """
        Placeholder for Dempster-Shafer aggregation.
        """
        raise NotImplementedError

    def meta_classifier(self, issues: List[Issue]) -> float:
        """
        Placeholder for Meta-Classifier aggregation.
        """
        raise NotImplementedError