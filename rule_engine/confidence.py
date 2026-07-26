"""
Confidence Aggregator.

Aggregates confidence scores from multiple detectors.
"""

from __future__ import annotations

from collections import defaultdict
from statistics import mean, variance
from typing import Dict, List

from models.issue import Issue


class ConfidenceAggregator:
    """
    Aggregate confidence scores using different strategies.

    Supported methods:
        - weighted_voting
        - bias_variance
        - dempster_shafer
        - meta_classifier
    """

    def __init__(self):
        self.supported_methods = {
            "weighted_voting": self.weighted_voting,
            "bias_variance": self.bias_variance,
            "dempster_shafer": self.dempster_shafer,
            "meta_classifier": self.meta_classifier,
        }

    def aggregate(
        self,
        issues: List[Issue],
        method: str = "weighted_voting",
    ) -> List[Issue]:
        """
        Aggregate confidence scores for duplicate issues.
        """

        if method not in self.supported_methods:
            raise ValueError(
                f"Unsupported confidence method: {method}"
            )

        grouped = self._group_issues(issues)

        aggregated = []

        for _, group in grouped.items():

            confidence = self.supported_methods[method](group)

            issue = group[0]
            issue.confidence = confidence

            aggregated.append(issue)

        return aggregated

    def _group_issues(
        self,
        issues: List[Issue],
    ) -> Dict[str, List[Issue]]:
        """
        Group issues referring to the same data point.
        """

        grouped = defaultdict(list)

        for issue in issues:

            key = (
                issue.table,
                issue.row_index,
                issue.column,
                issue.issue_type,
            )

            grouped[key].append(issue)

        return grouped

    ###############################################################
    # Method 1 : Weighted Voting
    ###############################################################

    def weighted_voting(
        self,
        issues: List[Issue],
    ) -> float:
        """
        Average detector confidence.
        """

        if not issues:
            return 0.0

        return mean(
            issue.confidence
            for issue in issues
        )

    ###############################################################
    # Method 2 : Bias-Variance Combination
    ###############################################################

    def bias_variance(
        self,
        issues: List[Issue],
    ) -> float:
        """
        Penalize detector disagreement.
        """

        if not issues:
            return 0.0

        scores = [
            issue.confidence
            for issue in issues
        ]

        if len(scores) == 1:
            return scores[0]

        avg = mean(scores)
        var = variance(scores)

        confidence = avg - (0.5 * var)

        return max(
            0.0,
            min(1.0, confidence),
        )

    ###############################################################
    # Method 3 : Dempster-Shafer Approximation
    ###############################################################

    def dempster_shafer(
        self,
        issues: List[Issue],
    ) -> float:
        """
        Approximate evidence combination.
        """

        if not issues:
            return 0.0

        belief = 1.0

        for issue in issues:
            belief *= (
                1 - issue.confidence
            )

        confidence = 1 - belief

        return max(
            0.0,
            min(1.0, confidence),
        )

    ###############################################################
    # Method 4 : Meta-Classifier Placeholder
    ###############################################################

    def meta_classifier(
        self,
        issues: List[Issue],
    ) -> float:
        """
        Placeholder for future ML meta-classifier.

        Later this method will load a trained
        classifier that combines detector outputs.
        """

        return self.weighted_voting(issues)