"""
Issue model for Module 2 (Data Quality Detection).

Represents a single data quality issue detected within a dataset.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(slots=True)
class Issue:
    """
    Represents a single data quality issue.

    Attributes
    ----------
    table:
        Name of the table where the issue was detected.

    row_index:
        Index of the affected row.

    column:
        Name of the affected column.
        None if the issue affects an entire row or multiple columns.

    issue_type:
        Type of issue.
        Example:
            - missing
            - duplicate
            - datatype
            - outlier
            - inconsistency

    severity:
        LOW | MEDIUM | HIGH | CRITICAL

    detector:
        Name of the detector that generated this issue.

    original_value:
        Value present in the dataset.

    expected_value:
        Expected value or datatype if applicable.

    confidence:
        Confidence score in range [0,1].

    metadata:
        Additional detector-specific information.
    """

    table: str
    row_index: int
    column: Optional[str]

    issue_type: str
    severity: str
    detector: str

    original_value: Any = None
    expected_value: Any = None

    confidence: float = 1.0

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the issue into a serializable dictionary."""

        return {
            "table": self.table,
            "row_index": self.row_index,
            "column": self.column,
            "issue_type": self.issue_type,
            "severity": self.severity,
            "detector": self.detector,
            "original_value": self.original_value,
            "expected_value": self.expected_value,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }