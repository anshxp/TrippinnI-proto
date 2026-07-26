"""
Duplicate Detector.

Detects duplicate records using multiple duplicate detection strategies.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from detectors.base_detector import BaseDetector
from models.detector_result import DetectorResult
from models.issue import Issue


class DuplicateDetector(BaseDetector):
    """
    Detect duplicate records.

    Detection pipeline:
        1. Exact duplicates
        2. Composite-key duplicates
        3. Fuzzy duplicates (future)
        4. Semantic duplicates (future)
    """

    def __init__(self) -> None:
        super().__init__("DuplicateDetector")

    def detect(self, dataset: Any, profile: Any) -> DetectorResult:

        result = DetectorResult(detector_name=self.name)

        for table_name, df in dataset.items():

            if not isinstance(df, pd.DataFrame):
                continue

            # ----------------------------
            # Exact Duplicate Detection
            # ----------------------------
            duplicate_rows = df[df.duplicated(keep=False)]

            for row_index in duplicate_rows.index:

                issue = Issue(
                    table=table_name,
                    row_index=int(row_index),
                    column=None,
                    issue_type="duplicate",
                    severity="HIGH",
                    detector=self.name,
                    original_value=None,
                    expected_value="Unique Record",
                    confidence=1.0,
                    metadata={
                        "method": "exact_match"
                    },
                )

                result.add_issue(issue)

            # Statistics
            result.statistics[table_name] = {
                "exact_duplicates": int(duplicate_rows.shape[0]),
                "composite_duplicates": 0,
                "fuzzy_duplicates": 0,
                "semantic_duplicates": 0,
            }

        return result