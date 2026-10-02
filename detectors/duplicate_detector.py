"""Duplicate detection for exact rows and profiled candidate keys."""

from __future__ import annotations

from typing import Any

import pandas as pd

from detectors.base_detector import BaseDetector
from models.detector_result import DetectorResult
from models.issue import Issue


class DuplicateDetector(BaseDetector):
    """Detect exact duplicate rows and duplicate candidate primary keys."""

    def __init__(self) -> None:
        super().__init__("DuplicateDetector")

    def detect(self, dataset: Any, profile: Any) -> DetectorResult:
        result = DetectorResult(detector_name=self.name)
        for table_name, df in dataset.items():
            if not isinstance(df, pd.DataFrame):
                continue

            exact = df[df.duplicated(keep=False)]
            for row_index in exact.index:
                result.add_issue(Issue(
                    table=table_name, row_index=int(row_index), column=None,
                    issue_type="duplicate", severity="HIGH", detector=self.name,
                    expected_value="Unique Record", confidence=1.0,
                    metadata={"method": "exact_match"},
                ))

            primary_keys = []
            if isinstance(profile, dict):
                primary_keys = profile.get("keys", {}).get("primary_keys", [])
                if not primary_keys:
                    primary_keys = profile.get("primary_keys", [])

            key_duplicates = 0
            for column in primary_keys:
                if column not in df.columns:
                    continue
                duplicated = df[column].notna() & df[column].duplicated(keep=False)
                for row_index in df.index[duplicated]:
                    result.add_issue(Issue(
                        table=table_name, row_index=int(row_index), column=column,
                        issue_type="duplicate", severity="HIGH", detector=self.name,
                        original_value=df.at[row_index, column],
                        expected_value="Unique candidate key", confidence=1.0,
                        metadata={"method": "candidate_key"},
                    ))
                    key_duplicates += 1

            result.statistics[table_name] = {
                "exact_duplicates": int(len(exact)),
                "candidate_key_duplicates": int(key_duplicates),
                "fuzzy_duplicates": 0,
                "semantic_duplicates": 0,
            }
        return result