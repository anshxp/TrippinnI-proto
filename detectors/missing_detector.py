"""Missing-value detector with context-aware severity."""

from __future__ import annotations

from typing import Any

import pandas as pd

from detectors.base_detector import BaseDetector
from models.detector_result import DetectorResult
from models.issue import Issue


class MissingDetector(BaseDetector):
    def __init__(self) -> None:
        super().__init__("MissingDetector")

    def detect(self, dataset: Any, profile: Any) -> DetectorResult:
        result = DetectorResult(detector_name=self.name)
        columns_profile = profile.get("columns", {}) if isinstance(profile, dict) else {}

        for table_name, df in dataset.items():
            if not isinstance(df, pd.DataFrame):
                continue
            for column in df.columns:
                missing_count = int(df[column].isna().sum())
                if missing_count == 0:
                    continue

                column_profile = columns_profile.get(column, {})
                semantic = column_profile.get("semantic_type")
                null_percentage = float(
                    column_profile.get("null_percentage", 0.0)
                )
                all_null = bool(column_profile.get("all_null", False))

                # Missingness is a completeness measurement, not one
                # independent defect per empty cell. Keep one finding per
                # affected column and retain the affected-cell count for
                # dimension-level scoring.
                if all_null and semantic == "identifier":
                    severity = "CRITICAL"
                elif semantic == "identifier" and null_percentage >= 50.0:
                    severity = "HIGH"
                elif all_null:
                    severity = "HIGH"
                else:
                    severity = "MEDIUM"

                result.add_issue(Issue(
                    table=table_name,
                    row_index=-1,
                    column=column,
                    issue_type="missing",
                    severity=severity,
                    detector=self.name,
                    original_value=None,
                    expected_value="Non-null value",
                    confidence=1.0,
                    metadata={
                        "null_percentage": null_percentage,
                        "semantic_type": semantic,
                        "all_null_column": all_null,
                        "missing_cells": missing_count,
                        "rows_scanned": int(len(df)),
                        "aggregation": "column",
                    },
                ))

            result.statistics[table_name] = {
                "missing_cells": int(df.isna().sum().sum()),
                "missing_columns": int((df.isna().sum() > 0).sum()),
                "rows_scanned": int(len(df)),
            }
        return result