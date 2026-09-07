"""Datatype validation against the Module 1 profiling report."""

from __future__ import annotations

from typing import Any

import pandas as pd

from detectors.base_detector import BaseDetector
from models.detector_result import DetectorResult
from models.issue import Issue


class DatatypeDetector(BaseDetector):
    """Detect values that do not conform to profiled physical types."""

    def __init__(self) -> None:
        super().__init__("DatatypeDetector")

    def detect(self, dataset: Any, profile: Any) -> DetectorResult:
        result = DetectorResult(detector_name=self.name)
        if not isinstance(profile, dict):
            return result

        columns = profile.get("columns", {})
        for table_name, df in dataset.items():
            if not isinstance(df, pd.DataFrame):
                continue

            errors = 0
            for column in df.columns:
                column_profile = columns.get(column)
                if not isinstance(column_profile, dict):
                    continue
                expected_type = column_profile.get("validation_type", "string")

                for row_index, value in df[column].items():
                    if pd.isna(value):
                        continue
                    if not self._is_valid(value, expected_type):
                        result.add_issue(Issue(
                            table=table_name,
                            row_index=int(row_index),
                            column=column,
                            issue_type="datatype",
                            severity="HIGH",
                            detector=self.name,
                            original_value=value,
                            expected_value=expected_type,
                            confidence=1.0,
                        ))
                        errors += 1

            result.statistics[table_name] = {"datatype_errors": errors}

        return result

    def _is_valid(self, value: Any, expected_type: str) -> bool:
        try:
            if expected_type == "integer":
                if isinstance(value, bool):
                    return False
                numeric = float(value)
                return numeric.is_integer()
            if expected_type == "float":
                float(value)
                return True
            if expected_type == "boolean":
                return str(value).strip().lower() in {
                    "true", "false", "yes", "no", "y", "n", "0", "1"
                }
            if expected_type == "datetime":
                pd.to_datetime(value, errors="raise", format="mixed")
                return True
            if expected_type == "string":
                return isinstance(value, str)
            return True
        except (TypeError, ValueError, OverflowError):
            return False
