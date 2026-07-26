"""
Data Type Detector.

Detects values that do not conform to the expected data type.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from detectors.base_detector import BaseDetector
from models.detector_result import DetectorResult
from models.issue import Issue


class DatatypeDetector(BaseDetector):
    """
    Detect datatype inconsistencies.
    """

    def __init__(self) -> None:
        super().__init__("DatatypeDetector")

    def detect(self, dataset: Any, profile: Any) -> DetectorResult:
        """
        Detect datatype mismatches using the DatasetProfile.

        Parameters
        ----------
        dataset
            Dictionary of pandas DataFrames.

        profile
            DatasetProfile produced by Module 1.

        Returns
        -------
        DetectorResult
        """

        result = DetectorResult(detector_name=self.name)

        for table_name, df in dataset.items():

            if table_name not in profile.tables:
                continue

            table_profile = profile.tables[table_name]

            for column in df.columns:

                if column not in table_profile.columns:
                    continue

                expected_dtype = (
                    table_profile.columns[column]
                    .inferred_type
                )

                for row_index, value in df[column].items():

                    if pd.isna(value):
                        continue

                    if not self._is_valid(value, expected_dtype):

                        issue = Issue(
                            table=table_name,
                            row_index=int(row_index),
                            column=column,
                            issue_type="datatype",
                            severity="HIGH",
                            detector=self.name,
                            original_value=value,
                            expected_value=expected_dtype,
                            confidence=1.0,
                        )

                        result.add_issue(issue)

            result.statistics[table_name] = {
                "datatype_errors": result.issue_count
            }

        return result

    def _is_valid(self, value: Any, expected_type: str) -> bool:
        """
        Validate a value against an expected datatype.
        """

        try:

            if expected_type == "integer":
                int(value)

            elif expected_type == "float":
                float(value)

            elif expected_type == "boolean":
                if str(value).lower() not in {
                    "true",
                    "false",
                    "0",
                    "1",
                }:
                    return False

            elif expected_type == "datetime":
                pd.to_datetime(value)

            elif expected_type == "string":
                str(value)

            return True

        except Exception:
            return False