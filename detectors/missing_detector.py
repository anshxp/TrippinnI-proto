"""
Missing Value Detector.

Detects missing values in dataset tables.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from detectors.base_detector import BaseDetector
from models.detector_result import DetectorResult
from models.issue import Issue


class MissingDetector(BaseDetector):
    """
    Detect missing values in dataset tables.
    """

    def __init__(self) -> None:
        super().__init__("MissingDetector")

    def detect(self, dataset: Any, profile: Any) -> DetectorResult:
        """
        Detect missing values in every dataframe.

        Parameters
        ----------
        dataset
            Dictionary of pandas DataFrames.

        profile
            DatasetProfile from Module 1.
            (Reserved for future use.)

        Returns
        -------
        DetectorResult
        """

        result = DetectorResult(detector_name=self.name)

        for table_name, df in dataset.items():

            if not isinstance(df, pd.DataFrame):
                continue

            for column in df.columns:

                missing_rows = df[df[column].isna()].index

                for row_index in missing_rows:

                    issue = Issue(
                        table=table_name,
                        row_index=int(row_index),
                        column=column,
                        issue_type="missing",
                        severity="MEDIUM",
                        detector=self.name,
                        original_value=None,
                        expected_value="Non-null value",
                        confidence=1.0,
                    )

                    result.add_issue(issue)

            result.statistics[table_name] = {
                "missing_cells": int(df.isna().sum().sum()),
                "missing_columns": int((df.isna().sum() > 0).sum()),
            }

        return result