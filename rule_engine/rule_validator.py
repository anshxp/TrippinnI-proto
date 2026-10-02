"""Deterministic, healthcare-aware validation rules for Module 2."""

from __future__ import annotations

from typing import Any, Callable, List

import pandas as pd

from detectors.base_detector import BaseDetector
from models.detector_result import DetectorResult
from models.issue import Issue
from rule_engine.rules import (
    DATE_ORDER_PAIRS, IDENTIFIER_HIERARCHY, MIMIC_REQUIRED_FIELDS, NUMERIC_RANGES,
)


class RuleValidator(BaseDetector):
    """Run conservative row/table validation without mutating the dataset."""

    def __init__(self) -> None:
        super().__init__("RuleValidator")
        self.rules: List[Callable] = [
            self.check_missing_required,
            self.check_date_consistency,
            self.check_clinical_ranges,
            self.check_referential_integrity,
            self.check_identifier_format,
        ]

    def detect(self, dataset: Any, profile: Any) -> DetectorResult:
        result = DetectorResult(detector_name=self.name)
        for table_name, dataframe in dataset.items():
            if not isinstance(dataframe, pd.DataFrame):
                continue
            table_profile = profile if isinstance(profile, dict) else {}
            table_issues = []
            for rule in self.rules:
                table_issues.extend(rule(table_name, dataframe, table_profile))
            result.issues.extend(table_issues)
            result.statistics[table_name] = {"violations": len(table_issues)}
        result.statistics["rules_executed"] = len(self.rules)
        result.statistics["violations"] = result.issue_count
        return result

    # Backward-compatible name used by callers outside QualityDetector.
    validate = detect

    def check_missing_required(self, table: str, df: pd.DataFrame, profile: dict) -> List[Issue]:
        issues: List[Issue] = []
        required = MIMIC_REQUIRED_FIELDS.get(table.lower(), ())
        for column in required:
            if column not in df.columns:
                issues.append(Issue(
                    table=table, row_index=-1, column=column,
                    issue_type="conformance", severity="CRITICAL",
                    detector=self.name, original_value=None,
                    expected_value="Required field present", confidence=1.0,
                    metadata={"rule": "required_field", "scope": "table"},
                ))
                continue
            missing = df[column].isna()
            for idx in df.index[missing]:
                issues.append(Issue(
                    table=table, row_index=int(idx), column=column,
                    issue_type="conformance", severity="HIGH",
                    detector=self.name, original_value=None,
                    expected_value="Non-null required identifier/time field", confidence=1.0,
                    metadata={"rule": "required_field"},
                ))
        return issues

    def check_date_consistency(self, table: str, df: pd.DataFrame, profile: dict) -> List[Issue]:
        issues: List[Issue] = []
        for start_col, end_col in DATE_ORDER_PAIRS.get(table.lower(), ()):
            if start_col not in df.columns or end_col not in df.columns:
                continue
            start = pd.to_datetime(df[start_col], errors="coerce", format="mixed")
            end = pd.to_datetime(df[end_col], errors="coerce", format="mixed")
            invalid = start.notna() & end.notna() & (start > end)
            for idx in df.index[invalid]:
                issues.append(Issue(
                    table=table, row_index=int(idx), column=f"{start_col}/{end_col}",
                    issue_type="temporal", severity="HIGH", detector=self.name,
                    original_value={"start": str(df.at[idx, start_col]), "end": str(df.at[idx, end_col])},
                    expected_value=f"{start_col} <= {end_col}", confidence=1.0,
                    metadata={"rule": "date_order"},
                ))
        return issues

    def check_clinical_ranges(self, table: str, df: pd.DataFrame, profile: dict) -> List[Issue]:
        issues: List[Issue] = []
        for column, bounds in NUMERIC_RANGES.items():
            if column not in df.columns or bounds is None:
                continue
            lower, upper = bounds
            values = pd.to_numeric(df[column], errors="coerce")
            invalid = values.notna()
            if lower is not None:
                invalid &= values < lower
            if upper is not None:
                invalid |= values.notna() & (values > upper)
            for idx in df.index[invalid]:
                issues.append(Issue(
                    table=table, row_index=int(idx), column=column,
                    issue_type="plausibility", severity="HIGH", detector=self.name,
                    original_value=df.at[idx, column], expected_value={"min": lower, "max": upper},
                    confidence=0.98, metadata={"rule": "plausible_range"},
                ))
        return issues

    def check_referential_integrity(self, table: str, df: pd.DataFrame, profile: dict) -> List[Issue]:
        issues: List[Issue] = []
        for child, parents in IDENTIFIER_HIERARCHY:
            if child not in df.columns:
                continue
            child_present = df[child].notna()
            for parent in parents:
                if parent not in df.columns:
                    continue
                invalid = child_present & df[parent].isna()
                for idx in df.index[invalid]:
                    issues.append(Issue(
                        table=table, row_index=int(idx), column=parent,
                        issue_type="referential", severity="HIGH", detector=self.name,
                        original_value=None, expected_value=f"{parent} present when {child} is present",
                        confidence=1.0, metadata={"rule": "identifier_hierarchy", "child": child},
                    ))
        return issues

    def check_identifier_format(self, table: str, df: pd.DataFrame, profile: dict) -> List[Issue]:
        issues: List[Issue] = []
        columns = profile.get("columns", {}) if isinstance(profile, dict) else {}
        for column in df.columns:
            name = column.lower()
            if not (name.endswith("_id") or name == "itemid" or name == "subject_id"):
                continue
            if "code" in name:
                continue
            semantic = columns.get(column, {}).get("semantic_type")
            if semantic and semantic not in {"identifier", "numeric"}:
                continue
            values = pd.to_numeric(df[column], errors="coerce")
            non_null = df[column].notna()
            invalid = non_null & (values.isna() | (values < 0) | (values % 1 != 0))
            for idx in df.index[invalid]:
                issues.append(Issue(
                    table=table, row_index=int(idx), column=column,
                    issue_type="conformance", severity="HIGH", detector=self.name,
                    original_value=df.at[idx, column], expected_value="Non-negative integer identifier",
                    confidence=1.0, metadata={"rule": "identifier_format"},
                ))
        return issues