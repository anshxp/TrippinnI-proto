"""Automatically inferred healthcare consistency and plausibility validation."""

from __future__ import annotations

from typing import Any, Callable, List

import pandas as pd

from detectors.base_detector import BaseDetector
from models.detector_result import DetectorResult
from models.issue import Issue
from rule_engine.constraint_engine import ConstraintEngine
from rule_engine.rules import MIMIC_REQUIRED_FIELDS


class RuleValidator(BaseDetector):
    """Infer constraints from schema/data instead of table-specific value lists."""

    def __init__(self) -> None:
        super().__init__("RuleValidator")
        self.constraint_engine = ConstraintEngine()
        self.rules: List[Callable] = [
            self.check_missing_required,
            self.check_inferred_temporal_consistency,
            self.check_inferred_numeric_plausibility,
            self.check_inferred_identifier_hierarchy,
            self.check_identifier_format,
        ]

    def detect(self, dataset: Any, profile: Any) -> DetectorResult:
        result = DetectorResult(detector_name=self.name)
        for table_name, dataframe in dataset.items():
            if not isinstance(dataframe, pd.DataFrame):
                continue
            table_profile = profile if isinstance(profile, dict) else {}
            inferred = self.constraint_engine.infer(dataframe, table_profile)
            table_issues = []
            for rule in self.rules:
                table_issues.extend(rule(table_name, dataframe, table_profile, inferred))
            result.issues.extend(table_issues)
            result.statistics[table_name] = {
                "violations": len(table_issues),
                "inferred_constraints": inferred,
            }
        result.statistics["rules_executed"] = len(self.rules)
        result.statistics["violations"] = result.issue_count
        return result

    validate = detect

    def check_missing_required(self, table: str, df: pd.DataFrame, profile: dict, inferred: dict) -> List[Issue]:
        # Required structural fields are schema facts, not clinical ranges.
        issues: List[Issue] = []
        required = MIMIC_REQUIRED_FIELDS.get(table.lower(), ())
        for column in required:
            if column not in df.columns:
                issues.append(Issue(
                    table=table, row_index=-1, column=column,
                    issue_type="conformance", severity="CRITICAL", detector=self.name,
                    expected_value="Required field present", confidence=1.0,
                    metadata={"rule": "required_field", "scope": "table"},
                ))
                continue
            for idx in df.index[df[column].isna()]:
                issues.append(Issue(
                    table=table, row_index=int(idx), column=column,
                    issue_type="conformance", severity="HIGH", detector=self.name,
                    expected_value="Non-null required identifier/time field", confidence=1.0,
                    metadata={"rule": "required_field"},
                ))
        return issues

    def check_inferred_temporal_consistency(self, table, df, profile, inferred) -> List[Issue]:
        issues = []
        for constraint in inferred.get("temporal", []):
            start_col, end_col = constraint["start"], constraint["end"]
            start = pd.to_datetime(df[start_col], errors="coerce", format="mixed")
            end = pd.to_datetime(df[end_col], errors="coerce", format="mixed")
            invalid = start.notna() & end.notna() & (start > end)
            for idx in df.index[invalid]:
                issues.append(Issue(
                    table=table, row_index=int(idx), column=f"{start_col}/{end_col}",
                    issue_type="temporal", severity="HIGH", detector=self.name,
                    original_value={"start": str(df.at[idx, start_col]), "end": str(df.at[idx, end_col])},
                    expected_value=f"{start_col} <= {end_col}", confidence=constraint["confidence"],
                    metadata={"rule": "ml_inferred_temporal_order", "evidence": constraint["evidence"], "model": constraint.get("model", "unknown")},
                ))
        return issues

    def check_inferred_identifier_hierarchy(self, table, df, profile, inferred) -> List[Issue]:
        issues = []
        for constraint in inferred.get("hierarchy", []):
            child, parent = constraint["child"], constraint["parent"]
            invalid = df[child].notna() & df[parent].isna()
            for idx in df.index[invalid]:
                issues.append(Issue(
                    table=table, row_index=int(idx), column=parent,
                    issue_type="referential", severity="HIGH", detector=self.name,
                    original_value=None, expected_value=f"{parent} present when {child} is present",
                    confidence=constraint["confidence"],
                    metadata={"rule": "ml_inferred_functional_dependency", "child": child, "evidence": constraint["evidence"], "model": constraint.get("model", "unknown")},
                ))
        return issues

    def check_inferred_numeric_plausibility(self, table, df, profile, inferred) -> List[Issue]:
        issues = []
        for constraint in inferred.get("binary", []) + inferred.get("numeric", []):
            column = constraint["column"]
            values = pd.to_numeric(df[column], errors="coerce")
            invalid = values.notna() & ((values < constraint["lower"]) | (values > constraint["upper"]))
            for idx in df.index[invalid]:
                issues.append(Issue(
                    table=table, row_index=int(idx), column=column,
                    issue_type="plausibility", severity="MEDIUM", detector=self.name,
                    original_value=df.at[idx, column],
                    expected_value={"lower": constraint["lower"], "upper": constraint["upper"]},
                    confidence=constraint["confidence"],
                    metadata={
                        "rule": "ml_inferred_empirical_plausibility",
                        "evidence": constraint["evidence"],
                        "model": constraint.get("model", "unknown"),
                        "clinical_reference_range": False,
                    },
                ))
        return issues

    def check_identifier_format(self, table, df, profile, inferred) -> List[Issue]:
        issues = []
        columns = profile.get("columns", {}) if isinstance(profile, dict) else {}
        for column in df.columns:
            meta = columns.get(column, {})
            if meta.get("semantic_type") != "identifier":
                continue
            if meta.get("validation_type") not in {"integer", "float"} and not pd.api.types.is_numeric_dtype(df[column]):
                continue
            values = pd.to_numeric(df[column], errors="coerce")
            non_null = df[column].notna()
            invalid = non_null & (values.isna() | (values < 0) | (values % 1 != 0))
            for idx in df.index[invalid]:
                issues.append(Issue(
                    table=table, row_index=int(idx), column=column,
                    issue_type="conformance", severity="HIGH", detector=self.name,
                    original_value=df.at[idx, column], expected_value="Integer-like identifier",
                    confidence=0.99, metadata={"rule": "inferred_identifier_format"},
                ))
        return issues