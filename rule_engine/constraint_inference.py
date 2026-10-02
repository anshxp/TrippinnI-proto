"""Automatic constraint inference for healthcare tables.

The engine derives validation rules from the observed schema, semantic types,
cardinality, functional dependencies, and value distributions. It does not
contain a table-name or column-name allowlist of clinical ranges.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TemporalConstraint:
    start: str
    end: str
    confidence: float
    evidence: str


@dataclass(frozen=True)
class HierarchyConstraint:
    child: str
    parent: str
    confidence: float
    evidence: str


@dataclass(frozen=True)
class NumericConstraint:
    column: str
    lower: float | None
    upper: float | None
    confidence: float
    evidence: str


class ConstraintInferer:
    """Infer structural and data-driven constraints without table-specific rules."""

    _START_TOKENS = {"start", "begin", "from", "in", "admit", "admission", "register", "reg", "birth", "onset"}
    _END_TOKENS = {"end", "stop", "to", "out", "discharge", "death", "expire", "finish"}
    _TIME_TOKENS = {"time", "date", "datetime", "timestamp"}
    _BINARY_TOKENS = {"flag", "indicator", "boolean", "bool", "status"}

    def infer(self, dataframe: pd.DataFrame, profile: dict[str, Any] | None = None) -> dict[str, list[dict[str, Any]]]:
        profile = profile or {}
        return {
            "temporal": [asdict(x) for x in self.infer_temporal(dataframe, profile)],
            "hierarchy": [asdict(x) for x in self.infer_hierarchy(dataframe, profile)],
            "numeric": [asdict(x) for x in self.infer_numeric(dataframe, profile)],
            "binary": [asdict(x) for x in self.infer_binary(dataframe, profile)],
        }

    def infer_temporal(self, df: pd.DataFrame, profile: dict[str, Any]) -> list[TemporalConstraint]:
        datetime_columns = []
        column_profiles = profile.get("columns", {})
        for column in df.columns:
            meta = column_profiles.get(column, {})
            if meta.get("semantic_type") == "datetime" or meta.get("validation_type") == "datetime":
                datetime_columns.append(column)
        constraints = []
        for left, right in combinations(datetime_columns, 2):
            left_score = self._temporal_role_score(left, "start")
            right_score = self._temporal_role_score(right, "end")
            reverse_left = self._temporal_role_score(left, "end")
            reverse_right = self._temporal_role_score(right, "start")
            if left_score > 0 and right_score > 0 and self._same_temporal_context(left, right):
                confidence = min(0.99, 0.55 + 0.08 * left_score + 0.08 * right_score)
                constraints.append(TemporalConstraint(left, right, confidence, "semantic-name role inference"))
            elif reverse_left > 0 and reverse_right > 0 and self._same_temporal_context(left, right):
                confidence = min(0.99, 0.55 + 0.08 * reverse_left + 0.08 * reverse_right)
                constraints.append(TemporalConstraint(right, left, confidence, "semantic-name role inference"))
        return self._dedupe_temporal(constraints)

    def infer_hierarchy(self, df: pd.DataFrame, profile: dict[str, Any]) -> list[HierarchyConstraint]:
        column_profiles = profile.get("columns", {})
        identifiers = [
            c for c in df.columns
            if column_profiles.get(c, {}).get("semantic_type") == "identifier"
        ]
        constraints = []
        for child, parent in combinations(identifiers, 2):
            a = df[[child, parent]].dropna()
            if len(a) < 10:
                continue
            child_unique = a[child].nunique(dropna=True)
            parent_unique = a[parent].nunique(dropna=True)
            if child_unique <= parent_unique:
                continue
            grouped = a.groupby(child, dropna=True)[parent].nunique(dropna=True)
            if grouped.empty or grouped.max() > 1:
                continue
            dependency = float((grouped == 1).mean())
            cardinality_ratio = child_unique / max(parent_unique, 1)
            confidence = min(0.99, 0.55 + 0.35 * dependency + 0.05 * min(np.log1p(cardinality_ratio), 2.0))
            constraints.append(HierarchyConstraint(child, parent, confidence, "functional-dependency and cardinality inference"))
        return constraints

    def infer_binary(self, df: pd.DataFrame, profile: dict[str, Any]) -> list[NumericConstraint]:
        """Infer binary-domain constraints from semantic type and observed support."""
        column_profiles = profile.get("columns", {})
        constraints = []
        for column in df.columns:
            meta = column_profiles.get(column, {})
            values = pd.to_numeric(df[column], errors="coerce").dropna()
            if values.empty or values.nunique() > 2:
                continue
            name_tokens = set(column.lower().replace("-", "_").split("_"))
            semantic = meta.get("semantic_type")
            if semantic == "boolean" or name_tokens & self._BINARY_TOKENS:
                observed = sorted(values.unique().tolist())
                if all(float(v).is_integer() for v in observed):
                    lower = min(observed)
                    upper = max(observed)
                    if lower == 0 and upper == 1:
                        constraints.append(NumericConstraint(
                            column, 0.0, 1.0, 0.95,
                            "binary semantic type/flag inference"
                        ))
        return constraints
    def infer_numeric(self, df: pd.DataFrame, profile: dict[str, Any]) -> list[NumericConstraint]:
        column_profiles = profile.get("columns", {})
        constraints = []
        for column in df.columns:
            meta = column_profiles.get(column, {})
            if meta.get("semantic_type") != "numeric":
                continue
            values = pd.to_numeric(df[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if len(values) < 30 or values.nunique() < 3:
                continue
            q1, q3 = values.quantile([0.25, 0.75])
            iqr = float(q3 - q1)
            if not np.isfinite(iqr) or iqr <= 0:
                continue
            # Conservative empirical envelope. This is an inferred plausibility
            # boundary, not a clinical reference range.
            lower = float(q1 - 6.0 * iqr)
            upper = float(q3 + 6.0 * iqr)
            constraints.append(NumericConstraint(
                column, lower, upper, 0.60, "robust empirical distribution envelope"
            ))
        return constraints

    @classmethod
    def _temporal_role_score(cls, name: str, role: str) -> int:
        normalized = name.lower().replace("-", "_")
        tokens = {token for token in normalized.split("_") if token}
        tokens.discard("datetime")
        tokens.discard("timestamp")
        score = 0
        if tokens & cls._TIME_TOKENS:
            score += 1
        target = cls._START_TOKENS if role == "start" else cls._END_TOKENS
        score += 2 * len(tokens & target)
        if any(token in normalized for token in target):
            score += 2
        return score

    @staticmethod
    def _same_temporal_context(left: str, right: str) -> bool:
        left_tokens = set(left.lower().replace("-", "_").split("_"))
        right_tokens = set(right.lower().replace("-", "_").split("_"))
        shared = left_tokens & right_tokens
        shared -= {"time", "date", "datetime", "timestamp", "start", "stop", "end", "out", "in"}
        return bool(shared) or len(left_tokens) <= 2 or len(right_tokens) <= 2

    @staticmethod
    def _dedupe_temporal(items: list[TemporalConstraint]) -> list[TemporalConstraint]:
        seen = set()
        output = []
        for item in items:
            key = (item.start, item.end)
            if key not in seen:
                seen.add(key)
                output.append(item)
        return output