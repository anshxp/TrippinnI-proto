"""Machine-learning-assisted constraint discovery for healthcare tables.

Constraints are learned from the observed data rather than from a MIMIC-specific
column allowlist. The learner is deliberately conservative: unsupervised models
discover empirical structure/plausibility, while structural schema facts remain
separate.

The learned constraints are *dataset-derived* and must not be interpreted as
clinical reference ranges unless an external clinical knowledge source is added.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


@dataclass(frozen=True)
class TemporalConstraint:
    start: str
    end: str
    confidence: float
    evidence: str
    model: str = "IsolationForest"


@dataclass(frozen=True)
class HierarchyConstraint:
    child: str
    parent: str
    confidence: float
    evidence: str
    model: str = "IsolationForest"


@dataclass(frozen=True)
class NumericConstraint:
    column: str
    lower: float | None
    upper: float | None
    confidence: float
    evidence: str
    model: str = "IsolationForest"


class ConstraintInferer:
    """Discover empirical constraints with unsupervised machine learning."""

    RANDOM_STATE = 42
    MIN_RELATION_ROWS = 20
    MIN_NUMERIC_ROWS = 30
    MIN_UNIQUE_NUMERIC = 3
    MIN_TEMPORAL_VARIATION = 1e-9

    def infer(
        self,
        dataframe: pd.DataFrame,
        profile: dict[str, Any] | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        profile = profile or {}
        return {
            "temporal": [asdict(x) for x in self.infer_temporal(dataframe, profile)],
            "hierarchy": [asdict(x) for x in self.infer_hierarchy(dataframe, profile)],
            "numeric": [asdict(x) for x in self.infer_numeric(dataframe, profile)],
            "binary": [asdict(x) for x in self.infer_binary(dataframe, profile)],
        }

    def infer_temporal(
        self,
        df: pd.DataFrame,
        profile: dict[str, Any],
    ) -> list[TemporalConstraint]:
        """Learn temporal ordering from timestamp-pair distributions.

        No start/end vocabulary is required. For each timestamp pair we model
        the observed signed time difference with IsolationForest. A candidate
        ordering is emitted only when the model regards a highly dominant side
        of zero as the normal population and there is enough variation to make
        the direction identifiable.
        """
        datetime_columns = self._datetime_columns(df, profile)
        constraints: list[TemporalConstraint] = []

        for left, right in combinations(datetime_columns, 2):
            pair = pd.DataFrame({
                "left": pd.to_datetime(df[left], errors="coerce", format="mixed"),
                "right": pd.to_datetime(df[right], errors="coerce", format="mixed"),
            }).dropna()

            if len(pair) < self.MIN_RELATION_ROWS:
                continue

            delta = (
                pair["right"].astype("int64") - pair["left"].astype("int64")
            ).to_numpy(dtype=np.float64) / 1e9

            if not np.isfinite(delta).all() or np.ptp(delta) <= self.MIN_TEMPORAL_VARIATION:
                continue

            # Two features prevent the model from treating a large duration
            # and a small duration as equivalent just because both have the
            # same sign.
            X = np.column_stack([delta, np.abs(delta)])
            model = self._isolation_forest(len(delta))
            model.fit(X)
            inlier = model.predict(X) == 1
            if inlier.sum() < max(10, int(0.5 * len(delta))):
                continue

            normal_delta = delta[inlier]
            positive_rate = float(np.mean(normal_delta >= 0))
            negative_rate = float(np.mean(normal_delta <= 0))

            if positive_rate >= 0.98:
                direction = (left, right)
                dominance = positive_rate
            elif negative_rate >= 0.98:
                direction = (right, left)
                dominance = negative_rate
            else:
                continue

            inlier_fraction = float(np.mean(inlier))
            confidence = min(
                0.995,
                0.50 + 0.30 * dominance + 0.20 * inlier_fraction,
            )
            constraints.append(TemporalConstraint(
                start=direction[0],
                end=direction[1],
                confidence=confidence,
                evidence=(
                    "unsupervised temporal-order learning: "
                    f"{inlier_fraction:.3f} inlier support, "
                    f"{dominance:.3f} directional dominance"
                ),
            ))

        return self._dedupe_temporal(constraints)

    def infer_hierarchy(
        self,
        df: pd.DataFrame,
        profile: dict[str, Any],
    ) -> list[HierarchyConstraint]:
        """Learn identifier dependencies using an ML anomaly model.

        For each candidate identifier pair, the model learns the normal
        structure of (child frequency, parent frequency, pair frequency,
        parents-per-child). A candidate is retained only when the learned
        normal population is also functionally dependent: one child maps to
        one parent. This avoids encoding MIMIC identifiers such as
        stay_id -> hadm_id -> subject_id directly in code.
        """
        column_profiles = profile.get("columns", {})
        identifiers = [
            c for c in df.columns
            if column_profiles.get(c, {}).get("semantic_type") == "identifier"
        ]

        constraints: list[HierarchyConstraint] = []

        for child, parent in combinations(identifiers, 2):
            pair = df[[child, parent]].dropna()
            if len(pair) < self.MIN_RELATION_ROWS:
                continue

            child_counts = pair[child].value_counts(dropna=False)
            parent_counts = pair[parent].value_counts(dropna=False)
            pair_counts = pair.groupby([child, parent], dropna=False).size()

            parents_per_child = pair.groupby(child, dropna=False)[parent].nunique()
            if parents_per_child.empty:
                continue

            # The child must have at least as many distinct values as its
            # proposed parent. Otherwise the direction is not hierarchical.
            child_unique = int(child_counts.size)
            parent_unique = int(parent_counts.size)
            if child_unique <= parent_unique:
                continue

            features = []
            for (child_value, parent_value), pair_count in pair_counts.items():
                features.append([
                    float(child_counts.loc[child_value]),
                    float(parent_counts.loc[parent_value]),
                    float(pair_count),
                    float(parents_per_child.loc[child_value]),
                ])

            X = np.asarray(features, dtype=np.float64)
            if len(X) < self.MIN_RELATION_ROWS:
                continue

            model = self._isolation_forest(len(X))
            model.fit(X)
            inlier = model.predict(X) == 1
            if inlier.mean() < 0.80:
                continue

            functional_dependency = float((parents_per_child == 1).mean())
            if functional_dependency < 0.995:
                continue

            cardinality_ratio = child_unique / max(parent_unique, 1)
            confidence = min(
                0.995,
                0.50
                + 0.25 * functional_dependency
                + 0.15 * float(inlier.mean())
                + 0.10 * min(np.log1p(cardinality_ratio) / 3.0, 1.0),
            )
            constraints.append(HierarchyConstraint(
                child=child,
                parent=parent,
                confidence=confidence,
                evidence=(
                    "unsupervised relational-pattern learning: "
                    f"{functional_dependency:.3f} functional dependency, "
                    f"{float(inlier.mean()):.3f} ML inlier support"
                ),
            ))

        return constraints

    def infer_binary(
        self,
        df: pd.DataFrame,
        profile: dict[str, Any],
    ) -> list[NumericConstraint]:
        """Learn binary domains from observed support, not field names."""
        column_profiles = profile.get("columns", {})
        constraints: list[NumericConstraint] = []

        for column in df.columns:
            meta = column_profiles.get(column, {})
            values = (
                pd.to_numeric(df[column], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            if values.empty or values.nunique() > 2:
                continue

            observed = sorted(values.unique().tolist())
            if (
                all(float(v).is_integer() for v in observed)
                and observed == [0, 1]
            ):
                constraints.append(NumericConstraint(
                    column=column,
                    lower=0.0,
                    upper=1.0,
                    confidence=0.95,
                    evidence="observed two-state integer domain",
                    model="empirical-domain",
                ))

        return constraints

    def infer_numeric(
        self,
        df: pd.DataFrame,
        profile: dict[str, Any],
    ) -> list[NumericConstraint]:
        """Learn a normal numeric support region with IsolationForest."""
        column_profiles = profile.get("columns", {})
        constraints: list[NumericConstraint] = []

        for column in df.columns:
            meta = column_profiles.get(column, {})
            if meta.get("semantic_type") != "numeric":
                continue

            values = (
                pd.to_numeric(df[column], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            if len(values) < self.MIN_NUMERIC_ROWS:
                continue
            if values.nunique() < self.MIN_UNIQUE_NUMERIC:
                continue

            X = values.to_numpy(dtype=np.float64).reshape(-1, 1)
            model = self._isolation_forest(len(X))
            model.fit(X)
            inlier = model.predict(X) == 1
            inlier_values = values.iloc[np.flatnonzero(inlier)]

            if len(inlier_values) < max(20, int(0.50 * len(values))):
                continue

            lower = float(inlier_values.min())
            upper = float(inlier_values.max())
            inlier_fraction = float(inlier.mean())

            if not np.isfinite(lower) or not np.isfinite(upper) or lower == upper:
                continue

            # Confidence is explicitly a model/data confidence, not clinical
            # truth. A clinical range requires a separate reference source.
            confidence = min(0.90, 0.45 + 0.45 * inlier_fraction)
            constraints.append(NumericConstraint(
                column=column,
                lower=lower,
                upper=upper,
                confidence=confidence,
                evidence=(
                    "IsolationForest learned empirical support region: "
                    f"{inlier_fraction:.3f} inlier fraction"
                ),
            ))

        return constraints

    @staticmethod
    def _datetime_columns(
        df: pd.DataFrame,
        profile: dict[str, Any],
    ) -> list[str]:
        columns = profile.get("columns", {})
        output = []
        for column in df.columns:
            meta = columns.get(column, {})
            if (
                meta.get("semantic_type") == "datetime"
                or meta.get("validation_type") == "datetime"
                or pd.api.types.is_datetime64_any_dtype(df[column])
            ):
                output.append(column)
        return output

    @classmethod
    def _isolation_forest(cls, n_rows: int) -> IsolationForest:
        return IsolationForest(
            n_estimators=100,
            contamination="auto",
            max_samples=min(256, n_rows),
            random_state=cls.RANDOM_STATE,
            n_jobs=-1,
        )

    @staticmethod
    def _dedupe_temporal(
        items: list[TemporalConstraint],
    ) -> list[TemporalConstraint]:
        seen = set()
        output = []
        for item in items:
            key = (item.start, item.end)
            if key not in seen:
                seen.add(key)
                output.append(item)
        return output
