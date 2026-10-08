"""Ensemble machine-learning constraint discovery for healthcare tables.

The learner combines independent unsupervised/relational models. Constraints
remain empirical observations of the supplied dataset; they are not clinical
reference ranges unless external clinical provenance is attached.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from rule_engine.ensemble.confidence import confidence as calibrate_confidence
from rule_engine.learners.numeric.isolation_forest import anomaly_scores as if_scores
from rule_engine.learners.numeric.lof import anomaly_scores as lof_scores
from rule_engine.learners.numeric.one_class_svm import anomaly_scores as svm_scores
from rule_engine.learners.relational.functional_dependency import dependency_score
from rule_engine.learners.temporal.temporal_distribution import directional_support


@dataclass(frozen=True)
class TemporalConstraint:
    start: str
    end: str
    confidence: float
    evidence: str
    model: str = "ensemble"


@dataclass(frozen=True)
class HierarchyConstraint:
    child: str
    parent: str
    confidence: float
    evidence: str
    model: str = "ensemble"


@dataclass(frozen=True)
class NumericConstraint:
    column: str
    lower: float | None
    upper: float | None
    confidence: float
    evidence: str
    model: str = "ensemble"


class ConstraintInferer:
    """Discover empirical constraints using multiple independent learners."""

    RANDOM_STATE = 42
    MIN_RELATION_ROWS = 20
    MIN_NUMERIC_ROWS = 30
    MIN_UNIQUE_NUMERIC = 3
    MIN_TEMPORAL_VARIATION = 1e-9

    def infer(self, dataframe: pd.DataFrame, profile: dict[str, Any] | None = None) -> dict[str, list[dict[str, Any]]]:
        profile = profile or {}
        return {
            "temporal": [asdict(x) for x in self.infer_temporal(dataframe, profile)],
            "hierarchy": [asdict(x) for x in self.infer_hierarchy(dataframe, profile)],
            "numeric": [asdict(x) for x in self.infer_numeric(dataframe, profile)],
            "binary": [asdict(x) for x in self.infer_binary(dataframe, profile)],
        }

    def infer_temporal(self, df: pd.DataFrame, profile: dict[str, Any]) -> list[TemporalConstraint]:
        datetime_columns = self._datetime_columns(df, profile)
        constraints: list[TemporalConstraint] = []
        for left, right in combinations(datetime_columns, 2):
            pair = pd.DataFrame({
                "left": pd.to_datetime(df[left], errors="coerce", format="mixed"),
                "right": pd.to_datetime(df[right], errors="coerce", format="mixed"),
            }).dropna()
            if len(pair) < self.MIN_RELATION_ROWS:
                continue
            delta = ((pair["right"].astype("int64") - pair["left"].astype("int64"))
                     .to_numpy(dtype=np.float64) / 1e9)
            if not np.isfinite(delta).all() or np.ptp(delta) <= self.MIN_TEMPORAL_VARIATION:
                continue

            direction = directional_support(delta)
            positive_rate, negative_rate = direction["positive"], direction["negative"]
            if positive_rate >= 0.95:
                start, end, dominance = left, right, positive_rate
            elif negative_rate >= 0.95:
                start, end, dominance = right, left, negative_rate
            else:
                continue

            X = np.column_stack([delta, np.abs(delta)])
            model = self._isolation_forest(len(delta))
            model.fit(X)
            ml_inlier_fraction = float(np.mean(model.predict(X) == 1))

            # Repeated clinical timestamps can make IsolationForest unstable:
            # the same small set of deltas may appear hundreds of times. Use a
            # robust MAD-based support estimate as the second ensemble signal,
            # rather than allowing the tree model alone to suppress an otherwise
            # strongly directional temporal relationship.
            median_delta = float(np.median(delta))
            mad = float(np.median(np.abs(delta - median_delta)))
            if mad > 0:
                robust_inlier_fraction = float(
                    np.mean(np.abs(delta - median_delta) <= 6.0 * mad)
                )
            else:
                robust_inlier_fraction = float(np.mean(delta == median_delta))

            inlier_fraction = max(ml_inlier_fraction, robust_inlier_fraction)
            if inlier_fraction < 0.50:
                continue

            conf = calibrate_confidence(inlier_fraction, dominance)
            constraints.append(TemporalConstraint(
                start=start,
                end=end,
                confidence=conf,
                evidence=(
                    "temporal ensemble: directional distribution + IsolationForest + robust MAD support; "
                    f"{inlier_fraction:.3f} ensemble inlier support, {dominance:.3f} directional support"
                ),
            ))
        return self._dedupe_temporal(constraints)

    def infer_hierarchy(self, df: pd.DataFrame, profile: dict[str, Any]) -> list[HierarchyConstraint]:
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
            child_unique, parent_unique = pair[child].nunique(), pair[parent].nunique()
            if child_unique <= parent_unique:
                continue

            fd = dependency_score(pair, child, parent)
            if fd < 0.995:
                continue

            child_counts = pair[child].value_counts()
            parent_counts = pair[parent].value_counts()
            pair_counts = pair.groupby([child, parent], dropna=False).size()
            parents_per_child = pair.groupby(child)[parent].nunique()
            features = [
                [float(child_counts.loc[c]), float(parent_counts.loc[p]),
                 float(n), float(parents_per_child.loc[c])]
                for (c, p), n in pair_counts.items()
            ]
            X = np.asarray(features, dtype=float)
            if len(X) < self.MIN_RELATION_ROWS:
                continue

            model = self._isolation_forest(len(X))
            model.fit(X)
            ml_inlier_fraction = float(np.mean(model.predict(X) == 1))

            # A strong functional dependency is already direct relational
            # evidence. IsolationForest is used as supporting evidence here,
            # not as a hard veto: small, repeated relationship tables can make
            # unsupervised tree models unstable even when the dependency is exact.
            relational_inlier_support = max(fd, ml_inlier_fraction)
            conf = calibrate_confidence(fd, relational_inlier_support)
            constraints.append(HierarchyConstraint(
                child=child,
                parent=parent,
                confidence=conf,
                evidence=(
                    "relational ensemble: functional-dependency analysis + IsolationForest; "
                    f"{fd:.3f} functional dependency, {ml_inlier_fraction:.3f} ML inlier support, "
                    f"{relational_inlier_support:.3f} combined relational support"
                ),
            ))
        return constraints

    def infer_binary(self, df: pd.DataFrame, profile: dict[str, Any]) -> list[NumericConstraint]:
        constraints: list[NumericConstraint] = []
        for column in df.columns:
            values = pd.to_numeric(df[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if values.empty or values.nunique() != 2:
                continue
            observed = sorted(values.unique().tolist())
            if all(float(v).is_integer() for v in observed) and observed == [0, 1]:
                constraints.append(NumericConstraint(
                    column=column, lower=0.0, upper=1.0, confidence=0.95,
                    evidence="observed two-state integer domain", model="empirical-domain",
                ))
        return constraints

    def infer_numeric(self, df: pd.DataFrame, profile: dict[str, Any]) -> list[NumericConstraint]:
        column_profiles = profile.get("columns", {})
        constraints: list[NumericConstraint] = []

        for column in df.columns:
            if column_profiles.get(column, {}).get("semantic_type") != "numeric":
                continue
            values = pd.to_numeric(df[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if len(values) < self.MIN_NUMERIC_ROWS or values.nunique() < self.MIN_UNIQUE_NUMERIC:
                continue

            X = values.to_numpy(dtype=float)
            learner_outputs = {
                "IsolationForest": if_scores(X, self.RANDOM_STATE),
                "LOF": lof_scores(X),
                "OneClassSVM": svm_scores(X),
            }
            inlier_masks = []
            for scores in learner_outputs.values():
                if not np.any(scores):
                    continue
                inlier_masks.append(scores <= float(np.quantile(scores, 0.95)))
            if not inlier_masks:
                continue

            stacked = np.vstack(inlier_masks)
            consensus_inlier = np.mean(stacked, axis=0) >= 0.5
            inlier_values = X[consensus_inlier]
            if len(inlier_values) < max(20, int(0.50 * len(X))):
                continue

            lower, upper = float(np.min(inlier_values)), float(np.max(inlier_values))
            if not np.isfinite(lower) or not np.isfinite(upper) or lower == upper:
                continue

            agreement = float(np.mean(stacked.mean(axis=0) >= 0.5))
            support = float(len(inlier_values) / len(X))
            conf = calibrate_confidence(support, agreement, base=0.40)
            models = ", ".join(learner_outputs)
            constraints.append(NumericConstraint(
                column=column, lower=lower, upper=upper, confidence=conf,
                evidence=(
                    f"numeric ensemble ({models}) learned empirical support: "
                    f"{support:.3f} consensus inlier support, {agreement:.3f} model agreement"
                ),
                model="IsolationForest+LOF+OneClassSVM",
            ))
        return constraints

    @staticmethod
    def _datetime_columns(df: pd.DataFrame, profile: dict[str, Any]) -> list[str]:
        columns = profile.get("columns", {})
        return [
            c for c in df.columns
            if columns.get(c, {}).get("semantic_type") == "datetime"
            or columns.get(c, {}).get("validation_type") == "datetime"
            or pd.api.types.is_datetime64_any_dtype(df[c])
        ]

    @classmethod
    def _isolation_forest(cls, n_rows: int) -> IsolationForest:
        return IsolationForest(
            n_estimators=200, contamination="auto", max_samples=min(256, n_rows),
            random_state=cls.RANDOM_STATE, n_jobs=-1,
        )

    @staticmethod
    def _dedupe_temporal(items: list[TemporalConstraint]) -> list[TemporalConstraint]:
        seen, output = set(), []
        for item in items:
            key = (item.start, item.end)
            if key not in seen:
                seen.add(key)
                output.append(item)
        return output
