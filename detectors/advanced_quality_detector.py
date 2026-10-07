"""Advanced EHR quality checks: representation, fairness, interoperability, drift, shift and robustness.

These checks are dataset-agnostic. They use observed schema/data evidence and do not
claim clinical correctness or protected-group fairness without an appropriate reference.
"""
from __future__ import annotations

from typing import Any
import math
import re

import numpy as np
import pandas as pd

from detectors.base_detector import BaseDetector
from models.detector_result import DetectorResult
from models.issue import Issue


class AdvancedQualityDetector(BaseDetector):
    """Detect higher-order EHR quality risks from observed data distributions."""

    MAX_ISSUES = 20
    MIN_GROUP_SIZE = 30
    MIN_GROUP_SHARE = 0.01
    REPRESENTATION_THRESHOLD = 0.05
    FAIRNESS_GAP_THRESHOLD = 0.10
    DRIFT_THRESHOLD = 0.20
    SHIFT_THRESHOLD = 0.20
    ROBUSTNESS_RANGE_THRESHOLD = 0.20

    DEMOGRAPHIC_PATTERNS = (
        "sex", "gender", "race", "ethnicity", "language", "payer",
        "insurance", "marital", "religion", "age_group", "age_band",
    )
    TIME_PATTERNS = (
        "time", "date", "timestamp", "datetime", "charttime", "admittime",
        "dischtime", "deathtime", "starttime", "endtime",
    )
    UNIT_PATTERNS = ("unit", "uom", "units", "valueuom")
    SEMANTIC_ALIASES = {
        "heart_rate": {"heart_rate", "heartrate", "hr", "pulse", "pulse_rate"},
        "respiratory_rate": {"respiratory_rate", "respiratoryrate", "rr", "resp_rate"},
        "oxygen_saturation": {"oxygen_saturation", "oxygensaturation", "spo2", "o2sat"},
        "temperature": {"temperature", "temp", "body_temperature"},
        "blood_pressure": {"blood_pressure", "bp", "bloodpressure"},
        "systolic_blood_pressure": {"sbp", "systolic_bp", "systolicbloodpressure"},
        "diastolic_blood_pressure": {"dbp", "diastolic_bp", "diastolicbloodpressure"},
    }

    def __init__(self) -> None:
        super().__init__("AdvancedQualityDetector")

    def detect(self, dataset: Any, profile: Any) -> DetectorResult:
        result = DetectorResult(detector_name=self.name)
        tables = dataset if isinstance(dataset, dict) else {"dataset": dataset}

        for table, df in tables.items():
            if not isinstance(df, pd.DataFrame) or len(df) < 2:
                continue

            table_issues: list[Issue] = []
            table_issues.extend(self._representation(table, df))
            table_issues.extend(self._fairness(table, df))
            table_issues.extend(self._interoperability(table, df))
            table_issues.extend(self._drift(table, df))
            table_issues.extend(self._robustness(table, df))

            result.issues.extend(table_issues[: self.MAX_ISSUES])
            result.statistics[table] = {
                "representation": self._representation_stats(df),
                "fairness": self._fairness_stats(df),
                "interoperability": self._interoperability_stats(df),
                "drift": self._drift_stats(df),
                "robustness": self._robustness_stats(df),
            }

        result.statistics["issues"] = result.issue_count
        return result

    @staticmethod
    def _candidate_groups(df: pd.DataFrame) -> list[str]:
        columns = []
        for col in df.columns:
            name = str(col).lower()
            if any(token in name for token in AdvancedQualityDetector.DEMOGRAPHIC_PATTERNS):
                series = df[col].dropna()
                if 2 <= series.nunique() <= 20:
                    columns.append(col)
        return columns

    def detect_distribution_shift(self, dataset: Any, reference_dataset: Any) -> DetectorResult:
        """Compare a current/reference dataset when both are supplied explicitly.

        This is separate from temporal drift because deployment-vs-training
        shift requires an external reference dataset.
        """
        result = DetectorResult(detector_name=self.name + ".DistributionShift")
        current = dataset if isinstance(dataset, dict) else {"dataset": dataset}
        reference = reference_dataset if isinstance(reference_dataset, dict) else {"dataset": reference_dataset}

        for table, current_df in current.items():
            ref_df = reference.get(table)
            if not isinstance(current_df, pd.DataFrame) or not isinstance(ref_df, pd.DataFrame):
                continue
            common = [c for c in current_df.columns if c in ref_df.columns]
            metrics = {}
            for col in common:
                if pd.api.types.is_numeric_dtype(current_df[col]) and pd.api.types.is_numeric_dtype(ref_df[col]):
                    metrics[col] = self._numeric_metric(current_df[col], ref_df[col])
                else:
                    metrics[col] = self._categorical_metric(current_df[col], ref_df[col])
            if not metrics:
                continue
            column = max(metrics, key=metrics.get)
            score = metrics[column]
            result.statistics[table] = {
                "reference_rows": len(ref_df),
                "current_rows": len(current_df),
                "max_shift": round(float(score), 6),
                "top_columns": dict(sorted(metrics.items(), key=lambda x: x[1], reverse=True)[:10]),
            }
            if score >= self.SHIFT_THRESHOLD:
                result.issues.append(Issue(
                    table=table, row_index=-1, column=column,
                    issue_type="distribution_shift", severity="MEDIUM", detector=result.detector_name,
                    original_value=round(float(score), 6),
                    expected_value=f"Reference/current divergence < {self.SHIFT_THRESHOLD}",
                    confidence=min(0.99, 0.65 + min(float(score), 0.34)),
                    metadata={
                        "dimension": "distribution_shift",
                        "rule": "reference_distribution_divergence",
                        "divergence": round(float(score), 6),
                        "metric": "quantile_standardized_distance" if pd.api.types.is_numeric_dtype(current_df[column]) else "total_variation_distance",
                        "reference_rows": len(ref_df),
                        "current_rows": len(current_df),
                    },
                ))
        result.statistics["issues"] = result.issue_count
        return result

    def detect_distribution_shift(self, dataset: Any, reference_dataset: Any) -> DetectorResult:
        """Compare current data with an explicitly supplied reference dataset."""
        result = DetectorResult(detector_name=self.name + ".DistributionShift")
        current = dataset if isinstance(dataset, dict) else {"dataset": dataset}
        reference = reference_dataset if isinstance(reference_dataset, dict) else {"dataset": reference_dataset}
        for table, current_df in current.items():
            ref_df = reference.get(table)
            if not isinstance(current_df, pd.DataFrame) or not isinstance(ref_df, pd.DataFrame):
                continue
            common = [c for c in current_df.columns if c in ref_df.columns]
            metrics = {}
            for col in common:
                if pd.api.types.is_numeric_dtype(current_df[col]) and pd.api.types.is_numeric_dtype(ref_df[col]):
                    metrics[col] = self._numeric_metric(current_df[col], ref_df[col])
                else:
                    metrics[col] = self._categorical_metric(current_df[col], ref_df[col])
            if not metrics:
                continue
            column = max(metrics, key=metrics.get)
            score = float(metrics[column])
            result.statistics[table] = {
                "reference_rows": len(ref_df),
                "current_rows": len(current_df),
                "max_shift": round(score, 6),
                "top_columns": dict(sorted(metrics.items(), key=lambda x: x[1], reverse=True)[:10]),
            }
            if score >= self.SHIFT_THRESHOLD:
                result.issues.append(Issue(
                    table=table, row_index=-1, column=column,
                    issue_type="distribution_shift", severity="MEDIUM", detector=result.detector_name,
                    original_value=round(score, 6),
                    expected_value=f"Reference/current divergence < {self.SHIFT_THRESHOLD}",
                    confidence=min(0.99, 0.65 + min(score, 0.34)),
                    metadata={
                        "dimension": "distribution_shift",
                        "rule": "reference_distribution_divergence",
                        "divergence": round(score, 6),
                        "metric": "quantile_standardized_distance" if pd.api.types.is_numeric_dtype(current_df[column]) else "total_variation_distance",
                        "reference_rows": len(ref_df),
                        "current_rows": len(current_df),
                    },
                ))
        result.statistics["issues"] = result.issue_count
        return result

    def _representation(self, table: str, df: pd.DataFrame) -> list[Issue]:
        issues = []
        n = len(df)
        for col in self._candidate_groups(df):
            counts = df[col].value_counts(dropna=True)
            for value, count in counts.items():
                share = count / max(n, 1)
                if count >= self.MIN_GROUP_SIZE and share < self.REPRESENTATION_THRESHOLD:
                    issues.append(Issue(
                        table=table, row_index=-1, column=col,
                        issue_type="bias", severity="MEDIUM", detector=self.name,
                        original_value={"group": str(value), "count": int(count), "share": round(share, 6)},
                        expected_value=f"Group representation >= {self.REPRESENTATION_THRESHOLD:.0%}",
                        confidence=min(0.99, 0.60 + (self.REPRESENTATION_THRESHOLD - share)),
                        metadata={
                            "dimension": "representativeness",
                            "rule": "underrepresented_group",
                            "group": str(value),
                            "group_count": int(count),
                            "group_share": round(share, 6),
                        },
                    ))
        return issues

    def _fairness(self, table: str, df: pd.DataFrame) -> list[Issue]:
        issues = []
        for group_col in self._candidate_groups(df):
            groups = df[group_col].dropna()
            if groups.empty:
                continue
            missing_rates = {}
            for value, subset in df.loc[df[group_col].notna()].groupby(group_col, dropna=False):
                eligible = subset.size
                if eligible < self.MIN_GROUP_SIZE:
                    continue
                missing = float(subset.isna().sum().sum()) / max(eligible * max(len(df.columns), 1), 1)
                missing_rates[str(value)] = missing
            if len(missing_rates) < 2:
                continue
            gap = max(missing_rates.values()) - min(missing_rates.values())
            if gap >= self.FAIRNESS_GAP_THRESHOLD:
                issues.append(Issue(
                    table=table, row_index=-1, column=group_col,
                    issue_type="fairness", severity="MEDIUM", detector=self.name,
                    original_value=missing_rates,
                    expected_value=f"Missingness-rate gap < {self.FAIRNESS_GAP_THRESHOLD:.0%}",
                    confidence=min(0.99, 0.70 + min(gap, 0.29)),
                    metadata={
                        "dimension": "data_fairness",
                        "rule": "group_missingness_gap",
                        "missingness_gap": round(gap, 6),
                        "group_missingness": missing_rates,
                    },
                ))
        return issues

    def _interoperability(self, table: str, df: pd.DataFrame) -> list[Issue]:
        issues = []
        normalized = {}
        for col in df.columns:
            key = self._semantic_key(str(col))
            if key:
                normalized.setdefault(key, []).append(col)

        for concept, columns in normalized.items():
            if len(columns) > 1:
                issues.append(Issue(
                    table=table, row_index=-1, column=None,
                    issue_type="interoperability", severity="LOW", detector=self.name,
                    original_value=columns,
                    expected_value=f"Consistent representation of concept '{concept}'",
                    confidence=0.85,
                    metadata={
                        "dimension": "interoperability",
                        "rule": "semantic_alias_collision",
                        "canonical_concept": concept,
                        "columns": columns,
                    },
                ))

        for unit_col in [c for c in df.columns if any(p in str(c).lower() for p in self.UNIT_PATTERNS)]:
            values = df[unit_col].dropna().astype(str).str.strip().str.casefold()
            unique = sorted(set(values))
            if len(unique) > 1:
                issues.append(Issue(
                    table=table, row_index=-1, column=unit_col,
                    issue_type="interoperability", severity="MEDIUM", detector=self.name,
                    original_value=unique[:50],
                    expected_value="Normalized, explicit unit representation",
                    confidence=0.90,
                    metadata={
                        "dimension": "interoperability",
                        "rule": "mixed_units",
                        "unique_units": unique[:50],
                    },
                ))
        return issues

    @staticmethod
    def _numeric_metric(a: pd.Series, b: pd.Series) -> float:
        a = pd.to_numeric(a, errors="coerce").dropna()
        b = pd.to_numeric(b, errors="coerce").dropna()
        if len(a) < 20 or len(b) < 20:
            return 0.0
        # Scale-free difference using standardized Wasserstein-like quantiles.
        qa = np.quantile(a, np.linspace(0.05, 0.95, 19))
        qb = np.quantile(b, np.linspace(0.05, 0.95, 19))
        scale = max(float(np.std(pd.concat([a, b]))), 1e-9)
        return float(np.mean(np.abs(qa - qb)) / scale)

    @staticmethod
    def _categorical_metric(a: pd.Series, b: pd.Series) -> float:
        a_counts = a.dropna().astype(str).value_counts(normalize=True)
        b_counts = b.dropna().astype(str).value_counts(normalize=True)
        keys = set(a_counts.index) | set(b_counts.index)
        if not keys:
            return 0.0
        # Total variation distance is bounded [0, 1].
        return float(0.5 * sum(abs(a_counts.get(k, 0.0) - b_counts.get(k, 0.0)) for k in keys))

    def _split_metrics(self, df: pd.DataFrame) -> tuple[dict, str | None]:
        time_col = None
        for col in df.columns:
            name = str(col).lower()
            if any(token in name for token in self.TIME_PATTERNS):
                parsed = pd.to_datetime(df[col], errors="coerce", format="mixed")
                if parsed.notna().sum() >= max(20, len(df) // 10):
                    time_col = col
                    break
        work = df
        if time_col:
            parsed = pd.to_datetime(df[time_col], errors="coerce", format="mixed")
            work = df.assign(_trippinni_time=parsed).dropna(subset=["_trippinni_time"]).sort_values("_trippinni_time")
        midpoint = len(work) // 2
        if midpoint < 20 or len(work) - midpoint < 20:
            return {}, time_col
        first, second = work.iloc[:midpoint], work.iloc[midpoint:]
        metrics = {}
        for col in df.columns:
            if col == time_col:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                metrics[col] = self._numeric_metric(first[col], second[col])
            else:
                metrics[col] = self._categorical_metric(first[col], second[col])
        return metrics, time_col

    def _drift(self, table: str, df: pd.DataFrame) -> list[Issue]:
        metrics, time_col = self._split_metrics(df)
        if not metrics:
            return []
        max_col = max(metrics, key=metrics.get)
        score = metrics[max_col]
        if score < self.DRIFT_THRESHOLD:
            return []
        return [Issue(
            table=table, row_index=-1, column=max_col,
            issue_type="drift", severity="MEDIUM", detector=self.name,
            original_value=round(score, 6),
            expected_value=f"Distribution divergence < {self.DRIFT_THRESHOLD}",
            confidence=min(0.99, 0.65 + min(score, 0.34)),
            metadata={
                "dimension": "data_drift",
                "rule": "split_distribution_divergence",
                "metric": "quantile_standardized_distance" if pd.api.types.is_numeric_dtype(df[max_col]) else "total_variation_distance",
                "divergence": round(score, 6),
                "split_column": time_col,
                "split": "chronological" if time_col else "row_order",
            },
        )]

    def _robustness(self, table: str, df: pd.DataFrame) -> list[Issue]:
        rng = np.random.default_rng(42)
        rates = []
        n = len(df)
        if n < 40:
            return []
        for _ in range(8):
            idx = rng.integers(0, n, size=n)
            sample = df.iloc[idx]
            rates.append(float(sample.isna().mean().mean()))
        spread = max(rates) - min(rates)
        if spread < self.ROBUSTNESS_RANGE_THRESHOLD:
            return []
        return [Issue(
            table=table, row_index=-1, column=None,
            issue_type="robustness", severity="MEDIUM", detector=self.name,
            original_value={"bootstrap_min": min(rates), "bootstrap_max": max(rates)},
            expected_value=f"Bootstrap missingness range < {self.ROBUSTNESS_RANGE_THRESHOLD:.0%}",
            confidence=0.80,
            metadata={
                "dimension": "robustness",
                "rule": "bootstrap_missingness_sensitivity",
                "bootstrap_runs": len(rates),
                "range": round(spread, 6),
                "mean": round(float(np.mean(rates)), 6),
                "std": round(float(np.std(rates)), 6),
            },
        )]

    def _representation_stats(self, df: pd.DataFrame) -> dict:
        output = {}
        for col in self._candidate_groups(df):
            output[col] = {
                str(k): round(float(v), 6)
                for k, v in df[col].value_counts(normalize=True, dropna=True).items()
            }
        return output

    def _fairness_stats(self, df: pd.DataFrame) -> dict:
        output = {}
        for col in self._candidate_groups(df):
            groups = {}
            for value, subset in df.groupby(col, dropna=True):
                if len(subset) >= self.MIN_GROUP_SIZE:
                    groups[str(value)] = round(float(subset.isna().mean().mean()), 6)
            if len(groups) >= 2:
                output[col] = {
                    "group_missingness": groups,
                    "max_gap": round(max(groups.values()) - min(groups.values()), 6),
                }
        return output

    def _interoperability_stats(self, df: pd.DataFrame) -> dict:
        aliases = {}
        for col in df.columns:
            key = self._semantic_key(str(col))
            if key:
                aliases.setdefault(key, []).append(col)
        return {
            "semantic_aliases": {k: v for k, v in aliases.items() if len(v) > 1},
            "unit_columns": [c for c in df.columns if any(p in str(c).lower() for p in self.UNIT_PATTERNS)],
        }

    def _drift_stats(self, df: pd.DataFrame) -> dict:
        metrics, time_col = self._split_metrics(df)
        if not metrics:
            return {"available": False}
        top = sorted(metrics.items(), key=lambda item: item[1], reverse=True)[:10]
        return {"available": True, "split_column": time_col, "top_divergences": dict(top)}

    def _robustness_stats(self, df: pd.DataFrame) -> dict:
        n = len(df)
        if n < 40:
            return {"available": False}
        rng = np.random.default_rng(42)
        rates = [float(df.iloc[rng.integers(0, n, size=n)].isna().mean().mean()) for _ in range(8)]
        return {
            "available": True,
            "bootstrap_runs": len(rates),
            "missingness_mean": round(float(np.mean(rates)), 6),
            "missingness_std": round(float(np.std(rates)), 6),
            "missingness_range": round(float(max(rates) - min(rates)), 6),
        }

    @classmethod
    def _semantic_key(cls, column: str) -> str | None:
        normalized = re.sub(r"[^a-z0-9]+", "_", column.casefold()).strip("_")
        for concept, aliases in cls.SEMANTIC_ALIASES.items():
            if normalized in aliases:
                return concept
        return None
