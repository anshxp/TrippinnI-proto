"""Robust statistical and multivariate outlier detection for numeric clinical variables.

Detection is flag-only. It never removes or modifies source observations.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from detectors.base_detector import BaseDetector
from models.detector_result import DetectorResult
from models.issue import Issue


class OutlierDetector(BaseDetector):
    """Run IQR, Modified Z-score (MAD), Isolation Forest, and LOF."""

    RANDOM_STATE = 42
    MAX_FEATURES = 20
    MAX_ISSUES_PER_METHOD = 100
    MIN_ROWS_STATISTICAL = 20
    MIN_ROWS_ISOLATION_FOREST = 50
    MIN_ROWS_LOF = 20
    IQR_MULTIPLIER = 1.5
    MODIFIED_Z_THRESHOLD = 3.5
    CONTINUOUS_MIN_UNIQUE = 3

    def __init__(self) -> None:
        super().__init__("OutlierDetector")

    def detect(self, dataset: Any, profile: Any) -> DetectorResult:
        result = DetectorResult(detector_name=self.name)
        for table_name, df in dataset.items():
            if not isinstance(df, pd.DataFrame):
                continue
            table_profile = profile if isinstance(profile, dict) else {}
            numeric, exclusions = self._numeric_frame(df, table_profile)
            table_stats = self._empty_stats(numeric, exclusions)

            if numeric.shape[1] == 0 or len(numeric) < self.MIN_ROWS_STATISTICAL:
                table_stats["status"] = "not_applicable"
                table_stats["reason"] = (
                    "No eligible continuous numeric features"
                    if numeric.shape[1] == 0
                    else f"Requires at least {self.MIN_ROWS_STATISTICAL} sample rows"
                )
                result.statistics[table_name] = table_stats
                continue

            method_results = [
                self._iqr(table_name, df, numeric),
                self._mad(table_name, df, numeric),
                self._isolation_forest(table_name, df, numeric),
                self._lof(table_name, df, numeric),
            ]
            for method_result in method_results:
                result.issues.extend(method_result.issues)

            method_counts = {
                "iqr": method_results[0].issue_count,
                "modified_z_mad": method_results[1].issue_count,
                "isolation_forest": method_results[2].issue_count,
                "lof": method_results[3].issue_count,
            }
            # Count row-level detector agreement from flags, not issue totals:
            # univariate detectors can emit multiple column issues per row.
            flags = self._method_flags(numeric)
            votes = flags.sum(axis=1)
            table_stats.update({
                "status": "completed",
                "method_issue_counts": method_counts,
                "flagged_rows_by_method": {
                    method: int(flags[method].sum()) for method in flags.columns
                },
                "rows_flagged_by_at_least_two_methods": int((votes >= 2).sum()),
                "rows_flagged_by_at_least_three_methods": int((votes >= 3).sum()),
                "rows_flagged_by_all_methods": int((votes == len(flags.columns)).sum()),
                "sample_rows": int(len(numeric)),
                "numeric_features": int(numeric.shape[1]),
            })
            result.statistics[table_name] = table_stats
        return result

    def _empty_stats(self, numeric: pd.DataFrame, exclusions: dict) -> dict:
        return {
            "methods": ["iqr_1.5", "modified_z_mad_3.5", "isolation_forest", "lof"],
            "status": "pending",
            "eligible_features": list(numeric.columns),
            "excluded_features": exclusions,
            "numeric_features": int(numeric.shape[1]),
            "method_issue_counts": {},
            "flagged_rows_by_method": {},
            "rows_flagged_by_at_least_two_methods": 0,
            "rows_flagged_by_at_least_three_methods": 0,
            "rows_flagged_by_all_methods": 0,
        }

    def _numeric_frame(self, df: pd.DataFrame, profile: dict) -> tuple[pd.DataFrame, dict]:
        columns = profile.get("columns", {}) if isinstance(profile, dict) else {}
        selected, exclusions = [], {}
        for column in df.columns:
            meta = columns.get(column, {})
            semantic = meta.get("semantic_type")
            name = str(column).lower()
            if semantic != "numeric":
                exclusions[column] = "not_profiled_as_numeric"
                continue
            if name.endswith("_id") or name in {"itemid", "seq_num"}:
                exclusions[column] = "identifier_like_column"
                continue
            values = pd.to_numeric(df[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
            unique_count = int(values.nunique(dropna=True))
            if unique_count < self.CONTINUOUS_MIN_UNIQUE:
                exclusions[column] = "binary_or_low_cardinality"
                continue
            if int(values.notna().sum()) < self.MIN_ROWS_STATISTICAL:
                exclusions[column] = "insufficient_non_missing_values"
                continue
            if values.dropna().empty or values.dropna().std() == 0:
                exclusions[column] = "constant_column"
                continue
            selected.append(column)

        numeric = (
            df[selected].apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .dropna(how="all")
            .iloc[:, : self.MAX_FEATURES]
        )
        return numeric, exclusions

    def _make_issue(self, table, idx, column, method, value, confidence=0.8, score=None):
        metadata = {"method": method}
        if score is not None and np.isfinite(score):
            metadata["anomaly_score"] = float(score)
        return Issue(
            table=table, row_index=int(idx), column=column, issue_type="outlier",
            severity="MEDIUM", detector=self.name, original_value=value,
            expected_value="Investigate against clinical/contextual validity",
            confidence=float(confidence), metadata=metadata,
        )

    @staticmethod
    def _series_flags(series: pd.Series, mask: pd.Series, method: str) -> pd.Series:
        return pd.Series(mask.fillna(False).to_numpy(), index=series.index, name=method)

    def _iqr(self, table, df, numeric):
        result = DetectorResult("IQR")
        for column in numeric.columns:
            series = numeric[column]
            q1, q3 = series.quantile([0.25, 0.75])
            spread = q3 - q1
            if not np.isfinite(spread) or spread <= 0:
                continue
            lower = q1 - self.IQR_MULTIPLIER * spread
            upper = q3 + self.IQR_MULTIPLIER * spread
            mask = ((series < lower) | (series > upper)).fillna(False)
            for idx in series.index[mask][: self.MAX_ISSUES_PER_METHOD]:
                result.add_issue(self._make_issue(
                    table, idx, column, "iqr_1.5", df.at[idx, column], 0.85
                ))
        return result

    def _mad(self, table, df, numeric):
        result = DetectorResult("ModifiedZScoreMAD")
        for column in numeric.columns:
            series = numeric[column]
            median = series.median()
            mad = float(np.median(np.abs(series.dropna() - median)))
            if not np.isfinite(mad) or mad <= 0:
                continue
            modified_z = 0.6745 * (series - median) / mad
            mask = (modified_z.abs() > self.MODIFIED_Z_THRESHOLD).fillna(False)
            for idx in series.index[mask][: self.MAX_ISSUES_PER_METHOD]:
                result.add_issue(self._make_issue(
                    table, idx, column, "modified_z_mad_3.5", df.at[idx, column], 0.85,
                    float(modified_z.loc[idx])
                ))
        return result

    def _scaled_matrix(self, numeric):
        filled = numeric.copy()
        for column in filled.columns:
            filled[column] = filled[column].fillna(filled[column].median())
        return StandardScaler().fit_transform(filled)

    def _isolation_forest(self, table, df, numeric):
        result = DetectorResult("IsolationForest")
        if len(numeric) < self.MIN_ROWS_ISOLATION_FOREST:
            result.statistics["skipped_reason"] = (
                f"Requires at least {self.MIN_ROWS_ISOLATION_FOREST} rows"
            )
            return result
        try:
            matrix = self._scaled_matrix(numeric)
            model = IsolationForest(
                n_estimators=200, contamination="auto",
                random_state=self.RANDOM_STATE, n_jobs=-1,
            )
            labels = model.fit_predict(matrix)
            scores = model.decision_function(matrix)
            flagged = np.where(labels == -1)[0]
            flagged = flagged[np.argsort(scores[flagged])][: self.MAX_ISSUES_PER_METHOD]
            for pos in flagged:
                idx = numeric.index[int(pos)]
                result.add_issue(self._make_issue(
                    table, idx, None, "isolation_forest", None, 0.75, float(scores[pos])
                ))
        except Exception as exc:
            result.success, result.error = False, str(exc)
        return result

    def _lof(self, table, df, numeric):
        result = DetectorResult("LOF")
        if len(numeric) < self.MIN_ROWS_LOF:
            result.statistics["skipped_reason"] = f"Requires at least {self.MIN_ROWS_LOF} rows"
            return result
        try:
            matrix = self._scaled_matrix(numeric)
            neighbors = min(20, len(numeric) - 1)
            model = LocalOutlierFactor(
                n_neighbors=neighbors, contamination="auto", novelty=False
            )
            labels = model.fit_predict(matrix)
            scores = model.negative_outlier_factor_
            flagged = np.where(labels == -1)[0]
            flagged = flagged[np.argsort(scores[flagged])][: self.MAX_ISSUES_PER_METHOD]
            for pos in flagged:
                idx = numeric.index[int(pos)]
                result.add_issue(self._make_issue(
                    table, idx, None, "lof", None, 0.75, float(scores[pos])
                ))
        except Exception as exc:
            result.success, result.error = False, str(exc)
        return result

    def _method_flags(self, numeric: pd.DataFrame) -> pd.DataFrame:
        flags = pd.DataFrame(False, index=numeric.index, columns=[
            "iqr", "modified_z_mad", "isolation_forest", "lof"
        ])
        for column in numeric.columns:
            series = numeric[column]
            q1, q3 = series.quantile([0.25, 0.75])
            spread = q3 - q1
            if np.isfinite(spread) and spread > 0:
                flags["iqr"] |= ((series < q1 - self.IQR_MULTIPLIER * spread) |
                                 (series > q3 + self.IQR_MULTIPLIER * spread)).fillna(False)
            median = series.median()
            mad = float(np.median(np.abs(series.dropna() - median)))
            if np.isfinite(mad) and mad > 0:
                z = 0.6745 * (series - median) / mad
                flags["modified_z_mad"] |= (z.abs() > self.MODIFIED_Z_THRESHOLD).fillna(False)

        if len(numeric) >= self.MIN_ROWS_ISOLATION_FOREST:
            matrix = self._scaled_matrix(numeric)
            model = IsolationForest(n_estimators=200, contamination="auto",
                                   random_state=self.RANDOM_STATE, n_jobs=-1)
            flags["isolation_forest"] = model.fit_predict(matrix) == -1
        if len(numeric) >= self.MIN_ROWS_LOF:
            matrix = self._scaled_matrix(numeric)
            model = LocalOutlierFactor(n_neighbors=min(20, len(numeric)-1), contamination="auto")
            flags["lof"] = model.fit_predict(matrix) == -1
        return flags
