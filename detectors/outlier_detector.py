"""Statistical outlier detection for numeric clinical variables."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from detectors.base_detector import BaseDetector
from models.detector_result import DetectorResult
from models.issue import Issue


class OutlierDetector(BaseDetector):
    """Combine robust rules, Isolation Forest, COPOD, and an autoencoder."""

    RANDOM_STATE = 42
    MAX_FEATURES = 20
    MAX_ISSUES_PER_METHOD = 100

    def __init__(self) -> None:
        super().__init__("OutlierDetector")

    def detect(self, dataset: Any, profile: Any) -> DetectorResult:
        result = DetectorResult(detector_name=self.name)
        for table_name, df in dataset.items():
            if not isinstance(df, pd.DataFrame):
                continue
            table_profile = profile if isinstance(profile, dict) else {}
            numeric = self._numeric_frame(df, table_profile)
            if numeric.shape[1] == 0 or len(numeric) < 20:
                result.statistics[table_name] = {
                    "rule_based": 0, "isolation_forest": 0,
                    "copod": 0, "autoencoder": 0, "numeric_features": int(numeric.shape[1]),
                }
                continue

            rule = self._rule_based(table_name, df, numeric)
            isolation = self._isolation_forest(table_name, df, numeric)
            copod = self._copod(table_name, df, numeric)
            autoencoder = self._autoencoder(table_name, df, numeric)
            for detector_result in (rule, isolation, copod, autoencoder):
                result.issues.extend(detector_result.issues)

            result.statistics[table_name] = {
                "rule_based": rule.issue_count,
                "isolation_forest": isolation.issue_count,
                "copod": copod.issue_count,
                "autoencoder": autoencoder.issue_count,
                "numeric_features": int(numeric.shape[1]),
            }
        return result

    def _numeric_frame(self, df: pd.DataFrame, profile: dict) -> pd.DataFrame:
        columns = profile.get("columns", {}) if isinstance(profile, dict) else {}
        selected = []
        for column in df.columns:
            meta = columns.get(column, {})
            semantic = meta.get("semantic_type")
            name = column.lower()
            if semantic != "numeric":
                continue
            if name.endswith("_id") or name in {"itemid", "seq_num"}:
                continue
            values = pd.to_numeric(df[column], errors="coerce")
            if values.notna().sum() < 20 or values.nunique(dropna=True) <= 1:
                continue
            selected.append(column)
        return df[selected].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna(how="all").iloc[:, :self.MAX_FEATURES]

    def _make_issue(self, table, idx, column, method, value, confidence=0.8):
        return Issue(
            table=table, row_index=int(idx), column=column, issue_type="outlier",
            severity="MEDIUM", detector=self.name, original_value=value,
            expected_value="Typical distribution for the observed variable",
            confidence=float(confidence), metadata={"method": method},
        )

    def _rule_based(self, table, df, numeric):
        result = DetectorResult("RuleBasedOutlier")
        count = 0
        for column in numeric.columns:
            series = numeric[column].dropna()
            q1, q3 = series.quantile([0.25, 0.75])
            iqr = q3 - q1
            if not np.isfinite(iqr) or iqr <= 0:
                continue
            lower, upper = q1 - 3.0 * iqr, q3 + 3.0 * iqr
            mask = (numeric[column] < lower) | (numeric[column] > upper)
            for idx in numeric.index[mask.fillna(False)][:self.MAX_ISSUES_PER_METHOD]:
                result.add_issue(self._make_issue(table, idx, column, "iqr_3x", df.at[idx, column], 0.85))
                count += 1
        result.statistics["issues"] = count
        return result

    def _scaled_matrix(self, numeric):
        filled = numeric.copy()
        for column in filled.columns:
            filled[column] = filled[column].fillna(filled[column].median())
        return StandardScaler().fit_transform(filled)

    def _isolation_forest(self, table, df, numeric):
        result = DetectorResult("IsolationForest")
        if len(numeric) < 50:
            return result
        try:
            model = IsolationForest(
                n_estimators=100, contamination="auto", random_state=self.RANDOM_STATE, n_jobs=-1
            )
            labels = model.fit_predict(self._scaled_matrix(numeric))
            scores = model.decision_function(self._scaled_matrix(numeric))
            order = np.argsort(scores)
            flagged = [i for i in order if labels[i] == -1][:self.MAX_ISSUES_PER_METHOD]
            for pos in flagged:
                idx = numeric.index[pos]
                result.add_issue(self._make_issue(table, idx, None, "isolation_forest", None, 0.75))
        except Exception as exc:
            result.success = False
            result.error = str(exc)
        return result

    def _copod(self, table, df, numeric):
        result = DetectorResult("COPOD")
        try:
            from pyod.models.copod import COPOD
            model = COPOD(contamination=0.01)
            matrix = self._scaled_matrix(numeric)
            model.fit(matrix)
            labels = np.asarray(model.labels_)
            scores = np.asarray(model.decision_scores_)
            flagged = np.where(labels == 1)[0]
            flagged = flagged[np.argsort(scores[flagged])[::-1]][:self.MAX_ISSUES_PER_METHOD]
            for pos in flagged:
                idx = numeric.index[int(pos)]
                result.add_issue(self._make_issue(table, idx, None, "copod", None, 0.75))
        except Exception as exc:
            result.success = False
            result.error = str(exc)
        return result

    def _autoencoder(self, table, df, numeric):
        result = DetectorResult("Autoencoder")
        if len(numeric) < 100 or numeric.shape[1] < 2:
            return result
        try:
            matrix = self._scaled_matrix(numeric)
            hidden = max(2, min(16, numeric.shape[1] // 2))
            model = MLPRegressor(
                hidden_layer_sizes=(hidden,), activation="relu", solver="adam",
                max_iter=40, random_state=self.RANDOM_STATE, early_stopping=True,
            )
            model.fit(matrix, matrix)
            reconstructed = model.predict(matrix)
            errors = np.mean((matrix - reconstructed) ** 2, axis=1)
            threshold = float(np.quantile(errors, 0.995))
            flagged = np.where(errors > threshold)[0]
            for pos in flagged[:self.MAX_ISSUES_PER_METHOD]:
                idx = numeric.index[int(pos)]
                result.add_issue(self._make_issue(table, idx, None, "mlp_autoencoder", None, 0.70))
        except Exception as exc:
            result.success = False
            result.error = str(exc)
        return result