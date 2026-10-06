"""Duplicate detection for exact rows, candidate keys, and fuzzy text matches."""

from __future__ import annotations

from typing import Any

import pandas as pd
from rapidfuzz import fuzz, process

import config
from detectors.base_detector import BaseDetector
from models.detector_result import DetectorResult
from models.issue import Issue


class DuplicateDetector(BaseDetector):
    """Detect exact duplicates, key violations, and probable fuzzy duplicates."""

    def __init__(self) -> None:
        super().__init__("DuplicateDetector")

    @staticmethod
    def _text_columns(df: pd.DataFrame, profile: Any) -> list[str]:
        """Return safe columns for fuzzy matching.

        Identifier columns are excluded because small edit distances in IDs
        are not evidence that two clinical records represent the same entity.
        """
        columns: dict[str, Any] = {}
        if isinstance(profile, dict):
            columns = profile.get("columns", {}) or {}

        selected: list[str] = []
        for column in df.columns:
            series = df[column]
            if not (
                pd.api.types.is_object_dtype(series)
                or pd.api.types.is_string_dtype(series)
            ):
                continue

            semantic_type = str(
                columns.get(column, {}).get("semantic_type", "")
            ).lower()
            normalized_name = str(column).lower()

            if semantic_type in {"identifier", "id", "key"}:
                continue
            if normalized_name.endswith("_id") or normalized_name in {
                "subject_id",
                "hadm_id",
                "stay_id",
                "event_id",
                "row_id",
            }:
                continue

            selected.append(column)

        return selected

    @staticmethod
    def _normalise_text(value: Any) -> str:
        """Normalize text before approximate comparison."""
        if pd.isna(value):
            return ""
        return " ".join(str(value).strip().casefold().split())

    def _fuzzy_matches(
        self,
        df: pd.DataFrame,
        columns: list[str],
    ) -> list[tuple[str, int, int, float, str, str]]:
        """Find high-confidence approximate matches within each text column.

        RapidFuzz's indexed extraction avoids a naive all-pairs comparison.
        Returned tuples contain column, row A, row B, score, value A, value B.
        """
        matches: list[tuple[str, int, int, float, str, str]] = []
        max_values = max(1, int(config.FUZZY_DUPLICATE_MAX_VALUES_PER_COLUMN))
        max_matches = max(1, int(config.FUZZY_DUPLICATE_MAX_MATCHES_PER_VALUE))
        threshold = float(config.FUZZY_DUPLICATE_THRESHOLD)

        for column in columns:
            series = df[column]
            normalized = series.map(self._normalise_text)

            # Work on unique non-empty values. This keeps repeated exact values
            # out of the fuzzy path and bounds the matching workload.
            value_to_indices: dict[str, list[int]] = {}
            for index, value in normalized.items():
                if not value:
                    continue
                value_to_indices.setdefault(value, []).append(int(index))

            choices = list(value_to_indices)
            if len(choices) < 2:
                continue

            # Deterministic cap for prototype-scale detection.
            if len(choices) > max_values:
                choices = choices[:max_values]

            choice_set = set(choices)
            for value in choices:
                extracted = process.extract(
                    value,
                    choices,
                    scorer=fuzz.ratio,
                    score_cutoff=threshold,
                    limit=max_matches + 1,
                )
                for matched_value, score, _ in extracted:
                    if matched_value == value or matched_value not in choice_set:
                        continue

                    left, right = sorted((value, matched_value))
                    if value != left:
                        continue

                    for row_a in value_to_indices[left]:
                        for row_b in value_to_indices[right]:
                            matches.append(
                                (
                                    column,
                                    row_a,
                                    row_b,
                                    float(score),
                                    left,
                                    right,
                                )
                            )

        return matches

    def detect(self, dataset: Any, profile: Any) -> DetectorResult:
        result = DetectorResult(detector_name=self.name)

        for table_name, df in dataset.items():
            if not isinstance(df, pd.DataFrame):
                continue

            exact = df[df.duplicated(keep=False)]
            for row_index in exact.index:
                result.add_issue(Issue(
                    table=table_name,
                    row_index=int(row_index),
                    column=None,
                    issue_type="duplicate",
                    severity="HIGH",
                    detector=self.name,
                    expected_value="Unique Record",
                    confidence=1.0,
                    metadata={"method": "exact_match"},
                ))

            primary_keys = []
            if isinstance(profile, dict):
                primary_keys = profile.get("keys", {}).get("primary_keys", [])
                if not primary_keys:
                    primary_keys = profile.get("primary_keys", [])

            key_duplicates = 0
            for column in primary_keys:
                if column not in df.columns:
                    continue
                duplicated = df[column].notna() & df[column].duplicated(keep=False)
                for row_index in df.index[duplicated]:
                    result.add_issue(Issue(
                        table=table_name,
                        row_index=int(row_index),
                        column=column,
                        issue_type="duplicate",
                        severity="HIGH",
                        detector=self.name,
                        original_value=df.at[row_index, column],
                        expected_value="Unique candidate key",
                        confidence=1.0,
                        metadata={"method": "candidate_key"},
                    ))
                    key_duplicates += 1

            fuzzy_count = 0
            fuzzy_columns = self._text_columns(df, profile) if config.FUZZY_DUPLICATE_ENABLED else []
            if config.FUZZY_DUPLICATE_ENABLED:
                for column, row_a, row_b, score, value_a, value_b in self._fuzzy_matches(
                    df, fuzzy_columns
                ):
                    # The two rows are the same approximate finding. Emit an
                    # issue for each row so existing row-level reporting works.
                    metadata = {
                        "method": "fuzzy_match",
                        "matched_row_index": row_b,
                        "matched_value": value_b,
                        "similarity_score": round(score, 2),
                        "threshold": float(config.FUZZY_DUPLICATE_THRESHOLD),
                    }
                    result.add_issue(Issue(
                        table=table_name,
                        row_index=row_a,
                        column=column,
                        issue_type="duplicate",
                        severity="MEDIUM",
                        detector=self.name,
                        original_value=value_a,
                        expected_value=value_b,
                        confidence=round(score / 100.0, 4),
                        metadata=metadata,
                    ))

                    reverse_metadata = {
                        "method": "fuzzy_match",
                        "matched_row_index": row_a,
                        "matched_value": value_a,
                        "similarity_score": round(score, 2),
                        "threshold": float(config.FUZZY_DUPLICATE_THRESHOLD),
                    }
                    result.add_issue(Issue(
                        table=table_name,
                        row_index=row_b,
                        column=column,
                        issue_type="duplicate",
                        severity="MEDIUM",
                        detector=self.name,
                        original_value=value_b,
                        expected_value=value_a,
                        confidence=round(score / 100.0, 4),
                        metadata=reverse_metadata,
                    ))
                    fuzzy_count += 2

            result.statistics[table_name] = {
                "exact_duplicates": int(len(exact)),
                "candidate_key_duplicates": int(key_duplicates),
                "fuzzy_duplicates": int(fuzzy_count),
                "semantic_duplicates": 0,
                "fuzzy_columns_checked": fuzzy_columns,
                "fuzzy_threshold": float(config.FUZZY_DUPLICATE_THRESHOLD),
            }

        return result
