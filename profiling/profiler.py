from __future__ import annotations

import random

import pandas as pd

from profiling.dataset_profiler import DatasetProfiler
from profiling.column_profiler import ColumnProfiler
from profiling.memory_profiler import MemoryProfiler
from profiling.key_detector import KeyDetector
from profiling.report_builder import ReportBuilder
from profiling.report_writer import ReportWriter
from detectors.outlier_detector import OutlierDetector


class DatasetProfilerEngine:
    """Coordinates profiling and bounded-sample outlier screening."""

    OUTLIER_SAMPLE_CAP = 5000
    OUTLIER_SAMPLE_SEED = 42

    def __init__(self):
        self.dataset = DatasetProfiler()
        self.columns = ColumnProfiler()
        self.memory = MemoryProfiler()
        self.keys = KeyDetector()
        self.builder = ReportBuilder()
        self.writer = ReportWriter()
        self.outliers = OutlierDetector()

    def profile(self, table_name, dataframe):
        dataset_profile = self.dataset.profile(table_name, dataframe)
        column_profiles = self.columns.profile(dataframe)
        memory_profile = self.memory.profile(dataframe)
        key_profile = self.keys.detect(dataframe)

        report = self.builder.build(
            dataset_profile, column_profiles, memory_profile, key_profile
        )
        detection_sample = self._bounded_sample(dataframe)
        self._attach_outlier_summary(table_name, report, detection_sample)
        self.writer.save(table_name, report)
        return report

    def profile_chunks(
        self,
        table_name,
        chunks,
        sample_size: int | None = None,
        sample_seed: int = 42,
    ):
        """Profile a chunk iterator in one pass and reservoir-sample rows.

        Outlier screening is run on the bounded reservoir, not the complete
        streamed table. It is a screening summary, not a full-table count.
        Returns (profile_report, sample_dataframe_or_None).
        """
        dataset_state = self.dataset.start_streaming(table_name)
        column_state = self.columns.start_streaming()
        memory_state = self.memory.start_streaming()
        key_state = self.keys.start_streaming()

        reservoir: list = []
        rows_seen = 0
        rng = random.Random(sample_seed)

        for chunk in chunks:
            self.dataset.update_streaming(dataset_state, chunk)
            self.columns.update_streaming(column_state, chunk)
            self.memory.update_streaming(memory_state, chunk)
            self.keys.update_streaming(key_state, chunk)

            if sample_size is not None:
                rows_seen = self._update_reservoir(
                    reservoir, rows_seen, chunk, sample_size, rng
                )

        report = self.builder.build(
            self.dataset.finalize_streaming(dataset_state),
            self.columns.finalize_streaming(column_state),
            self.memory.finalize_streaming(memory_state),
            self.keys.finalize_streaming(key_state),
        )

        sample_dataframe = (
            pd.DataFrame(reservoir).reset_index(drop=True)
            if reservoir else None
        )
        if sample_dataframe is not None:
            sample_dataframe = self._bounded_sample(sample_dataframe)
        self._attach_outlier_summary(table_name, report, sample_dataframe)
        self.writer.save(table_name, report)
        return report, sample_dataframe

    def _bounded_sample(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        if len(dataframe) <= self.OUTLIER_SAMPLE_CAP:
            return dataframe
        return dataframe.sample(
            n=self.OUTLIER_SAMPLE_CAP,
            random_state=self.OUTLIER_SAMPLE_SEED,
        )

    def _attach_outlier_summary(
        self,
        table_name: str,
        profile_report: dict,
        sample: pd.DataFrame | None,
    ) -> None:
        """Embed flag-only detector summaries into the profiling JSON."""
        if sample is None or sample.empty:
            profile_report["outlier_detection"] = {
                "status": "not_applicable",
                "reason": "No detection sample available",
                "sample_rows": 0,
                "methods": ["IQR 1.5x", "Modified Z-score (MAD)", "Isolation Forest", "LOF"],
            }
            return

        try:
            detection = self.outliers.detect({table_name: sample}, profile_report)
            stats = detection.statistics.get(table_name, {})
            profile_report["outlier_detection"] = {
                "status": stats.get("status", "completed"),
                "sample_rows": int(len(sample)),
                "sample_scope": "bounded profile sample; not full-table outlier counts",
                "configured_methods": stats.get("methods", [
                    "iqr_1.5", "modified_z_mad_3.5", "isolation_forest", "lof"
                ]),
                "active_methods": stats.get("active_methods", []),
                "eligible_features": stats.get("eligible_features", []),
                "excluded_features": stats.get("excluded_features", {}),
                "numeric_features": int(stats.get("numeric_features", 0)),
                "flagged_rows_by_method": stats.get("flagged_rows_by_method", {}),
                "method_issue_counts": stats.get("method_issue_counts", {}),
                "rows_flagged_by_at_least_two_methods": int(
                    stats.get("rows_flagged_by_at_least_two_methods", 0)
                ),
                "rows_flagged_by_at_least_three_methods": int(
                    stats.get("rows_flagged_by_at_least_three_methods", 0)
                ),
                "rows_flagged_by_all_methods": int(
                    stats.get("rows_flagged_by_all_methods", 0)
                ),
                "reason": stats.get("reason"),
            }
        except Exception as exc:
            profile_report["outlier_detection"] = {
                "status": "failed",
                "sample_rows": int(len(sample)),
                "error": str(exc),
            }

    @staticmethod
    def _update_reservoir(reservoir, rows_seen, chunk, size, rng):
        """Approximate chunk-vectorized reservoir sampling (bounded memory)."""
        if len(reservoir) < size:
            needed = size - len(reservoir)
            take = min(needed, len(chunk))
            reservoir.extend(row for _, row in chunk.iloc[:take].iterrows())
            rows_seen += take
            chunk = chunk.iloc[take:]

        remaining = len(chunk)
        if remaining == 0:
            return rows_seen

        inclusion_prob = size / max(rows_seen + remaining, 1)
        num_replacements = min(
            remaining, size, round(inclusion_prob * remaining)
        )
        if num_replacements > 0:
            selected = chunk.sample(
                n=num_replacements,
                random_state=rng.randint(0, 2**31 - 1),
            )
            slots = rng.sample(range(size), num_replacements)
            for slot, (_, row) in zip(slots, selected.iterrows()):
                reservoir[slot] = row

        rows_seen += remaining
        return rows_seen
