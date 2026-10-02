"""
orchestrator.py

Coordinates all TrippinnI modules.
"""

import gc
import json
import time
from pathlib import Path

import psutil

import config
from core.loader import LoaderManager
from core.memory_utils import downcast_dataframe, release
from profiling.profiler import DatasetProfilerEngine
from quality.detector import QualityDetector
from quality.confidence import ConfidenceAggregator


class Orchestrator:

    def __init__(self):

        self.loader_manager = LoaderManager()

        self.profiler = DatasetProfilerEngine()

        self.quality_detector = QualityDetector()
        self.confidence = ConfidenceAggregator()

        self.profiles = {}
        self.quality_results = {}

        # Persistent table-level checkpoint. A table is marked complete only
        # after profiling AND quality detection finish successfully.
        self.checkpoint_path = config.OUTPUT_ROOT / "reports" / "profiling" / ".checkpoint.json"
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        # For visibility while running on constrained hardware - see
        # _log_memory below. Not used for any decision-making, purely
        # so you can watch RSS stay bounded across a real 10GB run
        # instead of taking it on faith.
        self._process = psutil.Process()
        self._checkpoint_version = 2

    ##################################################################

    def initialize(
        self,
        dataset_type,
        dataset_path
    ):

        self.loader_manager.initialize(
            dataset_type,
            dataset_path
        )

        print("Dataset initialized.")

        self._process_tables()

    ##################################################################

    def _process_tables(self):

        self.profiles = {}
        self.quality_results = {}

        loader = self.loader_manager.get_loader()
        tables = loader.get_tables()
        total_tables = len(tables)
        completed_tables = self._load_checkpoint()

        if completed_tables:
            print()
            print(f"RESUME: skipping {len(completed_tables):,} completed table(s)")

        print()
        print("=" * 78)
        print(f"PROFILING STARTED | {total_tables} table(s)")
        print("=" * 78)

        for table_index, table in enumerate(tables, start=1):

            if table in completed_tables:
                print(f"[{table_index}/{total_tables}] SKIP: {table} (already completed)")
                continue

            if hasattr(loader, "get_dataframe_chunks"):
                # Profiling stays fully streaming (never holds the whole
                # table). Detection gets a reservoir-sampled subset built
                # in the same pass, so MissingDetector/DuplicateDetector/
                # etc. (which expect a real in-memory DataFrame) still
                # run, without ever materializing the full table.
                source_path = self._get_source_path(loader, table)
                total_rows = self._get_mimic_row_count(table)
                file_size_mb = source_path.stat().st_size / (1024 ** 2)

                if total_rows <= 0 and hasattr(loader, "get_row_count"):
                    if file_size_mb <= config.CSV_ROW_COUNT_FALLBACK_MAX_MB:
                        total_rows = loader.get_row_count(
                            table,
                            max_file_size_mb=config.CSV_ROW_COUNT_FALLBACK_MAX_MB,
                        )

                print()
                print("-" * 78)
                print(
                    f"[{table_index}/{total_tables}] "
                    f"PROFILING: {source_path.name}"
                )
                print(f"  Source: {source_path}")
                print(f"  File size: {file_size_mb:,.1f} MB")
                if total_rows > 0:
                    print(
                        f"  Estimated data rows: {total_rows:,} "
                        f"(MIMIC-IV v3.1 reference row count)"
                    )
                else:
                    print("  Estimated data rows: unavailable")
                print(f"  Chunk size: {config.CSV_CHUNK_SIZE:,} rows")
                prefix_rows = (
                    max(1, int(total_rows * config.PROFILE_PREFIX_FRACTION))
                    if total_rows > 0
                    else 0
                )
                print(f"  Prototype prefix: first {config.PROFILE_PREFIX_FRACTION:.0%} of rows")
                if prefix_rows > 0:
                    print(f"  Prefix row limit: {prefix_rows:,} of {total_rows:,}")
                print("  Stage: first-prefix streaming + profiling + key detection")

                chunks = loader.get_dataframe_chunks(
                    table,
                    chunksize=config.CSV_CHUNK_SIZE,
                    prefix_fraction=config.PROFILE_PREFIX_FRACTION,
                    total_rows=total_rows,
                )
                progress_chunks = self._progress_chunks(
                    chunks,
                    table=table,
                    filename=source_path.name,
                    total_rows=total_rows,
                    table_index=table_index,
                    total_tables=total_tables,
                )

                profile_started = time.monotonic()
                report, sample = self.profiler.profile_chunks(
                    table,
                    progress_chunks,
                    sample_size=config.MAX_ROWS_FOR_DETECTION,
                    sample_seed=config.DETECTION_SAMPLE_SEED,
                )
                profile_elapsed = time.monotonic() - profile_started
                self.profiles[table] = report

                print(
                    f"  Stage complete: profiling | "
                    f"elapsed {self._format_duration(profile_elapsed)}"
                )

                if sample is not None:
                    print(
                        f"  Stage: quality detection | "
                        f"sample rows: {len(sample):,}"
                    )
                    sample = downcast_dataframe(sample)

                    detection_started = time.monotonic()
                    result = self.quality_detector.run(
                        {table: sample},
                        report,
                    )
                    result.issues = self.confidence.aggregate(result.issues)
                    self.quality_results[table] = result
                    detection_elapsed = time.monotonic() - detection_started

                    print(
                        f"  Stage complete: quality detection | "
                        f"elapsed {self._format_duration(detection_elapsed)} | "
                        f"issues: {result.total_issues}"
                    )

                    release(sample)

                print("  Stage: saving report + releasing table resources")
                loader.clear_cache()
                gc.collect()
                self._log_memory(table)
                self._mark_table_complete(table)
                completed_tables.add(table)
                print(f"  Checkpoint saved: {table}")
                continue

            # Fallback for any loader without chunked reading support
            # (e.g. a future JSON/FHIR loader). Full-load, then downcast
            # and subsample before detection, same as the chunked path
            # achieves via streaming.
            print()
            print("-" * 78)
            print(
                f"[{table_index}/{total_tables}] "
                f"PROFILING: {table} (non-chunked loader)"
            )
            print("  Stage: loading full table into memory")
            profile_started = time.monotonic()
            dataframe = loader.get_dataframe(table)
            dataframe = downcast_dataframe(dataframe)

            self.profiles[table] = self.profiler.profile(
                table,
                dataframe,
            )
            profile_elapsed = time.monotonic() - profile_started
            print(
                f"  Stage complete: profiling | "
                f"rows: {len(dataframe):,} | "
                f"elapsed {self._format_duration(profile_elapsed)}"
            )

            detection_frame = dataframe
            if len(dataframe) > config.MAX_ROWS_FOR_DETECTION:
                detection_frame = dataframe.sample(
                    n=config.MAX_ROWS_FOR_DETECTION,
                    random_state=config.DETECTION_SAMPLE_SEED,
                )

            print(
                f"  Stage: quality detection | "
                f"sample rows: {len(detection_frame):,}"
            )
            result = self.quality_detector.run(
                {table: detection_frame},
                self.profiles[table],
            )

            result.issues = self.confidence.aggregate(result.issues)

            self.quality_results[table] = result

            release(dataframe, detection_frame)
            loader.clear_cache()
            self._log_memory(table)
            self._mark_table_complete(table)
            completed_tables.add(table)
            print(f"  Checkpoint saved: {table}")

        print()
        print("=" * 78)
        print("Dataset profiling and quality detection completed.")
        print("=" * 78)

    ##################################################################

    def _load_checkpoint(self) -> set[str]:
        """Load table names that completed the full pipeline previously."""
        if not self.checkpoint_path.exists():
            return set()
        try:
            with self.checkpoint_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if data.get("version") != self._checkpoint_version:
                print(
                    "  Checkpoint version changed; restarting profiling "
                    "with the current prefix policy."
                )
                return set()
            return set(data.get("completed_tables", []))
        except (OSError, ValueError, TypeError):
            print("  Warning: checkpoint could not be read; starting without resume state.")
            return set()

    def _mark_table_complete(self, table: str) -> None:
        """Persist completion immediately after a table finishes successfully."""
        completed = self._load_checkpoint()
        completed.add(table)
        payload = {
            "version": self._checkpoint_version,
            "completed_tables": sorted(completed),
        }
        temporary = self.checkpoint_path.with_suffix(".tmp")
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
            temporary.replace(self.checkpoint_path)
        except OSError as exc:
            print(f"  Warning: could not save checkpoint: {exc}")

    ##################################################################

    @staticmethod
    def _get_source_path(loader, table: str) -> Path:
        """Return the actual source file for a catalog-backed table."""
        catalog = getattr(loader, "catalog", None)
        if catalog is None:
            return Path(table)
        return Path(catalog.get_table_path(table))

    ##################################################################

    @staticmethod
    def _get_mimic_row_count(table: str) -> int:
        """Return the known MIMIC-IV v3.1 row count for a table."""
        return int(config.MIMIC_V31_ROW_COUNTS.get(table.lower(), 0))

    ##################################################################

    def _progress_chunks(
        self,
        chunks,
        *,
        table: str,
        filename: str,
        total_rows: int,
        table_index: int,
        total_tables: int,
    ):
        """Yield profiling chunks while printing live progress and ETA."""
        started = time.monotonic()
        rows_seen = 0
        chunk_number = 0

        for chunk in chunks:
            chunk_number += 1
            rows_seen += len(chunk)

            elapsed = max(time.monotonic() - started, 1e-9)
            rate = rows_seen / elapsed

            if total_rows > 0:
                percent = min(rows_seen / total_rows * 100.0, 100.0)
                remaining = max(total_rows - rows_seen, 0)
                eta_seconds = remaining / rate if rate > 0 else 0
                progress = (
                    f"{percent:6.2f}% | "
                    f"{rows_seen:,}/{total_rows:,} rows | "
                    f"~{remaining:,} left | "
                    f"ETA {self._format_duration(eta_seconds)}"
                )
            else:
                progress = (
                    f"{rows_seen:,} rows | "
                    f"ETA unavailable"
                )

            print(
                f"  [{table_index}/{total_tables}] "
                f"{filename} | chunk {chunk_number} | "
                f"{progress} | "
                f"{rate:,.0f} rows/s | "
                f"elapsed {self._format_duration(elapsed)}"
            )
            yield chunk

        elapsed = time.monotonic() - started
        print(
            f"  [{table_index}/{total_tables}] {filename} | "
            f"READ COMPLETE | {rows_seen:,} rows | "
            f"{chunk_number} chunks | "
            f"elapsed {self._format_duration(elapsed)}"
        )

    ##################################################################

    @staticmethod
    def _format_duration(seconds: float) -> str:
        seconds = max(float(seconds), 0.0)
        if seconds < 60:
            return f"{seconds:.1f}s"
        minutes, remaining = divmod(int(seconds), 60)
        if minutes < 60:
            return f"{minutes}m {remaining:02d}s"
        hours, minutes = divmod(minutes, 60)
        return f"{hours}h {minutes:02d}m"

    ##################################################################

    def _log_memory(self, table: str) -> None:
        """
        Print current process RSS after a table finishes. This is the
        thing to actually watch during a real 10GB run: if this number
        climbs steadily table over table instead of staying roughly flat,
        something is holding a reference it shouldn't.
        """

        rss_mb = self._process.memory_info().rss / (1024 ** 2)
        print(f"  [{table}] done - process RSS: {rss_mb:.1f} MB")

    ##################################################################

    def get_tables(self):

        return self.loader_manager.get_tables()

    ##################################################################

    def get_dataframe(
        self,
        table
    ):

        return self.loader_manager.get_dataframe(table)

    ##################################################################

    def get_schema(self):

        return self.loader_manager.get_schema()

    ##################################################################

    def get_profiles(self):

        return self.profiles

    ##################################################################

    def get_profile(
        self,
        table
    ):

        return self.profiles.get(table)

    ##################################################################

    def get_quality_results(self):

        return self.quality_results

    ##################################################################

    def get_quality_result(
        self,
        table
    ):

        return self.quality_results.get(table)