"""
orchestrator.py

Coordinates all TrippinnI modules.
"""

import gc

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

        # For visibility while running on constrained hardware - see
        # _log_memory below. Not used for any decision-making, purely
        # so you can watch RSS stay bounded across a real 10GB run
        # instead of taking it on faith.
        self._process = psutil.Process()

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

        for table in loader.get_tables():

            if hasattr(loader, "get_dataframe_chunks"):
                # Both MimicLoader and SyntheaLoader expose this now.
                # Profiling stays fully streaming (never holds the whole
                # table). Detection gets a reservoir-sampled subset built
                # in the same pass, so MissingDetector/DuplicateDetector/
                # etc. (which expect a real in-memory DataFrame) still
                # run, without ever materializing the full table.
                chunks = loader.get_dataframe_chunks(
                    table,
                    chunksize=config.CSV_CHUNK_SIZE,
                )

                report, sample = self.profiler.profile_chunks(
                    table,
                    chunks,
                    sample_size=config.MAX_ROWS_FOR_DETECTION,
                    sample_seed=config.DETECTION_SAMPLE_SEED,
                )
                self.profiles[table] = report

                if sample is not None:
                    sample = downcast_dataframe(sample)

                    result = self.quality_detector.run(
                        {table: sample},
                        report,
                    )
                    result.issues = self.confidence.aggregate(result.issues)
                    self.quality_results[table] = result

                    release(sample)

                loader.clear_cache()
                gc.collect()
                self._log_memory(table)
                continue

            # Fallback for any loader without chunked reading support
            # (e.g. a future JSON/FHIR loader). Full-load, then downcast
            # and subsample before detection, same as the chunked path
            # achieves via streaming.
            dataframe = loader.get_dataframe(table)
            dataframe = downcast_dataframe(dataframe)

            self.profiles[table] = self.profiler.profile(
                table,
                dataframe,
            )

            detection_frame = dataframe
            if len(dataframe) > config.MAX_ROWS_FOR_DETECTION:
                detection_frame = dataframe.sample(
                    n=config.MAX_ROWS_FOR_DETECTION,
                    random_state=config.DETECTION_SAMPLE_SEED,
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

        print("Dataset profiling and quality detection completed.")

    ##################################################################

    def _log_memory(self, table: str) -> None:
        """
        Print current process RSS after a table finishes. This is the
        thing to actually watch during a real run on your 10GB dataset:
        if this number climbs steadily table over table instead of
        staying roughly flat, something is holding a reference it
        shouldn't (check for accidental caching first).
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