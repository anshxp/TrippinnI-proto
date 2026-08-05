"""
orchestrator.py

Coordinates all TrippinnI modules.
"""

from core.loader import LoaderManager
from profiling.profiler import DatasetProfilerEngine


class Orchestrator:

    def __init__(self):

        self.loader_manager = LoaderManager()

        self.profiler = DatasetProfilerEngine()

        self.profiles = {}

    ##################################################################

    def initialize(
        self,
        dataset_type,
        dataset_path
    ):

        # Initialize loader
        self.loader_manager.initialize(
            dataset_type,
            dataset_path
        )

        print("Dataset initialized.")

        self._profile_tables()

    ##################################################################

    def _profile_tables(self):

        self.profiles = {}

        loader = self.loader_manager.get_loader()

        for table in loader.get_tables():
            # MIMIC exposes chunked CSV reads. Other loaders deliberately
            # retain their established DataFrame-based, lazy-loading path.
            if hasattr(loader, "get_dataframe_chunks"):
                self.profiles[table] = self.profiler.profile_chunks(
                    table,
                    loader.get_dataframe_chunks(table),
                )
            else:
                dataframe = loader.get_dataframe(table)
                self.profiles[table] = self.profiler.profile(
                    table,
                    dataframe
                )

        print("Dataset profiling completed.")

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
