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

        self.loader = None

        self.profiles = {}

    ##################################################################

    def initialize(
        self,
        dataset_type,
        dataset_path
    ):

        # Load Dataset
        self.loader = self.loader_manager.initialize(
            dataset_type,
            dataset_path
        )

        print("Dataset initialized.")

        # Profile Every Table

    ##################################################################

    def _profile_tables(self):

        self.profiles = {}

        for table in self.loader.get_tables():

            dataframe = self.loader.get_dataframe(table)

            self.profiles[table] = self.profiler.profile(
                table,
                dataframe
            )

        print("Dataset profiling completed.")

    ##################################################################

    def get_tables(self):

        return self.loader.get_tables()

    ##################################################################

    def get_dataframe(
        self,
        table
    ):

        return self.loader.get_dataframe(table)

    ##################################################################

    def get_schema(self):

        return self.loader.get_schema()

    ##################################################################

    def get_profiles(self):

        return self.profiles

    ##################################################################

    def get_profile(
        self,
        table
    ):

        return self.profiles.get(table)

        





       