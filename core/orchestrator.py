"""
orchestrator.py

Coordinates all TrippinnI modules.
"""

from core.loader import LoaderManager


class Orchestrator:

    def __init__(self):

        self.loader_manager = LoaderManager()

        self.loader = None

    def initialize(self,
                   dataset_type,
                   dataset_path):

        self.loader = self.loader_manager.initialize(
            dataset_type,
            dataset_path
        )

        print("Dataset initialized.")

    def get_tables(self):

        return self.loader.get_tables()

    def get_dataframe(self,
                      table):

        return self.loader.get_dataframe(table)

    def get_schema(self):

        return self.loader.get_schema()