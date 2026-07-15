"""
pipeline.py

Main execution pipeline.
"""

from core.orchestrator import Orchestrator


class Pipeline:

    def __init__(self):

        self.orchestrator = Orchestrator()

    def run(self,
            dataset_type,
            dataset_path):

        self.orchestrator.initialize(
            dataset_type,
            dataset_path
        )

        return self.orchestrator