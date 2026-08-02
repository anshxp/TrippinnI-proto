from __future__ import annotations

from pathlib import Path

from loaders.base_loader import BaseLoader
from loaders.factory import LoaderFactory


class LoaderManager:
    """
    Responsible for managing dataset loaders.

    The manager does not know anything about:
    - Synthea
    - MIMIC
    - OMOP
    - FHIR

    It simply requests a loader from the factory.
    """

    def __init__(self):
        self.loader: BaseLoader | None = None

    def initialize(
        self,
        dataset_type: str,
        dataset_path: str | Path,
    ) -> None:
        """
        Create and initialize the appropriate loader.
        """

        self.loader = LoaderFactory.create(
            dataset_type=dataset_type,
            dataset_path=dataset_path,
        )

        self.loader.load()

    def get_loader(self) -> BaseLoader:

        if self.loader is None:
            raise RuntimeError(
                "Loader has not been initialized."
            )

        return self.loader

    def get_tables(self) -> list[str]:

        return self.get_loader().get_tables()

    def get_dataframe(self, table_name: str):

        return self.get_loader().get_dataframe(table_name)

    def get_schema(self):

        return self.get_loader().get_schema()

    def clear_cache(self):

        self.get_loader().clear_cache()