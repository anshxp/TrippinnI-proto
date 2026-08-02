from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from loaders.catalog.dataset_catalog import DatasetCatalog
from loaders.manifest.dataset_manifest import DatasetManifest


class BaseLoader(ABC):
    """
    Base interface for all dataset loaders.

    Every dataset loader should:
        1. Discover files.
        2. Register them in the catalog.
        3. Lazily load DataFrames.
        4. Expose a common API to the pipeline.
    """

    def __init__(
        self,
        manifest: DatasetManifest,
        validator,
        discovery,
        readers: dict,
        catalog: DatasetCatalog,
    ):
        self.manifest = manifest
        self.validator = validator
        self.discovery = discovery
        self.readers = readers
        self.catalog = catalog

    # ---------------------------------------------------------
    # Discovery
    # ---------------------------------------------------------

    @abstractmethod
    def load(self) -> None:
        """
        Discover dataset files and register them.

        This method should NOT load DataFrames into memory.
        """
        pass

    # ---------------------------------------------------------
    # Dataset API
    # ---------------------------------------------------------

    @abstractmethod
    def get_tables(self) -> list[str]:
        """
        Return all available table names.
        """
        pass

    @abstractmethod
    def get_dataframe(
        self,
        table_name: str,
    ) -> pd.DataFrame:
        """
        Return a DataFrame for a table.

        Implementations should use lazy loading.
        """
        pass

    @abstractmethod
    def get_schema(self) -> dict:
        """
        Return schema information for the dataset.
        """
        pass

    # ---------------------------------------------------------
    # Convenience Methods
    # ---------------------------------------------------------

    def has_table(self, table_name: str) -> bool:
        """
        Check if a table exists.
        """

        return self.catalog.has_table(table_name.lower())

    def table_count(self) -> int:
        """
        Number of discovered tables.
        """

        return self.catalog.table_count

    def clear_cache(self) -> None:
        """
        Release all cached DataFrames.
        """

        self.catalog.clear_cache()