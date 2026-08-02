from __future__ import annotations

from pathlib import Path

import pandas as pd

from loaders.base_loader import BaseLoader
from loaders.catalog.dataset_catalog import DatasetCatalog
from loaders.manifest.dataset_manifest import DatasetManifest
from loaders.readers.csv_reader import CsvReader
from loaders.discovery.flat_discovery import FlatDiscovery
from loaders.validator.dataset_validator import DatasetValidator


class SyntheaLoader(BaseLoader):
    """
    Loader for Synthea datasets.

    Supports:
        - CSV
        - CSV.GZ

    Directory layout:

        synthea/
            patients.csv
            encounters.csv
            medications.csv
            ...
    """

    def __init__(
        self,
        manifest: DatasetManifest,
        validator: DatasetValidator,
        discovery: FlatDiscovery,
        readers: dict[str, object],
        catalog: DatasetCatalog,
    ):
        self.manifest = manifest
        self.validator = validator
        self.discovery = discovery
        self.catalog = catalog

        self.reader: CsvReader = readers["csv"]

    # -------------------------------------------------------
    # Discovery
    # -------------------------------------------------------

    def load(self) -> None:
        """
        Discover dataset files.

        No CSV is loaded into memory here.
        """

        self.validator.validate(self.manifest)

        files = self.discovery.discover(
            self.manifest.root_path
        )

        for file in files:

            if not self.reader.supports(file):
                continue

            table_name = file.stem

            # Handle .csv.gz correctly
            if table_name.endswith(".csv"):
                table_name = Path(table_name).stem

            self.catalog.register_table(
                table_name.lower(),
                file,
            )

    # -------------------------------------------------------
    # Tables
    # -------------------------------------------------------

    def get_tables(self) -> list[str]:
        return self.catalog.get_tables()

    # -------------------------------------------------------
    # Lazy Loading
    # -------------------------------------------------------

    def get_dataframe(
        self,
        table_name: str,
    ) -> pd.DataFrame:

        table_name = table_name.lower()

        if self.catalog.is_cached(table_name):
            return self.catalog.get_cached_dataframe(
                table_name
            )

        file_path = self.catalog.get_table_path(
            table_name
        )

        dataframe = self.reader.read(file_path)

        self.catalog.cache_dataframe(
            table_name,
            dataframe,
        )

        return dataframe

    # -------------------------------------------------------
    # Schema
    # -------------------------------------------------------

    def get_schema(self) -> dict:

        schema = {}

        for table in self.get_tables():

            df = self.get_dataframe(table)

            schema[table] = {
                "rows": len(df),
                "columns": list(df.columns),
                "shape": df.shape,
                "dtypes": {
                    c: str(t)
                    for c, t in df.dtypes.items()
                },
            }

        return schema