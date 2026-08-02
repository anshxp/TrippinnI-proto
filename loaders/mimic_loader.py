from __future__ import annotations

from pathlib import Path

import pandas as pd

from loaders.base_loader import BaseLoader
from loaders.catalog.dataset_catalog import DatasetCatalog
from loaders.readers.csv_reader import CsvReader


class MimicLoader(BaseLoader):
    """
    Loader for MIMIC-IV datasets.

    Supports:

        mimic/
            hosp/
            icu/

    including:

        *.csv
        *.csv.gz

    DataFrames are loaded lazily.
    """

    def __init__(
        self,
        manifest,
        validator,
        discovery,
        readers,
        catalog: DatasetCatalog,
    ):
        super().__init__(
            manifest,
            validator,
            discovery,
            readers,
            catalog,
        )

        self.reader: CsvReader = readers["csv"]

    ##################################################################

    def load(self):

        self.validator.validate(self.manifest)

        files = self.discovery.discover(
            self.manifest.root_path
        )

        for file in files:

            if not self.reader.supports(file):
                continue

            table_name = self._table_name(
                file.relative_to(self.manifest.root_path)
            )

            self.catalog.register_table(
                table_name,
                file,
            )

    ##################################################################

    def get_tables(self):

        return self.catalog.get_tables()

    ##################################################################

    def get_dataframe(
        self,
        table_name: str,
    ) -> pd.DataFrame:

        table_name = table_name.lower()

        cached = self.catalog.get_cached_dataframe(
            table_name
        )

        if cached is not None:
            return cached

        file_path = self.catalog.get_table_path(
            table_name
        )

        dataframe = self.reader.read(file_path)

        self.catalog.cache_dataframe(
            table_name,
            dataframe,
        )

        return dataframe

    ##################################################################

    def get_schema(self):

        schema = {}

        for table in self.get_tables():

            dataframe = self.get_dataframe(table)

            schema[table] = {

                "rows": len(dataframe),

                "columns": list(dataframe.columns),

                "shape": dataframe.shape,

                "dtypes": {
                    column: str(dtype)
                    for column, dtype in dataframe.dtypes.items()
                },
            }

        return schema

    ##################################################################

    @staticmethod
    def _table_name(file: Path) -> str:
        """
        Convert paths such as

            hosp/patients.csv
            icu/chartevents.csv.gz

        into

            hosp_patients
            icu_chartevents

        Including the relative directory prevents collisions between
        identically named files in different MIMIC-IV directories.
        """

        name = file.name.lower()

        if name.endswith(".csv.gz"):
            table_name = name[:-7]

        elif name.endswith(".csv"):
            table_name = name[:-4]

        else:
            table_name = file.stem.lower()

        directories = [part.lower() for part in file.parts[:-1]]

        return "_".join([*directories, table_name])
