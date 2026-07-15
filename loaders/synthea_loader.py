"""
synthea_loader.py

Loads Synthea CSV datasets.
"""

from pathlib import Path

import pandas as pd

from loaders.base_loader import BaseLoader


class SyntheaLoader(BaseLoader):

    def __init__(self, dataset_path: str):

        self.dataset_path = Path(dataset_path)

        self.tables = {}

    def load(self):

        csv_files = list(self.dataset_path.glob("*.csv"))

        if len(csv_files) == 0:
            raise FileNotFoundError(
                f"No CSV files found inside {self.dataset_path}"
            )

        for file in csv_files:

            table_name = file.stem.lower()

            self.tables[table_name] = pd.read_csv(file)

        print(f"Loaded {len(self.tables)} tables.")

    def get_tables(self):

        return list(self.tables.keys())

    def get_dataframe(self, table_name):

        if table_name not in self.tables:
            raise ValueError(f"{table_name} not loaded.")

        return self.tables[table_name]

    def get_schema(self):

        schema = {}

        for name, df in self.tables.items():

            schema[name] = {

                "rows": len(df),

                "columns": list(df.columns),

                "shape": df.shape,

                "dtypes": df.dtypes.astype(str).to_dict(),

            }

        return schema