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

        # Table name -> CSV file path
        self.tables = {}

        # Loaded DataFrames cache
        self.cache = {}

    def load(self):

        csv_files = list(self.dataset_path.glob("*.csv"))

        if len(csv_files) == 0:
            raise FileNotFoundError(
                f"No CSV files found inside {self.dataset_path}"
            )

        for file in csv_files:

            table_name = file.stem.lower()

            # Store only the file path
            self.tables[table_name] = file

        print(f"Registered {len(self.tables)} tables.")

    def get_tables(self):

        return list(self.tables.keys())

    def get_dataframe(self, table_name):

        if table_name not in self.tables:
            raise ValueError(f"{table_name} not found.")

        # Return cached DataFrame if already loaded
        if table_name in self.cache:
            return self.cache[table_name]

        # Load from disk
        df = pd.read_csv(self.tables[table_name])

        # Store in cache
        self.cache[table_name] = df

        return df

    def get_schema(self):

        schema = {}

        for name in self.tables:

            df = self.get_dataframe(name)

            schema[name] = {

                "rows": len(df),

                "columns": list(df.columns),

                "shape": df.shape,

                "dtypes": df.dtypes.astype(str).to_dict(),

            }

        return schema