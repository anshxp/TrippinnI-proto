from __future__ import annotations

from pathlib import Path

import pandas as pd


class CsvReader:
    """
    Responsible only for reading CSV files.

    Supports:
    - .csv
    - .csv.gz

    This class does NOT:
    - discover files
    - validate datasets
    - cache DataFrames
    - infer schemas
    """

    SUPPORTED_EXTENSIONS = {
        ".csv",
        ".csv.gz",
    }

    def __init__(self, **read_csv_kwargs):
        self.read_csv_kwargs = read_csv_kwargs

    def read(self, file_path: str | Path) -> pd.DataFrame:
        """
        Read a CSV or compressed CSV into a DataFrame.
        """

        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        if not self.supports(file_path):
            raise ValueError(
                f"Unsupported file type: {file_path.name}"
            )

        # Pandas automatically detects gzip compression.
        return pd.read_csv(file_path, **self.read_csv_kwargs)

    def supports(self, file_path: str | Path) -> bool:
        """
        Returns True if this reader supports the file.
        """

        path = Path(file_path)

        suffix = "".join(path.suffixes[-2:]).lower()

        if suffix == ".csv.gz":
            return True

        return path.suffix.lower() == ".csv"