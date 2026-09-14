from __future__ import annotations

from pathlib import Path
from typing import Iterator

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

    def _read_options(self, *, chunksize: int | None = None) -> dict:
        """Build pandas options while keeping mixed-type inference consistent."""
        options = dict(self.read_csv_kwargs)
        options.setdefault("low_memory", False)
        if chunksize is not None:
            options["chunksize"] = chunksize
        return options

    def read(self, file_path: str | Path) -> pd.DataFrame:
        """Read a CSV or compressed CSV into a DataFrame."""
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        if not self.supports(file_path):
            raise ValueError(f"Unsupported file type: {file_path.name}")

        # low_memory=False makes pandas infer each column consistently
        # instead of emitting repeated mixed-type DtypeWarnings while
        # processing the file in internal low-memory passes.
        return pd.read_csv(file_path, **self._read_options())

    def read_chunks(
        self,
        file_path: str | Path,
        chunksize: int = 100000,
    ) -> Iterator[pd.DataFrame]:
        """Read a CSV as an iterator of DataFrame chunks."""
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        if not self.supports(file_path):
            raise ValueError(f"Unsupported file type: {file_path.name}")

        if chunksize <= 0:
            raise ValueError("chunksize must be greater than zero")

        # chunksize still streams the source; low_memory=False only controls
        # pandas' dtype inference within each read operation.
        return pd.read_csv(
            file_path,
            **self._read_options(chunksize=chunksize),
        )

    def supports(self, file_path: str | Path) -> bool:
        """Return True if this reader supports the file."""
        path = Path(file_path)
        suffix = "".join(path.suffixes[-2:]).lower()

        if suffix == ".csv.gz":
            return True

        return path.suffix.lower() == ".csv"
