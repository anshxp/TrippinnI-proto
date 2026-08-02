from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


class JsonReader:
    """
    Responsible only for reading JSON files.

    Supports:
    - JSON objects
    - JSON arrays

    Does NOT:
    - discover files
    - validate datasets
    - convert FHIR resources
    - cache DataFrames
    """

    def read(self, file_path: str | Path) -> Any:
        """
        Read a JSON file.

        Returns the parsed Python object.
        """

        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"JSON file not found: {file_path}")

        with file_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def read_dataframe(self, file_path: str | Path) -> pd.DataFrame:
        """
        Read a JSON file directly into a DataFrame.

        Intended only for JSON datasets that are naturally tabular.
        """

        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"JSON file not found: {file_path}")

        return pd.read_json(file_path)

    def supports(self, file_path: str | Path) -> bool:
        """
        Returns True if this reader supports the file.
        """

        return Path(file_path).suffix.lower() == ".json"