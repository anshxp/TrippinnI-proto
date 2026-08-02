from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


class DatasetCatalog:
    """
    Stores all dataset information after discovery.

    The catalog is the loader's internal storage.

    Responsibilities:
    - Register discovered tables/files.
    - Cache loaded DataFrames.
    - Store schema information.
    - Store optional metadata.
    """

    def __init__(self) -> None:
        self._tables: dict[str, Path] = {}
        self._cache: dict[str, pd.DataFrame] = {}
        self._schemas: dict[str, dict[str, Any]] = {}
        self._metadata: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Table registration
    # ------------------------------------------------------------------

    def register_table(self, table_name: str, file_path: Path) -> None:
        """Register a discovered table."""

        self._tables[table_name] = file_path

    def get_table_path(self, table_name: str) -> Path:
        """Return the file path for a registered table."""

        return self._tables[table_name]

    def get_tables(self) -> list[str]:
        """Return all registered table names."""

        return sorted(self._tables.keys())

    def has_table(self, table_name: str) -> bool:
        return table_name in self._tables

    # ------------------------------------------------------------------
    # DataFrame cache
    # ------------------------------------------------------------------

    def cache_dataframe(
        self,
        table_name: str,
        dataframe: pd.DataFrame,
    ) -> None:
        """Cache a loaded DataFrame."""

        self._cache[table_name] = dataframe

    def get_cached_dataframe(
        self,
        table_name: str,
    ) -> pd.DataFrame | None:
        """Return cached DataFrame if available."""

        return self._cache.get(table_name)

    def is_cached(self, table_name: str) -> bool:
        return table_name in self._cache

    def clear_cache(self) -> None:
        """Clear all cached DataFrames."""

        self._cache.clear()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def set_schema(
        self,
        table_name: str,
        schema: dict[str, Any],
    ) -> None:
        self._schemas[table_name] = schema

    def get_schema(
        self,
        table_name: str,
    ) -> dict[str, Any] | None:
        return self._schemas.get(table_name)

    def get_all_schemas(self) -> dict[str, dict[str, Any]]:
        return self._schemas

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def set_metadata(self, key: str, value: Any) -> None:
        self._metadata[key] = value

    def get_metadata(self, key: str) -> Any:
        return self._metadata.get(key)

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def tables(self) -> dict[str, Path]:
        return self._tables

    @property
    def cache(self) -> dict[str, pd.DataFrame]:
        return self._cache

    @property
    def schemas(self) -> dict[str, dict[str, Any]]:
        return self._schemas

    @property
    def table_count(self) -> int:
        return len(self._tables)