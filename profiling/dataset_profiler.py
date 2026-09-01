from __future__ import annotations

import hashlib
import os
import sqlite3
import tempfile
from typing import Iterable

import pandas as pd


class DatasetProfiler:
    """Computes dataset-level metadata with bounded-RAM streaming support."""

    def profile(self, table_name: str, dataframe: pd.DataFrame) -> dict:
        memory = int(dataframe.memory_usage(deep=True).sum())
        missing = int(dataframe.isna().sum().sum())
        return {
            "table_name": table_name,
            "rows": int(len(dataframe)),
            "columns": int(len(dataframe.columns)),
            "shape": dataframe.shape,
            "memory_usage": memory,
            "duplicate_rows": int(dataframe.duplicated().sum()),
            "missing_cells": missing,
            "missing_percentage": round(missing / max(dataframe.size, 1) * 100, 2),
            "numeric_columns": len(dataframe.select_dtypes(include="number").columns),
            "categorical_columns": len(dataframe.select_dtypes(include="object").columns),
            "datetime_columns": len(dataframe.select_dtypes(include="datetime").columns),
            "boolean_columns": len(dataframe.select_dtypes(include="bool").columns),
        }

    def start_streaming(self, table_name: str) -> dict:
        """Create incremental state; exact duplicate fingerprints are disk-backed."""
        tmp = tempfile.NamedTemporaryFile(prefix="trippinni_profile_", suffix=".sqlite", delete=False)
        tmp.close()
        connection = sqlite3.connect(tmp.name)
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("CREATE TABLE seen_rows (row_hash BLOB PRIMARY KEY)")
        connection.commit()
        return {
            "table_name": table_name,
            "rows": 0,
            "columns": [],
            "memory_usage": 0,
            "duplicate_rows": 0,
            "duplicate_db": connection,
            "duplicate_db_path": tmp.name,
            "missing_cells": 0,
            "dtypes": {},
        }

    def update_streaming(self, state: dict, dataframe: pd.DataFrame) -> None:
        state["rows"] += len(dataframe)
        state["columns"] = list(dataframe.columns)
        state["memory_usage"] += int(dataframe.memory_usage(deep=True).sum())
        state["missing_cells"] += int(dataframe.isna().sum().sum())
        state["dtypes"].update({column: dtype for column, dtype in dataframe.dtypes.items()})

        cursor = state["duplicate_db"].cursor()
        for row_hash in pd.util.hash_pandas_object(dataframe, index=False):
            digest = hashlib.blake2b(str(int(row_hash)).encode(), digest_size=8).digest()
            cursor.execute("INSERT OR IGNORE INTO seen_rows(row_hash) VALUES (?)", (digest,))
            if cursor.rowcount == 0:
                state["duplicate_rows"] += 1
        state["duplicate_db"].commit()

    def finalize_streaming(self, state: dict) -> dict:
        columns = state["columns"]
        dtypes = pd.Series(state["dtypes"])
        size = state["rows"] * len(columns)
        result = {
            "table_name": state["table_name"],
            "rows": int(state["rows"]),
            "columns": int(len(columns)),
            "shape": (state["rows"], len(columns)),
            "memory_usage": int(state["memory_usage"]),
            "duplicate_rows": int(state["duplicate_rows"]),
            "missing_cells": int(state["missing_cells"]),
            "missing_percentage": round(state["missing_cells"] / max(size, 1) * 100, 2),
            "numeric_columns": int(sum(pd.api.types.is_numeric_dtype(dtype) for dtype in dtypes)),
            "categorical_columns": int(sum(pd.api.types.is_object_dtype(dtype) for dtype in dtypes)),
            "datetime_columns": int(sum(pd.api.types.is_datetime64_any_dtype(dtype) for dtype in dtypes)),
            "boolean_columns": int(sum(pd.api.types.is_bool_dtype(dtype) for dtype in dtypes)),
        }
        self._cleanup(state)
        return result

    def profile_chunks(self, table_name: str, chunks: Iterable[pd.DataFrame]) -> dict:
        state = self.start_streaming(table_name)
        try:
            for chunk in chunks:
                self.update_streaming(state, chunk)
            return self.finalize_streaming(state)
        except Exception:
            self._cleanup(state)
            raise

    @staticmethod
    def _cleanup(state: dict) -> None:
        connection = state.pop("duplicate_db", None)
        path = state.pop("duplicate_db_path", None)
        if connection is not None:
            connection.close()
        if path:
            try:
                os.unlink(path)
            except OSError:
                pass
