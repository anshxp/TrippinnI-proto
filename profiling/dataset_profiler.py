from __future__ import annotations

import hashlib
import os
import sqlite3
import tempfile
from typing import Iterable

import pandas as pd


class DatasetProfiler:
    """Compute dataset-level metadata with bounded-RAM streaming support."""

    PROFILE_VERSION = "1.0"

    def profile(self, table_name: str, dataframe: pd.DataFrame) -> dict:
        memory = int(dataframe.memory_usage(deep=True).sum())
        missing = int(dataframe.isna().sum().sum())
        rows, columns = len(dataframe), len(dataframe.columns)
        return {
            "profile_version": self.PROFILE_VERSION,
            "table_name": table_name,
            "rows": int(rows),
            "columns": int(columns),
            "shape": (int(rows), int(columns)),
            "memory_usage": memory,
            "memory_usage_mb": round(memory / (1024 ** 2), 2),
            "average_row_size_bytes": round(memory / max(rows, 1), 2),
            "duplicate_rows": int(dataframe.duplicated().sum()),
            "duplicate_percentage": round(float(dataframe.duplicated().mean() * 100), 2) if rows else 0.0,
            "missing_cells": missing,
            "missing_percentage": round(missing / max(dataframe.size, 1) * 100, 2),
            "complete_rows": int((~dataframe.isna().any(axis=1)).sum()),
            "complete_row_percentage": round(float((~dataframe.isna().any(axis=1)).mean() * 100), 2) if rows else 0.0,
            "numeric_columns": len(dataframe.select_dtypes(include="number").columns),
            "categorical_columns": len(dataframe.select_dtypes(include="object", exclude=None).columns),
            "datetime_columns": len(dataframe.select_dtypes(include="datetime").columns),
            "boolean_columns": len(dataframe.select_dtypes(include="bool").columns),
            "empty_columns": [column for column in dataframe.columns if dataframe[column].isna().all()],
            "constant_columns": [column for column in dataframe.columns if dataframe[column].nunique(dropna=False) <= 1],
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
            "peak_chunk_memory": 0,
            "duplicate_rows": 0,
            "duplicate_db": connection,
            "duplicate_db_path": tmp.name,
            "missing_cells": 0,
            "complete_rows": 0,
            "dtypes": {},
            "empty_columns": set(),
            "constant_values": {},
            "constant_valid": {},
        }

    def update_streaming(self, state: dict, dataframe: pd.DataFrame) -> None:
        rows = len(dataframe)
        chunk_memory = int(dataframe.memory_usage(deep=True).sum())
        state["rows"] += rows
        state["columns"] = list(dataframe.columns)
        state["memory_usage"] += chunk_memory
        state["peak_chunk_memory"] = max(state["peak_chunk_memory"], chunk_memory)
        state["missing_cells"] += int(dataframe.isna().sum().sum())
        state["complete_rows"] += int((~dataframe.isna().any(axis=1)).sum())
        state["dtypes"].update({column: dtype for column, dtype in dataframe.dtypes.items()})

        for column in dataframe.columns:
            series = dataframe[column]
            if series.notna().any():
                state["empty_columns"].discard(column)
            else:
                state["empty_columns"].add(column)
            if state["constant_valid"].get(column, True):
                non_null = series.dropna()
                if non_null.empty:
                    continue
                first = non_null.iloc[0]
                previous = state["constant_values"].get(column, first)
                state["constant_values"][column] = previous
                try:
                    if not bool((non_null == previous).all()):
                        state["constant_valid"][column] = False
                except Exception:
                    state["constant_valid"][column] = False

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
        rows = state["rows"]
        size = rows * len(columns)
        result = {
            "profile_version": self.PROFILE_VERSION,
            "table_name": state["table_name"],
            "rows": int(rows),
            "columns": int(len(columns)),
            "shape": (int(rows), int(len(columns))),
            "memory_usage": int(state["memory_usage"]),
            "memory_usage_mb": round(state["memory_usage"] / (1024 ** 2), 2),
            "average_row_size_bytes": round(state["memory_usage"] / max(rows, 1), 2),
            "peak_chunk_memory_bytes": int(state["peak_chunk_memory"]),
            "duplicate_rows": int(state["duplicate_rows"]),
            "duplicate_percentage": round(state["duplicate_rows"] / max(rows, 1) * 100, 2),
            "missing_cells": int(state["missing_cells"]),
            "missing_percentage": round(state["missing_cells"] / max(size, 1) * 100, 2),
            "complete_rows": int(state["complete_rows"]),
            "complete_row_percentage": round(state["complete_rows"] / max(rows, 1) * 100, 2),
            "numeric_columns": int(sum(pd.api.types.is_numeric_dtype(dtype) for dtype in dtypes)),
            "categorical_columns": int(sum(pd.api.types.is_object_dtype(dtype) for dtype in dtypes)),
            "datetime_columns": int(sum(pd.api.types.is_datetime64_any_dtype(dtype) for dtype in dtypes)),
            "boolean_columns": int(sum(pd.api.types.is_bool_dtype(dtype) for dtype in dtypes)),
            "empty_columns": sorted(state["empty_columns"]),
            "constant_columns": sorted(column for column in columns if state["constant_valid"].get(column, True)),
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
