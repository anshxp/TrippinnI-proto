import hashlib
import sqlite3
import tempfile
from pathlib import Path

import pandas as pd


class KeyDetector:
    """Detect candidate keys over a stream using disk-backed uniqueness tracking."""

    FK_KEYWORDS = {"patient", "encounter", "organization", "provider", "payer", "device"}

    def detect(self, dataframe: pd.DataFrame):
        return {"primary_keys": self.primary_keys(dataframe), "foreign_keys": self.foreign_keys(dataframe)}

    def primary_keys(self, dataframe):
        return [
            column for column in dataframe.columns
            if dataframe[column].notna().all() and dataframe[column].is_unique
        ]

    def foreign_keys(self, dataframe):
        return [
            column for column in dataframe.columns
            if any(keyword in column.lower() for keyword in self.FK_KEYWORDS)
        ]

    def start_streaming(self) -> dict:
        handle = tempfile.NamedTemporaryFile(prefix="trippinni_keys_", suffix=".sqlite3", delete=False)
        path = handle.name
        handle.close()
        connection = sqlite3.connect(path)
        connection.execute(
            "CREATE TABLE seen_values (column_name TEXT NOT NULL, value_hash BLOB NOT NULL, "
            "PRIMARY KEY (column_name, value_hash))"
        )
        connection.commit()
        connection.close()
        return {
            "columns": [],
            "not_unique": set(),
            "database_path": path,
            "rows_seen": 0,
        }

    @staticmethod
    def _fingerprint(value) -> bytes:
        # Hash a stable representation so values remain disk-backed rather than
        # accumulating in Python sets. This is probabilistic key detection.
        payload = (type(value).__name__ + ":" + repr(value)).encode("utf-8", errors="replace")
        return hashlib.blake2b(payload, digest_size=16).digest()

    def update_streaming(self, state: dict, dataframe: pd.DataFrame) -> None:
        state["columns"] = list(dataframe.columns)
        state["rows_seen"] += len(dataframe)
        connection = sqlite3.connect(state["database_path"])
        try:
            for column in dataframe.columns:
                if column in state["not_unique"]:
                    continue
                series = dataframe[column]
                if series.isna().any():
                    state["not_unique"].add(column)
                    connection.execute("DELETE FROM seen_values WHERE column_name = ?", (column,))
                    continue
                for value in series.tolist():
                    fingerprint = self._fingerprint(value)
                    cursor = connection.execute(
                        "INSERT OR IGNORE INTO seen_values (column_name, value_hash) VALUES (?, ?)",
                        (column, fingerprint),
                    )
                    if cursor.rowcount == 0:
                        state["not_unique"].add(column)
                        connection.execute(
                            "DELETE FROM seen_values WHERE column_name = ?", (column,)
                        )
                        break
            connection.commit()
        finally:
            connection.close()

    def finalize_streaming(self, state: dict) -> dict:
        try:
            primary_keys = [
                column for column in state["columns"]
                if column not in state["not_unique"]
            ]
            foreign_keys = [
                column for column in state["columns"]
                if any(keyword in column.lower() for keyword in self.FK_KEYWORDS)
            ]
            return {"primary_keys": primary_keys, "foreign_keys": foreign_keys}
        finally:
            path = state.get("database_path")
            if path:
                try:
                    Path(path).unlink(missing_ok=True)
                except OSError:
                    pass

    def detect_chunks(self, chunks) -> dict:
        state = self.start_streaming()
        try:
            for chunk in chunks:
                self.update_streaming(state, chunk)
            return self.finalize_streaming(state)
        except Exception:
            path = state.get("database_path")
            if path:
                try:
                    Path(path).unlink(missing_ok=True)
                except OSError:
                    pass
            raise
