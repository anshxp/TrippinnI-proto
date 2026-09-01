import hashlib
import os
import sqlite3
import tempfile

import pandas as pd


class KeyDetector:
    """Detect candidate primary and foreign keys with bounded RAM."""

    FK_KEYWORDS = {"patient", "encounter", "organization", "provider", "payer", "device"}

    def detect(self, dataframe: pd.DataFrame):
        return {"primary_keys": self.primary_keys(dataframe), "foreign_keys": self.foreign_keys(dataframe)}

    def primary_keys(self, dataframe):
        return [column for column in dataframe.columns if dataframe[column].is_unique]

    def foreign_keys(self, dataframe):
        return [column for column in dataframe.columns if any(keyword in column.lower() for keyword in self.FK_KEYWORDS)]

    def start_streaming(self) -> dict:
        tmp = tempfile.NamedTemporaryFile(prefix="trippinni_keys_", suffix=".sqlite", delete=False)
        tmp.close()
        connection = sqlite3.connect(tmp.name)
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("CREATE TABLE seen_values (column_name TEXT NOT NULL, value_hash BLOB NOT NULL, PRIMARY KEY(column_name, value_hash))")
        connection.commit()
        return {"columns": [], "not_unique": set(), "key_db": connection, "key_db_path": tmp.name}

    def update_streaming(self, state: dict, dataframe: pd.DataFrame) -> None:
        state["columns"] = list(dataframe.columns)
        cursor = state["key_db"].cursor()

        for column in dataframe.columns:
            if column in state["not_unique"]:
                continue

            series = dataframe[column]
            if series.isna().any():
                state["not_unique"].add(column)
                continue

            for value_hash in pd.util.hash_pandas_object(series, index=False):
                digest = hashlib.blake2b(str(int(value_hash)).encode(), digest_size=8).digest()
                cursor.execute("INSERT OR IGNORE INTO seen_values(column_name, value_hash) VALUES (?, ?)", (column, digest))
                if cursor.rowcount == 0:
                    state["not_unique"].add(column)
                    break

        state["key_db"].commit()

    def finalize_streaming(self, state: dict) -> dict:
        primary_keys = [column for column in state["columns"] if column not in state["not_unique"]]
        foreign_keys = [column for column in state["columns"] if any(keyword in column.lower() for keyword in self.FK_KEYWORDS)]

        connection = state.pop("key_db")
        path = state.pop("key_db_path")
        connection.close()
        try:
            os.unlink(path)
        except OSError:
            pass

        return {"primary_keys": primary_keys, "foreign_keys": foreign_keys}

    def detect_chunks(self, chunks) -> dict:
        state = self.start_streaming()
        try:
            for chunk in chunks:
                self.update_streaming(state, chunk)
            return self.finalize_streaming(state)
        except Exception:
            connection = state.get("key_db")
            path = state.get("key_db_path")
            if connection:
                connection.close()
            if path:
                try:
                    os.unlink(path)
                except OSError:
                    pass
            raise
