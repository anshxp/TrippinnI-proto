import pandas as pd


class KeyDetector:
    """Detect candidate keys over the bounded prototype profiling stream."""

    FK_KEYWORDS = {"patient", "encounter", "organization", "provider", "payer", "device"}

    def detect(self, dataframe: pd.DataFrame):
        return {"primary_keys": self.primary_keys(dataframe), "foreign_keys": self.foreign_keys(dataframe)}

    def primary_keys(self, dataframe):
        return [column for column in dataframe.columns if dataframe[column].notna().all() and dataframe[column].is_unique]

    def foreign_keys(self, dataframe):
        return [column for column in dataframe.columns if any(keyword in column.lower() for keyword in self.FK_KEYWORDS)]

    def start_streaming(self) -> dict:
        return {"columns": [], "not_unique": set(), "seen_values": {}}

    def update_streaming(self, state: dict, dataframe: pd.DataFrame) -> None:
        state["columns"] = list(dataframe.columns)
        for column in dataframe.columns:
            if column in state["not_unique"]:
                continue
            series = dataframe[column]
            if series.isna().any():
                state["not_unique"].add(column)
                continue
            seen = state["seen_values"].setdefault(column, set())
            for value in series.tolist():
                try:
                    duplicate = value in seen
                except TypeError:
                    value = repr(value)
                    duplicate = value in seen
                if duplicate:
                    state["not_unique"].add(column)
                    seen.clear()
                    break
                seen.add(value)

    def finalize_streaming(self, state: dict) -> dict:
        primary_keys = [column for column in state["columns"] if column not in state["not_unique"]]
        foreign_keys = [column for column in state["columns"] if any(keyword in column.lower() for keyword in self.FK_KEYWORDS)]
        return {"primary_keys": primary_keys, "foreign_keys": foreign_keys}

    def detect_chunks(self, chunks) -> dict:
        state = self.start_streaming()
        for chunk in chunks:
            self.update_streaming(state, chunk)
        return self.finalize_streaming(state)
