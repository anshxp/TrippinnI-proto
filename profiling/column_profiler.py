import pandas as pd

from profiling.statistics import StatisticsProfiler
from profiling.datatype_profiler import DatatypeProfiler


class ColumnProfiler:
    """Profiles columns while keeping streaming state bounded."""

    MAX_UNIQUE_VALUES = 5000
    SAMPLE_SIZE = 100

    def __init__(self):
        self.statistics = StatisticsProfiler()
        self.datatype = DatatypeProfiler()

    def profile(self, dataframe: pd.DataFrame) -> dict:
        return {column: self.profile_column(column, dataframe[column]) for column in dataframe.columns}

    def start_streaming(self) -> dict:
        return {"columns": {}}

    def update_streaming(self, state: dict, dataframe: pd.DataFrame) -> None:
        for column in dataframe.columns:
            series = dataframe[column]
            column_state = state["columns"].get(column)
            if column_state is None:
                is_numeric = pd.api.types.is_numeric_dtype(series)
                column_state = {"dtype": str(series.dtype), "is_numeric": is_numeric, "rows": 0, "null_count": 0, "memory_usage": 0, "unique_values": set(), "unique_overflow": False, "sample": [], "statistics": self.statistics.start_streaming(is_numeric)}
                state["columns"][column] = column_state

            column_state["rows"] += len(series)
            column_state["null_count"] += int(series.isna().sum())
            column_state["memory_usage"] += int(series.memory_usage(deep=True))
            values = series.dropna()

            if not column_state["unique_overflow"]:
                for value in values:
                    try:
                        column_state["unique_values"].add(value)
                    except TypeError:
                        continue
                    if len(column_state["unique_values"]) > self.MAX_UNIQUE_VALUES:
                        column_state["unique_values"].clear()
                        column_state["unique_overflow"] = True
                        break

            remaining = self.SAMPLE_SIZE - len(column_state["sample"])
            if remaining > 0:
                column_state["sample"].extend(values.iloc[:remaining].tolist())

            self.statistics.update_streaming(column_state["statistics"], series)

    def finalize_streaming(self, state: dict) -> dict:
        columns = {}
        for column, column_state in state["columns"].items():
            rows = column_state["rows"]
            exact = not column_state["unique_overflow"]
            unique_count = len(column_state["unique_values"]) if exact else self.MAX_UNIQUE_VALUES + 1
            sample = pd.Series(column_state["sample"], dtype=column_state["dtype"])
            columns[column] = {
                "name": column,
                "pandas_dtype": column_state["dtype"],
                "semantic_type": self.datatype.infer_streaming(column, column_state["dtype"], sample, rows - column_state["null_count"], unique_count),
                "memory_usage": int(column_state["memory_usage"]),
                "unique_values": int(unique_count),
                "unique_values_exact": exact,
                "null_count": int(column_state["null_count"]),
                "null_percentage": round(column_state["null_count"] / max(rows, 1) * 100, 2),
                "statistics": self.statistics.finalize_streaming(column_state["statistics"], rows),
            }
        return columns

    def profile_chunks(self, chunks) -> dict:
        state = self.start_streaming()
        for chunk in chunks:
            self.update_streaming(state, chunk)
        return self.finalize_streaming(state)

    def profile_column(self, column_name: str, series: pd.Series) -> dict:
        return {"name": column_name, "pandas_dtype": str(series.dtype), "semantic_type": self.datatype.infer(column_name, series), "memory_usage": int(series.memory_usage(deep=True)), "unique_values": int(series.nunique(dropna=True)), "unique_values_exact": True, "null_count": int(series.isna().sum()), "null_percentage": round(float(series.isna().mean() * 100), 2), "statistics": self.statistics.profile(series)}
