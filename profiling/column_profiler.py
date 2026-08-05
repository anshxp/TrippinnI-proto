import pandas as pd

from profiling.statistics import StatisticsProfiler
from profiling.datatype_profiler import DatatypeProfiler


class ColumnProfiler:
    """
    Profiles every column in a dataframe.

    Produces column-level metadata.
    """

    def __init__(self):

        self.statistics = StatisticsProfiler()
        self.datatype = DatatypeProfiler()

    ###################################################################

    def profile(self, dataframe: pd.DataFrame) -> dict:

        columns = {}

        for column in dataframe.columns:

            columns[column] = self.profile_column(
                column,
                dataframe[column]
            )

        return columns

    def start_streaming(self) -> dict:
        """Create bounded per-column state for chunked profiling."""

        return {"columns": {}}

    def update_streaming(self, state: dict, dataframe: pd.DataFrame) -> None:
        """Accumulate column profiles from one DataFrame chunk."""

        for column in dataframe.columns:
            series = dataframe[column]
            column_state = state["columns"].get(column)
            if column_state is None:
                is_numeric = pd.api.types.is_numeric_dtype(series)
                column_state = {
                    "dtype": str(series.dtype),
                    "is_numeric": is_numeric,
                    "rows": 0,
                    "null_count": 0,
                    "memory_usage": 0,
                    "unique_values": set(),
                    "sample": [],
                    "statistics": self.statistics.start_streaming(is_numeric),
                }
                state["columns"][column] = column_state

            column_state["rows"] += len(series)
            column_state["null_count"] += int(series.isna().sum())
            column_state["memory_usage"] += int(series.memory_usage(deep=True))
            values = series.dropna()
            column_state["unique_values"].update(values.tolist())
            self._add_to_sample(column_state["sample"], values)
            self.statistics.update_streaming(column_state["statistics"], series)

    def finalize_streaming(self, state: dict) -> dict:
        """Return the normal column-profile schema from streaming state."""

        columns = {}
        for column, column_state in state["columns"].items():
            rows = column_state["rows"]
            sample = pd.Series(column_state["sample"], dtype=column_state["dtype"])
            columns[column] = {
                "name": column,
                "pandas_dtype": column_state["dtype"],
                "semantic_type": self.datatype.infer_streaming(
                    column,
                    column_state["dtype"],
                    sample,
                    rows - column_state["null_count"],
                    len(column_state["unique_values"]),
                ),
                "memory_usage": int(column_state["memory_usage"]),
                "unique_values": int(len(column_state["unique_values"])),
                "null_count": int(column_state["null_count"]),
                "null_percentage": round(
                    (column_state["null_count"] / max(rows, 1)) * 100,
                    2,
                ),
                "statistics": self.statistics.finalize_streaming(
                    column_state["statistics"], rows
                ),
            }
        return columns

    def profile_chunks(self, chunks) -> dict:
        """Profile a chunk iterator without concatenating its DataFrames."""

        state = self.start_streaming()
        for chunk in chunks:
            self.update_streaming(state, chunk)
        return self.finalize_streaming(state)

    @staticmethod
    def _add_to_sample(sample: list, values: pd.Series, size: int = 100) -> None:
        """Keep only a small type-inference sample for each column."""

        remaining = size - len(sample)
        if remaining > 0:
            sample.extend(values.iloc[:remaining].tolist())

    ###################################################################

    def profile_column(
        self,
        column_name: str,
        series: pd.Series
    ) -> dict:

        semantic_type = self.datatype.infer(
            column_name,
            series
        )

        statistics = self.statistics.profile(series)

        profile = {

            "name": column_name,

            "pandas_dtype": str(series.dtype),

            "semantic_type": semantic_type,

            "memory_usage": int(
                series.memory_usage(deep=True)
            ),

            "unique_values": int(
                series.nunique(dropna=True)
            ),

            "null_count": int(
                series.isna().sum()
            ),

            "null_percentage": round(
                float(series.isna().mean() * 100),
                2
            ),

            "statistics": statistics
        }

        return profile
