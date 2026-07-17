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