import pandas as pd


class MemoryProfiler:
    """
    Computes memory usage statistics.
    """

    def profile(self, dataframe: pd.DataFrame) -> dict:

        return {

            "total_memory_bytes": int(
                dataframe.memory_usage(deep=True).sum()
            ),

            "total_memory_mb": round(
                dataframe.memory_usage(deep=True).sum() / (1024 ** 2),
                2
            ),

            "average_row_size_bytes": round(
                dataframe.memory_usage(deep=True).sum()
                / max(len(dataframe), 1),
                2
            ),

            "column_memory": {

                column: int(
                    dataframe[column].memory_usage(deep=True)
                )

                for column in dataframe.columns

            }

        }