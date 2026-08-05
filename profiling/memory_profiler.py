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

    def start_streaming(self) -> dict:
        return {"total_memory_bytes": 0, "rows": 0, "column_memory": {}}

    def update_streaming(self, state: dict, dataframe: pd.DataFrame) -> None:
        state["rows"] += len(dataframe)
        state["total_memory_bytes"] += int(dataframe.memory_usage(deep=True).sum())
        for column in dataframe.columns:
            state["column_memory"][column] = (
                state["column_memory"].get(column, 0)
                + int(dataframe[column].memory_usage(deep=True))
            )

    def finalize_streaming(self, state: dict) -> dict:
        total = state["total_memory_bytes"]
        return {
            "total_memory_bytes": int(total),
            "total_memory_mb": round(total / (1024 ** 2), 2),
            "average_row_size_bytes": round(total / max(state["rows"], 1), 2),
            "column_memory": {
                column: int(memory)
                for column, memory in state["column_memory"].items()
            },
        }

    def profile_chunks(self, chunks) -> dict:
        state = self.start_streaming()
        for chunk in chunks:
            self.update_streaming(state, chunk)
        return self.finalize_streaming(state)
