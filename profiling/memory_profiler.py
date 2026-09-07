import pandas as pd


class MemoryProfiler:
    """Compute memory-usage statistics without retaining table data."""

    def profile(self, dataframe: pd.DataFrame) -> dict:
        total = int(dataframe.memory_usage(deep=True).sum())
        return {
            "total_memory_bytes": total,
            "total_memory_mb": round(total / (1024 ** 2), 2),
            "average_row_size_bytes": round(total / max(len(dataframe), 1), 2),
            "peak_chunk_memory_bytes": total,
            "column_memory": {
                column: int(dataframe[column].memory_usage(deep=True))
                for column in dataframe.columns
            },
        }

    def start_streaming(self) -> dict:
        return {
            "total_memory_bytes": 0,
            "rows": 0,
            "peak_chunk_memory_bytes": 0,
            "column_memory": {},
        }

    def update_streaming(self, state: dict, dataframe: pd.DataFrame) -> None:
        chunk_memory = int(dataframe.memory_usage(deep=True).sum())
        state["rows"] += len(dataframe)
        state["total_memory_bytes"] += chunk_memory
        state["peak_chunk_memory_bytes"] = max(
            state["peak_chunk_memory_bytes"], chunk_memory
        )
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
            "peak_chunk_memory_bytes": int(state["peak_chunk_memory_bytes"]),
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
