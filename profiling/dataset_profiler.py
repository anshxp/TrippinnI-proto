import pandas as pd


class DatasetProfiler:
    """
    Computes table-level metadata.
    """

    ###################################################################

    def profile(
        self,
        table_name: str,
        dataframe: pd.DataFrame
    ) -> dict:

        return {

            "table_name": table_name,

            "rows": int(len(dataframe)),

            "columns": int(len(dataframe.columns)),

            "shape": dataframe.shape,

            "memory_usage": int(
                dataframe.memory_usage(
                    deep=True
                ).sum()
            ),

            "duplicate_rows": int(
                dataframe.duplicated().sum()
            ),

            "missing_cells": int(
                dataframe.isna().sum().sum()
            ),

            "missing_percentage": round(
                float(dataframe.isna().sum().sum() / max(dataframe.size, 1)) * 100,
                2,
            ),

            "numeric_columns": len(

                dataframe.select_dtypes(
                    include="number"
                ).columns

            ),

            "categorical_columns": len(

                dataframe.select_dtypes(
                    include="object"
                ).columns

            ),

            "datetime_columns": len(

                dataframe.select_dtypes(
                    include="datetime"
                ).columns

            ),

            "boolean_columns": len(

                dataframe.select_dtypes(
                    include="bool"
                ).columns

            )

        }

    def start_streaming(self, table_name: str) -> dict:
        return {
            "table_name": table_name,
            "rows": 0,
            "columns": [],
            "memory_usage": 0,
            "duplicate_rows": 0,
            "seen_rows": set(),
            "missing_cells": 0,
            "dtypes": {},
        }

    def update_streaming(self, state: dict, dataframe: pd.DataFrame) -> None:
        state["rows"] += len(dataframe)
        state["columns"] = list(dataframe.columns)
        state["memory_usage"] += int(dataframe.memory_usage(deep=True).sum())
        state["missing_cells"] += int(dataframe.isna().sum().sum())
        state["dtypes"].update({column: dtype for column, dtype in dataframe.dtypes.items()})

        row_hashes = pd.util.hash_pandas_object(dataframe, index=False)
        for row_hash in row_hashes:
            if row_hash in state["seen_rows"]:
                state["duplicate_rows"] += 1
            else:
                state["seen_rows"].add(row_hash)

    def finalize_streaming(self, state: dict) -> dict:
        columns = state["columns"]
        dtypes = pd.Series(state["dtypes"])
        size = state["rows"] * len(columns)
        return {
            "table_name": state["table_name"],
            "rows": int(state["rows"]),
            "columns": int(len(columns)),
            "shape": (state["rows"], len(columns)),
            "memory_usage": int(state["memory_usage"]),
            "duplicate_rows": int(state["duplicate_rows"]),
            "missing_cells": int(state["missing_cells"]),
            "missing_percentage": round(
                (state["missing_cells"] / max(size, 1)) * 100,
                2,
            ),
            "numeric_columns": int(sum(pd.api.types.is_numeric_dtype(dtype) for dtype in dtypes)),
            "categorical_columns": int(sum(pd.api.types.is_object_dtype(dtype) for dtype in dtypes)),
            "datetime_columns": int(sum(pd.api.types.is_datetime64_any_dtype(dtype) for dtype in dtypes)),
            "boolean_columns": int(sum(pd.api.types.is_bool_dtype(dtype) for dtype in dtypes)),
        }

    def profile_chunks(self, table_name: str, chunks) -> dict:
        state = self.start_streaming(table_name)
        for chunk in chunks:
            self.update_streaming(state, chunk)
        return self.finalize_streaming(state)
