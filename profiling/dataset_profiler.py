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

                float(

                    dataframe.isna().sum().sum()

                    /

                    dataframe.size

                ) * 100,

                2
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