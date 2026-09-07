"""Memory helpers used by the streaming prototype pipeline."""

from __future__ import annotations

import gc

import pandas as pd


def downcast_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce the in-memory footprint of a DataFrame where this is safe.

    Integer and floating-point columns are downcast to the smallest
    compatible NumPy dtype. Object columns are left unchanged because
    automatic conversion can materially alter memory use or semantics.
    """
    if dataframe.empty:
        return dataframe

    result = dataframe.copy()

    for column in result.columns:
        dtype = result[column].dtype

        if pd.api.types.is_integer_dtype(dtype):
            result[column] = pd.to_numeric(result[column], downcast="integer")
        elif pd.api.types.is_float_dtype(dtype):
            result[column] = pd.to_numeric(result[column], downcast="float")

    return result


def release(*objects) -> None:
    """Drop local references and request Python garbage collection."""
    for obj in objects:
        del obj
    gc.collect()
