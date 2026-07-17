import pandas as pd
import numpy as np


class StatisticsProfiler:
    """
    Generic statistics profiler.

    Responsible only for computing statistics.

    No healthcare logic.
    No dataset logic.
    """

    def profile(self, series: pd.Series) -> dict:

        if pd.api.types.is_numeric_dtype(series):
            return self.profile_numeric(series)

        return self.profile_categorical(series)

    ####################################################################

    def profile_numeric(self, series: pd.Series) -> dict:

        numeric = pd.to_numeric(series, errors="coerce")

        q1 = numeric.quantile(0.25)
        q2 = numeric.quantile(0.50)
        q3 = numeric.quantile(0.75)

        return {

            "count": int(numeric.count()),

            "missing": int(numeric.isna().sum()),

            "missing_percentage": round(
                float(numeric.isna().mean() * 100), 2
            ),

            "mean": self._safe_float(numeric.mean()),

            "median": self._safe_float(numeric.median()),

            "mode": self._mode(numeric),

            "minimum": self._safe_float(numeric.min()),

            "maximum": self._safe_float(numeric.max()),

            "range": self._safe_float(
                numeric.max() - numeric.min()
            ),

            "variance": self._safe_float(
                numeric.var()
            ),

            "std": self._safe_float(
                numeric.std()
            ),

            "q1": self._safe_float(q1),

            "q2": self._safe_float(q2),

            "q3": self._safe_float(q3),

            "iqr": self._safe_float(q3 - q1)
        }

    ####################################################################

    def profile_categorical(self, series: pd.Series) -> dict:

        sample = series.dropna()

        if len(sample) > 5000:
            sample = sample.sample(5000, random_state=42)

        return {

            "count": int(series.count()),

            "missing": int(series.isna().sum()),

            "missing_percentage": round(
                float(series.isna().mean() * 100), 2
            ),

            "unique": int(sample.nunique()),

            "top": self._top(series),

            "frequency": self._frequency(series)

        }

    ####################################################################

    def _mode(self, series):

        sample = series.dropna()

        if len(sample) > 5000:
            sample = sample.sample(5000, random_state=42)

        mode = sample.mode()

        if mode.empty:
            return None

        return self._safe_float(mode.iloc[0])

    ####################################################################

    def _top(self, series):

        sample = series.dropna()

        if len(sample) > 5000:
            sample = sample.sample(5000, random_state=42)

        mode = sample.mode()

        if mode.empty:
            return None

        return str(mode.iloc[0])

    ####################################################################

    def _frequency(self, series):

        sample = series.dropna()

        if len(sample) > 5000:
            sample = sample.sample(5000, random_state=42)

        mode = sample.mode()

        if mode.empty:
            return 0

        value = mode.iloc[0]

        return int((series == value).sum())

    ####################################################################

    def _safe_float(self, value):

        if pd.isna(value):
            return None

        return float(value)