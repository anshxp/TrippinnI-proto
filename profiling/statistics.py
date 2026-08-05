import pandas as pd
import numpy as np
from collections import Counter
import random


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

    def start_streaming(self, is_numeric: bool) -> dict:
        """Create state for statistics accumulated one chunk at a time."""

        return {
            "is_numeric": is_numeric,
            "count": 0,
            "missing": 0,
            "mean": 0.0,
            "m2": 0.0,
            "minimum": None,
            "maximum": None,
            "values": Counter(),
            # Quantiles use a fixed-size reservoir rather than retaining a
            # full column. Mean, variance, min, max, mode and frequencies are
            # still accumulated exactly.
            "reservoir": [],
            "seen_for_reservoir": 0,
            "random": random.Random(42),
        }

    def update_streaming(self, state: dict, series: pd.Series) -> None:
        """Add one series chunk to a streaming statistics state."""

        state["missing"] += int(series.isna().sum())

        if state["is_numeric"]:
            values = pd.to_numeric(series, errors="coerce").dropna()
            for value in values:
                value = float(value)
                state["count"] += 1
                delta = value - state["mean"]
                state["mean"] += delta / state["count"]
                state["m2"] += delta * (value - state["mean"])
                state["minimum"] = value if state["minimum"] is None else min(state["minimum"], value)
                state["maximum"] = value if state["maximum"] is None else max(state["maximum"], value)
                state["values"][value] += 1
                self._add_to_reservoir(state, value)
            return

        values = series.dropna()
        state["count"] += int(len(values))
        state["values"].update(values.tolist())

    def finalize_streaming(self, state: dict, total_rows: int) -> dict:
        """Convert a streaming state into the existing statistics schema."""

        missing_percentage = round(
            (state["missing"] / max(total_rows, 1)) * 100,
            2,
        )

        if not state["is_numeric"]:
            top, frequency = self._counter_mode(state["values"])
            return {
                "count": int(state["count"]),
                "missing": int(state["missing"]),
                "missing_percentage": missing_percentage,
                "unique": int(len(state["values"])),
                "top": None if top is None else str(top),
                "frequency": int(frequency),
            }

        reservoir = pd.Series(state["reservoir"], dtype="float64")
        q1 = reservoir.quantile(0.25) if not reservoir.empty else np.nan
        q2 = reservoir.quantile(0.50) if not reservoir.empty else np.nan
        q3 = reservoir.quantile(0.75) if not reservoir.empty else np.nan
        mode, _ = self._counter_mode(state["values"])
        variance = (
            state["m2"] / (state["count"] - 1)
            if state["count"] > 1 else np.nan
        )
        return {
            "count": int(state["count"]),
            "missing": int(state["missing"]),
            "missing_percentage": missing_percentage,
            "mean": self._safe_float(state["mean"] if state["count"] else np.nan),
            "median": self._safe_float(q2),
            "mode": self._safe_float(mode),
            "minimum": self._safe_float(state["minimum"]),
            "maximum": self._safe_float(state["maximum"]),
            "range": self._safe_float(
                state["maximum"] - state["minimum"]
                if state["minimum"] is not None else np.nan
            ),
            "variance": self._safe_float(variance),
            "std": self._safe_float(np.sqrt(variance)),
            "q1": self._safe_float(q1),
            "q2": self._safe_float(q2),
            "q3": self._safe_float(q3),
            "iqr": self._safe_float(q3 - q1),
        }

    @staticmethod
    def _add_to_reservoir(state: dict, value: float, size: int = 5000) -> None:
        state["seen_for_reservoir"] += 1
        reservoir = state["reservoir"]
        if len(reservoir) < size:
            reservoir.append(value)
            return
        position = state["random"].randint(0, state["seen_for_reservoir"] - 1)
        if position < size:
            reservoir[position] = value

    @staticmethod
    def _counter_mode(values: Counter):
        if not values:
            return None, 0
        frequency = max(values.values())
        candidates = [value for value, count in values.items() if count == frequency]
        try:
            return min(candidates), frequency
        except TypeError:
            return candidates[0], frequency

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
