import random
from collections import Counter

import numpy as np
import pandas as pd


class StatisticsProfiler:
    """Computes column statistics with bounded streaming state."""

    MAX_FREQUENCY_VALUES = 5000
    RESERVOIR_SIZE = 5000

    def profile(self, series: pd.Series) -> dict:
        if pd.api.types.is_numeric_dtype(series):
            return self.profile_numeric(series)
        return self.profile_categorical(series)

    def start_streaming(self, is_numeric: bool) -> dict:
        return {
            "is_numeric": is_numeric,
            "count": 0,
            "missing": 0,
            "mean": 0.0,
            "m2": 0.0,
            "minimum": None,
            "maximum": None,
            "values": Counter(),
            "frequency_overflow": False,
            "reservoir": [],
            "seen_for_reservoir": 0,
            "random": np.random.default_rng(42),
        }

    def update_streaming(self, state: dict, series: pd.Series) -> None:
        state["missing"] += int(series.isna().sum())

        if state["is_numeric"]:
            values = pd.to_numeric(series, errors="coerce").dropna()
            n = int(values.size)
            if n == 0:
                return

            chunk_mean = float(values.mean())
            chunk_m2 = float(((values - chunk_mean) ** 2).sum())
            old_count = state["count"]
            new_count = old_count + n
            if old_count == 0:
                state["mean"] = chunk_mean
                state["m2"] = chunk_m2
            else:
                delta = chunk_mean - state["mean"]
                state["m2"] += chunk_m2 + delta * delta * old_count * n / new_count
                state["mean"] += delta * n / new_count
            state["count"] = new_count

            chunk_min = float(values.min())
            chunk_max = float(values.max())
            state["minimum"] = chunk_min if state["minimum"] is None else min(state["minimum"], chunk_min)
            state["maximum"] = chunk_max if state["maximum"] is None else max(state["maximum"], chunk_max)

            self._update_frequency_from_counts(state, values.value_counts(dropna=True))
            self._update_reservoir_from_sample(state, values)
            return

        values = series.dropna()
        state["count"] += int(len(values))
        if not values.empty:
            self._update_frequency_from_counts(state, values.value_counts(dropna=True))

    def _update_frequency_from_counts(self, state: dict, counts: pd.Series) -> None:
        """Merge vectorized pandas value counts into bounded frequency state."""
        if state["frequency_overflow"] or counts.empty:
            return

        values = state["values"]
        for value, count in counts.items():
            if value in values:
                values[value] += int(count)
            elif len(values) < self.MAX_FREQUENCY_VALUES:
                values[value] = int(count)
            else:
                state["frequency_overflow"] = True
                break

    def _update_reservoir_from_sample(self, state: dict, values: pd.Series) -> None:
        """Update the bounded quantile sample using a pandas-compatible RNG."""
        n = len(values)
        if n == 0:
            return

        rng = state["random"]
        reservoir = state["reservoir"]
        seen_before = state["seen_for_reservoir"]

        if len(reservoir) < self.RESERVOIR_SIZE:
            needed = self.RESERVOIR_SIZE - len(reservoir)
            take = min(needed, n)
            indices = rng.choice(n, size=take, replace=False)
            reservoir.extend(values.iloc[indices].tolist())

        state["seen_for_reservoir"] = seen_before + n
        total_seen = state["seen_for_reservoir"]

        if len(reservoir) < self.RESERVOIR_SIZE or n == 0:
            return

        # Approximate bounded replacement. This avoids one Python-level
        # random decision for every incoming numeric cell.
        replace_probability = self.RESERVOIR_SIZE / max(total_seen, 1)
        replace_count = min(
            self.RESERVOIR_SIZE,
            int(round(n * replace_probability)),
        )
        if replace_count <= 0:
            return

        incoming_indices = rng.choice(n, size=replace_count, replace=False)
        reservoir_indices = rng.choice(
            self.RESERVOIR_SIZE,
            size=replace_count,
            replace=False,
        )
        incoming = values.iloc[incoming_indices].tolist()
        for position, value in zip(reservoir_indices, incoming):
            reservoir[int(position)] = value

    def finalize_streaming(self, state: dict, total_rows: int) -> dict:
        missing_percentage = round(state["missing"] / max(total_rows, 1) * 100, 2)
        if not state["is_numeric"]:
            top, frequency = self._counter_mode(state["values"])
            return {
                "count": int(state["count"]),
                "missing": int(state["missing"]),
                "missing_percentage": missing_percentage,
                "unique": int(len(state["values"])),
                "unique_exact": not state["frequency_overflow"],
                "top": None if top is None else str(top),
                "frequency": int(frequency),
                "frequency_exact": not state["frequency_overflow"],
            }

        reservoir = pd.Series(state["reservoir"], dtype="float64")
        q1 = reservoir.quantile(0.25) if not reservoir.empty else np.nan
        q2 = reservoir.quantile(0.50) if not reservoir.empty else np.nan
        q3 = reservoir.quantile(0.75) if not reservoir.empty else np.nan
        mode, _ = self._counter_mode(state["values"])
        variance = state["m2"] / (state["count"] - 1) if state["count"] > 1 else np.nan
        return {
            "count": int(state["count"]),
            "missing": int(state["missing"]),
            "missing_percentage": missing_percentage,
            "mean": self._safe_float(state["mean"] if state["count"] else np.nan),
            "median": self._safe_float(q2),
            "mode": self._safe_float(mode),
            "minimum": self._safe_float(state["minimum"]),
            "maximum": self._safe_float(state["maximum"]),
            "range": self._safe_float(state["maximum"] - state["minimum"] if state["minimum"] is not None else np.nan),
            "variance": self._safe_float(variance),
            "std": self._safe_float(np.sqrt(variance)) if not np.isnan(variance) else None,
            "q1": self._safe_float(q1),
            "q2": self._safe_float(q2),
            "q3": self._safe_float(q3),
            "iqr": self._safe_float(q3 - q1),
            "frequency_exact": not state["frequency_overflow"],
        }

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

    def profile_numeric(self, series: pd.Series) -> dict:
        numeric = pd.to_numeric(series, errors="coerce")
        q1, q2, q3 = numeric.quantile(0.25), numeric.quantile(0.50), numeric.quantile(0.75)
        return {
            "count": int(numeric.count()),
            "missing": int(numeric.isna().sum()),
            "missing_percentage": round(float(numeric.isna().mean() * 100), 2),
            "mean": self._safe_float(numeric.mean()),
            "median": self._safe_float(numeric.median()),
            "mode": self._mode(numeric),
            "minimum": self._safe_float(numeric.min()),
            "maximum": self._safe_float(numeric.max()),
            "range": self._safe_float(numeric.max() - numeric.min()),
            "variance": self._safe_float(numeric.var()),
            "std": self._safe_float(numeric.std()),
            "q1": self._safe_float(q1),
            "q2": self._safe_float(q2),
            "q3": self._safe_float(q3),
            "iqr": self._safe_float(q3 - q1),
        }

    def profile_categorical(self, series: pd.Series) -> dict:
        sample = series.dropna()
        if len(sample) > 5000:
            sample = sample.sample(5000, random_state=42)
        return {
            "count": int(series.count()),
            "missing": int(series.isna().sum()),
            "missing_percentage": round(float(series.isna().mean() * 100), 2),
            "unique": int(sample.nunique()),
            "top": self._top(series),
            "frequency": self._frequency(series),
        }

    def _mode(self, series):
        sample = series.dropna()
        if len(sample) > 5000:
            sample = sample.sample(5000, random_state=42)
        mode = sample.mode()
        return None if mode.empty else self._safe_float(mode.iloc[0])

    def _top(self, series):
        sample = series.dropna()
        if len(sample) > 5000:
            sample = sample.sample(5000, random_state=42)
        mode = sample.mode()
        return None if mode.empty else str(mode.iloc[0])

    def _frequency(self, series):
        sample = series.dropna()
        if len(sample) > 5000:
            sample = sample.sample(5000, random_state=42)
        mode = sample.mode()
        return 0 if mode.empty else int((series == mode.iloc[0]).sum())

    @staticmethod
    def _safe_float(value):
        if pd.isna(value):
            return None
        return float(value)
