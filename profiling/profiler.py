from __future__ import annotations

import random

import pandas as pd

from explainability import report
from profiling.dataset_profiler import DatasetProfiler
from profiling.column_profiler import ColumnProfiler
from profiling.memory_profiler import MemoryProfiler
from profiling.key_detector import KeyDetector
from profiling.report_builder import ReportBuilder
from profiling.report_writer import ReportWriter

class DatasetProfilerEngine:

    """
    Coordinates the entire profiling pipeline.
    """

    def __init__(self):

        self.dataset = DatasetProfiler()

        self.columns = ColumnProfiler()

        self.memory = MemoryProfiler()

        self.keys = KeyDetector()

        self.builder = ReportBuilder()
        
        self.writer = ReportWriter()

    #################################################################

    def profile(

        self,

        table_name,

        dataframe

    ):

        dataset_profile = self.dataset.profile(

            table_name,

            dataframe

        )

        column_profiles = self.columns.profile(

            dataframe

        )

        memory_profile = self.memory.profile(

            dataframe

        )

        key_profile = self.keys.detect(

            dataframe

        )

        report = self.builder.build(

            dataset_profile,

            column_profiles,

            memory_profile,

            key_profile

        )

        self.writer.save(
            table_name,
            report
        )
        return report

    def profile_chunks(
        self,
        table_name,
        chunks,
        sample_size: int | None = None,
        sample_seed: int = 42,
    ):
        """Profile a chunk iterator in one pass without concatenating it.

        If sample_size is given, also builds a bounded, uniformly random
        subsample of rows (Algorithm R reservoir sampling, extended
        from statistics.py's per-value version to whole rows) in the
        same pass - so detection gets a real dataset-representative
        sample without a second read of the source file.

        Returns (report, sample_dataframe_or_None). sample_dataframe is
        None if sample_size is None or the table is empty.

        Note: chunk.iterrows() is used to build the reservoir, which is
        slower than vectorized pandas ops (rough estimate: single-digit
        seconds per million rows on typical hardware) but only runs
        once per table regardless of dataset size. If this becomes a
        bottleneck, it can be replaced with vectorized per-chunk
        random-index sampling - not needed yet at prototype scale.
        """

        dataset_state = self.dataset.start_streaming(table_name)
        column_state = self.columns.start_streaming()
        memory_state = self.memory.start_streaming()
        key_state = self.keys.start_streaming()

        reservoir: list = []
        rows_seen = 0
        rng = random.Random(sample_seed)

        for chunk in chunks:
            self.dataset.update_streaming(dataset_state, chunk)
            self.columns.update_streaming(column_state, chunk)
            self.memory.update_streaming(memory_state, chunk)
            self.keys.update_streaming(key_state, chunk)

            if sample_size is not None:
                rows_seen = self._update_reservoir(
                    reservoir, rows_seen, chunk, sample_size, rng
                )

        report = self.builder.build(
            self.dataset.finalize_streaming(dataset_state),
            self.columns.finalize_streaming(column_state),
            self.memory.finalize_streaming(memory_state),
            self.keys.finalize_streaming(key_state),
        )
        self.writer.save(table_name, report)

        # iterrows() upcasts each row to a common dtype (often 'object'
        # for mixed-type rows), so the reconstructed sample may not have
        # the same per-column dtypes as the source chunks. Callers
        # should re-run dtype downcasting on the sample - it's a small
        # DataFrame at this point, so that's cheap.
        sample_dataframe = (
            pd.DataFrame(reservoir).reset_index(drop=True)
            if reservoir
            else None
        )

        return report, sample_dataframe

    @staticmethod
    def _update_reservoir(reservoir, rows_seen, chunk, size, rng):
        """
        Reservoir sampling, vectorized per chunk instead of per row.

        True Algorithm R decides row-by-row: each row's inclusion
        probability is sample_size / (its global index + 1). At
        MIMIC-IV scale (chartevents/labevents run into the hundreds of
        millions of rows), a Python-level per-row loop is too slow -
        plausibly tens of minutes on the largest tables. This instead:

          1. Fills the reservoir directly from the first rows seen.
          2. Once full, computes - per chunk - roughly how many of
             this chunk's rows should replace existing entries (using
             the chunk's average inclusion probability, size divided
             by rows seen so far), picks that many via ONE vectorized
             .sample() call, and swaps them into random reservoir
             slots.

        This is an approximation, not textbook-exact Algorithm R: it
        treats every row within a chunk as having the same inclusion
        probability (the chunk's average) rather than each row's exact
        position-dependent one. The error from that is negligible once
        rows_seen is large relative to chunk size - true for
        essentially the whole run except the first chunk or two - and
        it's the difference between building the sample in seconds
        versus potentially tens of minutes on chartevents-scale tables.
        """

        if len(reservoir) < size:
            needed = size - len(reservoir)
            take = min(needed, len(chunk))
            reservoir.extend(row for _, row in chunk.iloc[:take].iterrows())
            rows_seen += take
            chunk = chunk.iloc[take:]

        remaining = len(chunk)
        if remaining == 0:
            return rows_seen

        inclusion_prob = size / max(rows_seen + remaining, 1)
        num_replacements = min(
            remaining,
            size,
            round(inclusion_prob * remaining),
        )

        if num_replacements > 0:
            selected = chunk.sample(
                n=num_replacements,
                random_state=rng.randint(0, 2**31 - 1),
            )
            slots = rng.sample(range(size), num_replacements)
            for slot, (_, row) in zip(slots, selected.iterrows()):
                reservoir[slot] = row

        rows_seen += remaining
        return rows_seen