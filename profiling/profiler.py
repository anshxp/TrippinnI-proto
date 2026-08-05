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

    def profile_chunks(self, table_name, chunks):
        """Profile a chunk iterator in one pass without concatenating it."""

        dataset_state = self.dataset.start_streaming(table_name)
        column_state = self.columns.start_streaming()
        memory_state = self.memory.start_streaming()
        key_state = self.keys.start_streaming()

        for chunk in chunks:
            self.dataset.update_streaming(dataset_state, chunk)
            self.columns.update_streaming(column_state, chunk)
            self.memory.update_streaming(memory_state, chunk)
            self.keys.update_streaming(key_state, chunk)

        report = self.builder.build(
            self.dataset.finalize_streaming(dataset_state),
            self.columns.finalize_streaming(column_state),
            self.memory.finalize_streaming(memory_state),
            self.keys.finalize_streaming(key_state),
        )
        self.writer.save(table_name, report)
        return report
