from explainability import report
from profiling.dataset_profiler import DatasetProfiler
from profiling.column_profiler import ColumnProfiler
from profiling.memory_profiler import MemoryProfiler
from profiling.key_detector import KeyDetector
from profiling.report_builder import ReportBuilder


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