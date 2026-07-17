class ReportBuilder:

    """
    Combines all profiler outputs
    into one standard metadata object.
    """

    def build(

        self,

        dataset,

        columns,

        memory,

        keys

    ):

        return {

            "dataset": dataset,

            "columns": columns,

            "memory": memory,

            "keys": keys

        }