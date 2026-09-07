from datetime import datetime, timezone


class ReportBuilder:
    """Combine profiler outputs into a versioned profiling report."""

    REPORT_VERSION = "1.0"

    def build(self, dataset, columns, memory, keys):
        column_values = list(columns.values())
        return {
            "report_version": self.REPORT_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "dataset": dataset,
            "columns": columns,
            "memory": memory,
            "keys": keys,
            "summary": {
                "column_count": len(column_values),
                "missing_columns": sum(1 for item in column_values if item.get("null_count", 0) > 0),
                "all_null_columns": sum(1 for item in column_values if item.get("all_null", False)),
                "high_cardinality_columns": sum(
                    1 for item in column_values
                    if item.get("cardinality_ratio", 0) >= 0.90
                ),
                "candidate_primary_key_count": len(keys.get("primary_keys", [])),
                "candidate_foreign_key_count": len(keys.get("foreign_keys", [])),
            },
        }
