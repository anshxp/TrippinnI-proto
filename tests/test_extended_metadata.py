import json

import pandas as pd

from profiling.extended_metadata import profile_extended_series
from ml.generate_preprocessing_dataset import build_feature_rows, write_outputs


def test_extended_numeric_metadata_flags_outliers_without_claiming_clinical_error():
    series = pd.Series([1, 2, 2, 3, 4, 100], dtype="float64")
    metadata = profile_extended_series("lab_value", series)
    assert metadata["skewness"] is not None
    assert metadata["outlier_ratio"] > 0
    assert metadata["outlier_method"] == "IQR_1.5x"
    assert metadata["schema_violation_ratio"] is None
    assert metadata["trusted_range_violation_ratio"] is None


def test_extended_string_metadata_reports_length_and_date_parse_failure():
    series = pd.Series(["2024-01-01", "2024-02-01", "not-a-date"], dtype="object")
    metadata = profile_extended_series("event_time", series)
    assert metadata["string_length_applicable"] is True
    assert metadata["string_length_max"] > 0
    assert metadata["date_parse_applicable"] is True
    assert metadata["date_parse_failure_ratio"] > 0


def test_unit_metadata_is_descriptive_and_does_not_claim_linked_consistency():
    series = pd.Series(["mg/dL", "mmol/L", "mg/dL"], dtype="object")
    metadata = profile_extended_series("result_unit", series)
    assert metadata["unit_metadata_available"] is True
    assert metadata["unit_unique_count"] == 2
    assert metadata["unit_consistency_assessment"] == "requires_measurement_linkage"


def test_training_export_keeps_labels_unset(tmp_path):
    reports_dir = tmp_path / "profiles"
    reports_dir.mkdir()
    report = {
        "dataset": {"table_name": "example", "rows": 3, "columns": 1, "duplicate_rows": 0},
        "columns": {
            "lab_value": {
                "name": "lab_value",
                "pandas_dtype": "float64",
                "semantic_type": "numeric",
                "validation_type": "float",
                "null_count": 1,
                "null_percentage": 33.3,
                "all_null": False,
                "cardinality_ratio": 0.66,
                "statistics": {"mean": 2.0},
                "extended": {
                    "outlier_ratio": 0.1,
                    "numeric_parse_applicable": False,
                    "date_parse_applicable": False,
                    "schema_violation_ratio": None,
                    "trusted_range_violation_ratio": None,
                },
            }
        },
        "keys": {"primary_keys": [], "foreign_keys": []},
    }
    (reports_dir / "example_profile.json").write_text(json.dumps(report), encoding="utf-8")

    rows = build_feature_rows(reports_dir)
    assert rows
    assert all(row["target_action"] == "" for row in rows)
    assert {row["issue_type"] for row in rows} >= {"missing_values", "statistical_outliers"}

    feature_path, label_path, schema_path = write_outputs(rows, tmp_path / "out")
    assert feature_path.exists()
    assert label_path.exists()
    assert schema_path.exists()
    labels = pd.read_csv(label_path)
    assert labels["target_action"].isna().all()
