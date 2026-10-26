import pandas as pd

from profiling.column_profiler import ColumnProfiler
from profiling.key_detector import KeyDetector


def test_streaming_detects_late_non_numeric_clinical_values():
    first = pd.DataFrame({"lab_result": [float(i) for i in range(100)]})
    second = pd.DataFrame({"lab_result": [">100,000 CFU/mL"] * 20})

    result = ColumnProfiler().profile_chunks([first, second])["lab_result"]

    assert result["pandas_dtype"] == "mixed"
    assert result["semantic_type"] == "text"
    assert result["validation_type"] == "string"
    assert result["unique_values"] == 101
    assert result["statistics"]["numeric_count"] == 100
    assert result["statistics"]["numeric_parse_failures"] == 20
    assert result["statistics"]["actual_nulls"] == 0
    assert result["statistics"]["count"] == 120
    assert result["statistics"]["missing"] == 0
    assert result["statistics"]["missing_percentage"] == 0.0
    assert result["statistics"]["summary_scope"] == "numeric_values_only"


def test_streaming_column_statistics_match_full_frame_for_numeric_data():
    first = pd.DataFrame({"value": [1.0, 2.0, 3.0]})
    second = pd.DataFrame({"value": [4.0, 5.0, None]})
    combined = pd.concat([first, second], ignore_index=True)

    full = ColumnProfiler().profile(combined)["value"]
    streamed = ColumnProfiler().profile_chunks([first, second])["value"]

    assert streamed["unique_values"] == full["unique_values"]
    assert streamed["null_count"] == full["null_count"]
    assert streamed["statistics"]["count"] == full["statistics"]["count"]
    assert streamed["statistics"]["missing"] == full["statistics"]["missing"]
    assert streamed["statistics"]["mean"] == full["statistics"]["mean"]


def test_streaming_semantic_sample_is_not_limited_to_first_chunk():
    first = pd.DataFrame({"status": [f"unique_{i}" for i in range(150)]})
    second = pd.DataFrame({"status": ["positive"] * 1000})

    result = ColumnProfiler().profile_chunks([first, second])["status"]

    assert result["semantic_type"] in {"categorical", "text"}
    assert result["unique_values"] == 151


def test_key_detector_handles_duplicates_across_chunk_boundaries():
    first = pd.DataFrame({"row_id": [1, 2, 3], "patient_id": [10, 11, 12]})
    second = pd.DataFrame({"row_id": [4, 5, 2], "patient_id": [13, 14, 15]})

    result = KeyDetector().detect_chunks([first, second])

    assert "row_id" not in result["primary_keys"]
    assert "patient_id" in result["foreign_keys"]
    assert "patient_id" in result["primary_keys"]


def test_key_detector_excludes_columns_with_nulls():
    chunks = [
        pd.DataFrame({"candidate_id": [1, 2]}),
        pd.DataFrame({"candidate_id": [None, 4]}),
    ]

    result = KeyDetector().detect_chunks(chunks)

    assert "candidate_id" not in result["primary_keys"]
