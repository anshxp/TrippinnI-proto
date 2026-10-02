import pandas as pd

from detectors.datatype_detector import DatatypeDetector
from detectors.duplicate_detector import DuplicateDetector
from detectors.missing_detector import MissingDetector
from quality.detector import QualityDetector
from rule_engine.rule_validator import RuleValidator


def _profile(df):
    columns = {}
    for col in df.columns:
        semantic = "identifier" if col.endswith("_id") else "numeric"
        validation = "integer" if pd.api.types.is_integer_dtype(df[col]) else "datetime" if "time" in col else "float"
        columns[col] = {
            "semantic_type": semantic,
            "validation_type": validation,
            "null_percentage": float(df[col].isna().mean() * 100),
            "all_null": bool(df[col].isna().all()),
        }
    return {"columns": columns, "keys": {"primary_keys": []}}


def test_missing_duplicate_and_datatype_detectors():
    df = pd.DataFrame({
        "subject_id": [1, 1, 2],
        "value": [10.0, None, 12.0],
    })
    profile = _profile(df)

    missing = MissingDetector().detect({"demo": df}, profile)
    duplicate = DuplicateDetector().detect({"demo": df}, profile)
    datatype = DatatypeDetector().detect({"demo": df}, profile)

    assert missing.issue_count == 1
    assert duplicate.issue_count == 2
    assert datatype.issue_count == 0


def test_mimic_rule_validator_flags_temporal_and_hierarchy_errors():
    df = pd.DataFrame({
        "subject_id": [1, None],
        "hadm_id": [10, 11],
        "admittime": pd.to_datetime(["2024-01-02", "2024-01-04"]),
        "dischtime": pd.to_datetime(["2024-01-01", "2024-01-05"]),
    })
    result = RuleValidator().detect({"hosp_admissions": df}, _profile(df))
    issue_types = {issue.issue_type for issue in result.issues}

    assert "temporal" in issue_types
    assert "conformance" in issue_types


def test_quality_detector_produces_bounded_score():
    df = pd.DataFrame({"value": [1.0, 2.0, 3.0, 1000.0]})
    result = QualityDetector().run({"demo": df}, _profile(df))

    assert 0.0 <= result.quality_score <= 100.0
    assert result.summary["overall_score"] == result.quality_score