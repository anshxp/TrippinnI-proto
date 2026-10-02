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
        validation = (
            "integer"
            if pd.api.types.is_integer_dtype(df[col])
            else "datetime"
            if "time" in col
            else "float"
        )
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
    rows = 40
    df = pd.DataFrame({
        "subject_id": [1] * 20 + [2] * 20,
        "hadm_id": [10] * 10 + [11] * 10 + [20] * 20,
        "admittime": pd.to_datetime(
            ["2024-01-02"] * 20 + ["2024-01-04"] * 20
        ),
        "dischtime": pd.to_datetime(
            ["2024-01-01"] + ["2024-01-03"] * 19
            + ["2024-01-05"] * 20
        ),
    })
    # Inject one hierarchy violation without changing the learned dependency.
    df.loc[39, "subject_id"] = None

    result = RuleValidator().detect({"hosp_admissions": df}, _profile(df))
    issue_types = {issue.issue_type for issue in result.issues}

    assert "temporal" in issue_types
    assert "conformance" in issue_types


def test_quality_detector_produces_bounded_score():
    df = pd.DataFrame({"value": [1.0, 2.0, 3.0, 1000.0]})
    result = QualityDetector().run({"demo": df}, _profile(df))

    assert 0.0 <= result.quality_score <= 100.0
    assert result.summary["overall_score"] == result.quality_score


def test_constraints_are_ml_inferred_without_table_specific_ranges():
    from rule_engine.constraint_inference import ConstraintInferer

    rows = 100
    df = pd.DataFrame({
        "subject_id": [i // 10 for i in range(rows)],
        "hadm_id": [i // 5 for i in range(rows)],
        "admittime": pd.to_datetime(
            ["2024-01-01"] * 50 + ["2024-01-10"] * 50
        ),
        "dischtime": pd.to_datetime(
            ["2024-01-02"] * 50 + ["2024-01-11"] * 50
        ),
        "hospital_expire_flag": [0, 1] * 50,
        "numeric_measure": [10.0 + (i % 20) * 0.5 for i in range(rows)],
    })
    profile = {
        "columns": {
            "subject_id": {"semantic_type": "identifier", "validation_type": "integer"},
            "hadm_id": {"semantic_type": "identifier", "validation_type": "integer"},
            "admittime": {"semantic_type": "datetime", "validation_type": "datetime"},
            "dischtime": {"semantic_type": "datetime", "validation_type": "datetime"},
            "hospital_expire_flag": {"semantic_type": "numeric", "validation_type": "integer"},
            "numeric_measure": {"semantic_type": "numeric", "validation_type": "float"},
        }
    }

    inferred = ConstraintInferer().infer(df, profile)

    temporal = next(
        x for x in inferred["temporal"]
        if x["start"] == "admittime" and x["end"] == "dischtime"
    )
    hierarchy = next(
        x for x in inferred["hierarchy"]
        if x["child"] == "hadm_id" and x["parent"] == "subject_id"
    )
    binary = next(
        x for x in inferred["binary"]
        if x["column"] == "hospital_expire_flag"
    )
    numeric = next(
        x for x in inferred["numeric"]
        if x["column"] == "numeric_measure"
    )

    assert temporal["model"] == "IsolationForest"
    assert hierarchy["model"] == "IsolationForest"
    assert binary["lower"] == 0.0 and binary["upper"] == 1.0
    assert numeric["model"] == "IsolationForest"
    assert numeric["lower"] <= 10.0
    assert numeric["upper"] >= 19.5
