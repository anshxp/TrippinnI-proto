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
    missing_issue = missing.issues[0]
    assert missing_issue.row_index == -1
    assert missing_issue.metadata["missing_cells"] == 1
    assert missing_issue.metadata["aggregation"] == "column"
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
            ["2024-01-01"]
            + [f"2024-01-{3 + (i % 5):02d}" for i in range(19)]
            + [f"2024-01-{5 + (i % 5):02d}" for i in range(20)]
        ),
    })
    # Inject one hierarchy violation without changing the learned dependency.
    df.loc[39, "subject_id"] = None

    result = RuleValidator().detect({"hosp_admissions": df}, _profile(df))
    issue_types = {issue.issue_type for issue in result.issues}

    assert "temporal" in issue_types
    assert "conformance" in issue_types


def test_quality_result_deduplicates_same_finding_across_detectors():
    from models.detector_result import DetectorResult
    from models.issue import Issue
    from models.quality_result import QualityResult

    issue_a = Issue(
        table="demo", row_index=4, column="subject_id",
        issue_type="missing", severity="MEDIUM", detector="MissingDetector",
    )
    issue_b = Issue(
        table="demo", row_index=4, column="subject_id",
        issue_type="missing", severity="HIGH", detector="RuleValidator",
    )
    result = QualityResult.from_detector_results([
        DetectorResult(detector_name="MissingDetector", issues=[issue_a]),
        DetectorResult(detector_name="RuleValidator", issues=[issue_b]),
    ])

    assert result.total_issues == 1
    assert result.issues[0].severity == "HIGH"
    assert result.issues[0].metadata["source_detectors"] == [
        "MissingDetector", "RuleValidator"
    ]


def test_quality_detector_scores_missingness_by_affected_cells():
    df = pd.DataFrame({"value": [1.0, None, None, 4.0]})
    result = QualityDetector().run({"demo": df}, _profile(df))

    assert result.summary["category_scores"]["missing"]["count"] == 2
    assert result.summary["category_scores"]["missing"]["denominator"] == 4


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
            [f"2024-01-{2 + (i % 10):02d}" for i in range(50)]
            + [f"2024-01-{11 + (i % 10):02d}" for i in range(50)]
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


def test_duplicate_detector_fuzzy_matches_similar_text():
    df = pd.DataFrame({
        "description": [
            "acute kidney injury",
            "acute kidney injuri",
            "pneumonia",
        ],
        "value": [1, 1, 2],
    })
    profile = {
        "columns": {
            "description": {"semantic_type": "text"},
            "value": {"semantic_type": "numeric"},
        },
        "keys": {"primary_keys": []},
    }

    result = DuplicateDetector().detect({"demo": df}, profile)

    fuzzy = [
        issue for issue in result.issues
        if issue.metadata.get("method") == "fuzzy_match"
    ]
    assert len(fuzzy) == 2
    assert {issue.row_index for issue in fuzzy} == {0, 1}
    assert all(issue.severity == "MEDIUM" for issue in fuzzy)
    assert all(issue.confidence >= 0.92 for issue in fuzzy)
    assert all(issue.metadata["similarity_score"] >= 92.0 for issue in fuzzy)


def test_duplicate_detector_does_not_fuzzy_match_identifiers():
    df = pd.DataFrame({
        "subject_id": ["10001", "10002"],
        "note": ["patient record", "patient record"],
    })
    profile = {
        "columns": {
            "subject_id": {"semantic_type": "identifier"},
            "note": {"semantic_type": "text"},
        },
        "keys": {"primary_keys": []},
    }

    result = DuplicateDetector().detect({"demo": df}, profile)

    fuzzy = [
        issue for issue in result.issues
        if issue.metadata.get("method") == "fuzzy_match"
    ]
    assert all(issue.column != "subject_id" for issue in fuzzy)

def test_advanced_quality_detects_representation_and_fairness():
    from detectors.advanced_quality_detector import AdvancedQualityDetector

    rows = 100
    df = pd.DataFrame({
        "gender": ["A"] * 90 + ["B"] * 10,
        "lab_value": list(range(90)) + [None] * 10,
    })
    result = AdvancedQualityDetector().detect({"demo": df}, _profile(df))

    assert any(issue.issue_type == "bias" for issue in result.issues)
    assert any(issue.issue_type == "fairness" for issue in result.issues)


def test_advanced_quality_detects_interoperability_and_drift():
    from detectors.advanced_quality_detector import AdvancedQualityDetector

    rows = 100
    dates = pd.date_range("2024-01-01", periods=rows, freq="D")
    df = pd.DataFrame({
        "charttime": dates,
        "heart_rate": [70] * 50 + [150] * 50,
        "pulse": [70] * 100,
        "valueuom": ["mg/dL"] * 50 + ["mmol/L"] * 50,
    })
    result = AdvancedQualityDetector().detect({"demo": df}, _profile(df))

    types = {issue.issue_type for issue in result.issues}
    assert "interoperability" in types
    assert "drift" in types


def test_distribution_shift_requires_explicit_reference_dataset():
    from detectors.advanced_quality_detector import AdvancedQualityDetector

    reference = pd.DataFrame({"value": [1.0] * 50 + [2.0] * 50})
    current = pd.DataFrame({"value": [100.0] * 50 + [101.0] * 50})

    result = AdvancedQualityDetector().detect_distribution_shift(
        {"demo": current},
        {"demo": reference},
    )

    assert any(issue.issue_type == "distribution_shift" for issue in result.issues)
    assert result.statistics["demo"]["reference_rows"] == 100


def test_advanced_quality_robustness_metric_is_reported():
    from detectors.advanced_quality_detector import AdvancedQualityDetector

    df = pd.DataFrame({
        "value": list(range(100)),
        "other": [None] * 20 + list(range(80)),
    })
    result = AdvancedQualityDetector().detect({"demo": df}, _profile(df))

    assert result.statistics["demo"]["robustness"]["available"] is True
    assert "missingness_range" in result.statistics["demo"]["robustness"]
