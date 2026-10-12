import pandas as pd
from preprocessing.executor import RemediationExecutor
from preprocessing.policy import RemediationPolicy

def test_remediation_disabled_by_default_preserves_input():
    df = pd.DataFrame({"value": [1.0, None]})
    result = RemediationExecutor().run({"demo": df}, {"demo": {"columns": {}}})
    assert result.summary["enabled"] is False
    assert result.dataset["demo"].equals(df)

def test_enabled_remediation_audits_imputation_and_protects_identifiers():
    df = pd.DataFrame({"subject_id": [1, 2], "value": [10.0, None], "label": [" A ", None]})
    profile = {"columns": {
        "subject_id": {"semantic_type": "identifier", "dtype": "int64"},
        "value": {"semantic_type": "numeric", "expected_dtype": "float"},
        "label": {"semantic_type": "categorical", "expected_dtype": "string"},
    }}
    policy = RemediationPolicy(enabled=True, impute_numeric=True, impute_categorical=True)
    result = RemediationExecutor(policy).run({"demo": df}, {"demo": profile})
    refined = result.dataset["demo"]
    assert refined["subject_id"].isna().sum() == 0
    assert refined["value"].isna().sum() == 0
    assert "UNKNOWN" in refined["label"].tolist()
    assert result.logs["demo"].count >= 2
    assert result.logs["demo"].to_dict()["records"]

def test_duplicate_removal_is_opt_in():
    df = pd.DataFrame({"event_time": ["2026-01-01", "2026-01-01"], "value": [5, 5]})
    policy = RemediationPolicy(enabled=True, remove_exact_duplicates=False)
    result = RemediationExecutor(policy).run({"demo": df}, {"demo": {}})
    assert len(result.dataset["demo"]) == 2
