import pandas as pd

from preprocessing.executor import RemediationExecutor
from preprocessing.policy import RemediationPolicy


def test_remediation_is_auditable_and_protects_identifiers():
    df = pd.DataFrame({
        "subject_id": [1, 2, 2],
        "value": ["10", None, "10"],
        "label": [" A ", None, " A "],
    })
    profile = {
        "columns": {
            "subject_id": {"semantic_type": "identifier", "dtype": "int64"},
            "value": {"semantic_type": "numeric", "expected_dtype": "float"},
            "label": {"semantic_type": "categorical", "expected_dtype": "string"},
        }
    }

    policy = RemediationPolicy()
    result = RemediationExecutor(policy).run({"demo": df}, {"demo": profile})
    refined = result.dataset["demo"]

    assert len(refined) == 2
    assert refined["subject_id"].isna().sum() == 0
    assert refined["value"].isna().sum() == 0
    assert "UNKNOWN" not in refined["label"].tolist()
    assert result.logs["demo"].count >= 3


def test_outliers_are_retained_by_baseline_policy():
    df = pd.DataFrame({"value": [1.0, 2.0, 1000.0]})
    profile = {"columns": {"value": {"semantic_type": "numeric", "expected_dtype": "float"}}}

    result = RemediationExecutor().run({"demo": df}, {"demo": profile})
    assert result.dataset["demo"]["value"].tolist() == [1.0, 2.0, 1000.0]
