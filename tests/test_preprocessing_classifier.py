import pandas as pd
import pytest

from ml.preprocessing_classifier import train


def test_training_refuses_unvalidated_labels_and_joins_feature_table(tmp_path):
    keys = {
        "table_name": ["labs"] * 10,
        "column_name": [f"value_{i}" for i in range(10)],
        "profile_report": ["labs_profile.json"] * 10,
        "issue_type": ["missing_values"] * 10,
    }
    labels = pd.DataFrame({
        **keys,
        "target_action": ["impute_median"] * 5 + ["review_only"] * 5,
        "label_status": ["validated"] * 10,
        "label_source": ["controlled_corruption_test"] * 10,
        "reviewer_notes": [""] * 10,
    })
    features = pd.DataFrame({
        **keys,
        "null_percentage": [10, 20, 30, 40, 50, 0, 0, 0, 0, 0],
        "semantic_type": ["numeric"] * 10,
        "candidate_primary_key": [False] * 10,
        "target_action": [""] * 10,
        "label_status": ["unlabeled"] * 10,
        "label_source": [""] * 10,
        "issue_evidence": ["evidence"] * 10,
    })
    labels_path = tmp_path / "labels.csv"
    features_path = tmp_path / "features.csv"
    model_path = tmp_path / "model.joblib"
    labels.to_csv(labels_path, index=False)
    features.to_csv(features_path, index=False)

    result = train(labels_path, model_path, features_path, min_rows_per_class=5)
    assert model_path.exists()
    assert result["validated_rows"] == 10
    assert result["feature_count"] > 0


def test_training_rejects_unvalidated_label_rows(tmp_path):
    labels = pd.DataFrame({
        "table_name": ["labs"],
        "column_name": ["value"],
        "profile_report": ["labs_profile.json"],
        "issue_type": ["missing_values"],
        "target_action": ["impute_median"],
        "label_status": ["needs_review"],
        "label_source": [""],
        "reviewer_notes": [""],
    })
    features = pd.DataFrame({
        "table_name": ["labs"],
        "column_name": ["value"],
        "profile_report": ["labs_profile.json"],
        "issue_type": ["missing_values"],
        "null_percentage": [20],
    })
    labels_path = tmp_path / "labels.csv"
    features_path = tmp_path / "features.csv"
    labels.to_csv(labels_path, index=False)
    features.to_csv(features_path, index=False)
    with pytest.raises(ValueError, match="No validated labels"):
        train(labels_path, tmp_path / "model.joblib", features_path)
