"""Train and run the TrippinnI preprocessing-action Random Forest.

This model recommends actions; it never executes transformations. Training requires
reviewed labels with label_status='validated' and a non-empty label_source. The
confidence threshold is a routing heuristic, not a calibrated clinical probability.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TARGET = "target_action"
IDENTITY_OR_LABEL_FIELDS = {
    "table_name", "column_name", "profile_report", "issue_evidence",
    "target_action", "label_status", "label_source", "reviewer_notes",
}
DEFAULT_MIN_CONFIDENCE = 0.65


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = IDENTITY_OR_LABEL_FIELDS | {"label", "reviewer_notes"}
    return [c for c in frame.columns if c not in excluded and not c.endswith("_json")]


def _prepare_features(frame: pd.DataFrame, columns: Iterable[str] | None = None) -> pd.DataFrame:
    selected = list(columns) if columns is not None else _feature_columns(frame)
    X = frame.reindex(columns=selected).copy()
    for col in X.columns:
        if pd.api.types.is_bool_dtype(X[col]):
            X[col] = X[col].astype("object")
        # JSON-derived heterogeneous columns can otherwise break sklearn's dtype inference.
        if X[col].dtype == "object":
            X[col] = X[col].map(lambda value: str(value) if isinstance(value, (dict, list, tuple, set)) else value)
    return X


def _make_pipeline(X: pd.DataFrame, random_state: int = 42) -> Pipeline:
    categorical = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    numeric = [c for c in X.columns if c not in categorical]
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ]), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical))
    prep = ColumnTransformer(transformers=transformers, remainder="drop")
    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )
    return Pipeline([("preprocess", prep), ("classifier", model)])


def train(labels_path: Path, model_path: Path, features_path: Path | None = None, min_rows_per_class: int = 5) -> dict:
    labels = pd.read_csv(labels_path, keep_default_na=True)
    # Annotation templates contain labels and lineage, not model features. Join them
    # to the matching profiling feature rows before fitting.
    if features_path is None:
        features_path = labels_path.with_name("preprocessing_features_unlabeled.csv")
    features_frame = pd.read_csv(features_path, keep_default_na=True)
    join_keys = ["table_name", "column_name", "profile_report", "issue_type"]
    missing_keys = [key for key in join_keys if key not in labels.columns or key not in features_frame.columns]
    if missing_keys:
        raise ValueError(f"Cannot join labels to feature rows; missing keys: {missing_keys}")
    if labels.duplicated(join_keys).any() or features_frame.duplicated(join_keys).any():
        raise ValueError("Duplicate join keys found; each table/column/report/issue row must be unique.")
    label_columns = join_keys + [TARGET, "label_status", "label_source", "reviewer_notes"]
    # The feature exporter carries blank label placeholders for annotation. Drop those
    # before joining so reviewed labels remain authoritative and do not become suffixed.
    feature_label_fields = [TARGET, "label_status", "label_source", "reviewer_notes"]
    features_for_join = features_frame.drop(
        columns=[column for column in feature_label_fields if column in features_frame.columns]
    )
    frame = labels[label_columns].merge(
        features_for_join, on=join_keys, how="inner", validate="one_to_one"
    )
    if frame.empty:
        raise ValueError("No label rows matched feature rows. Generate both files from the same profiling run.")
    required = {TARGET, "label_status", "label_source"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Label file is missing required columns: {sorted(missing)}")
    frame[TARGET] = frame[TARGET].astype("string").str.strip()
    frame["label_status"] = frame["label_status"].astype("string").str.strip().str.casefold()
    frame["label_source"] = frame["label_source"].astype("string").str.strip()
    valid = frame["label_status"].eq("validated") & frame["label_source"].notna() & frame["label_source"].ne("") & frame[TARGET].notna() & frame[TARGET].ne("")
    train_frame = frame.loc[valid].copy()
    if train_frame.empty:
        raise ValueError("No validated labels found. Annotate target_action, set label_status=validated, and record label_source before training.")
    counts = train_frame[TARGET].value_counts()
    too_small = counts[counts < min_rows_per_class]
    if not too_small.empty:
        raise ValueError(f"Insufficient validated examples (need >= {min_rows_per_class} per action class): {too_small.to_dict()}")
    features = _feature_columns(train_frame)
    if not features:
        raise ValueError("No model feature columns were found.")
    X = _prepare_features(train_frame, features)
    y = train_frame[TARGET].astype(str)
    pipeline = _make_pipeline(X)
    pipeline.fit(X, y)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "feature_columns": features, "classes": sorted(y.unique()), "label_count": int(len(y)), "class_counts": counts.to_dict(), "model_version": "1.0"}, model_path)
    return {"model_path": str(model_path), "validated_rows": int(len(y)), "class_counts": counts.to_dict(), "feature_count": len(features), "model_version": "1.0"}


def predict(features_path: Path, model_path: Path, output_path: Path, min_confidence: float = DEFAULT_MIN_CONFIDENCE) -> dict:
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be between 0 and 1")
    artifact = joblib.load(model_path)
    frame = pd.read_csv(features_path, keep_default_na=True)
    X = _prepare_features(frame, artifact["feature_columns"])
    pipeline = artifact["pipeline"]
    probabilities = pipeline.predict_proba(X)
    classes = pipeline.named_steps["classifier"].classes_
    top_indices = probabilities.argmax(axis=1)
    top_probabilities = probabilities[np.arange(len(frame)), top_indices]
    predicted = classes[top_indices]
    result = frame.copy()
    result["recommended_action"] = [label if prob >= min_confidence else "REVIEW_REQUIRED" for label, prob in zip(predicted, top_probabilities)]
    result["model_top_class"] = predicted
    result["model_top_score"] = top_probabilities
    result["model_min_confidence_threshold"] = min_confidence
    result["recommendation_status"] = np.where(top_probabilities >= min_confidence, "candidate_for_rule_validation", "abstain_low_confidence")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return {"output_path": str(output_path), "rows_scored": int(len(result)), "review_required": int((result["recommended_action"] == "REVIEW_REQUIRED").sum()), "min_confidence": min_confidence, "note": "Recommendations are not executed; validate each action against deterministic rules and clinical context."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train_parser = sub.add_parser("train", help="Train only from reviewed, validated action labels")
    train_parser.add_argument("--labels", type=Path, default=Path("outputs/training/preprocessing_labels_template.csv"))
    train_parser.add_argument("--features", type=Path, default=Path("outputs/training/preprocessing_features_unlabeled.csv"))
    train_parser.add_argument("--model", type=Path, default=Path("outputs/models/preprocessing_random_forest.joblib"))
    train_parser.add_argument("--min-rows-per-class", type=int, default=5)
    predict_parser = sub.add_parser("predict", help="Recommend actions for a feature CSV")
    predict_parser.add_argument("--features", type=Path, default=Path("outputs/training/preprocessing_features_unlabeled.csv"))
    predict_parser.add_argument("--model", type=Path, default=Path("outputs/models/preprocessing_random_forest.joblib"))
    predict_parser.add_argument("--output", type=Path, default=Path("outputs/training/preprocessing_recommendations.csv"))
    predict_parser.add_argument("--min-confidence", type=float, default=DEFAULT_MIN_CONFIDENCE)
    args = parser.parse_args()
    if args.command == "train":
        result = train(args.labels, args.model, args.features, args.min_rows_per_class)
    else:
        result = predict(args.features, args.model, args.output, args.min_confidence)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
