"""Build an auditable feature table from TrippinnI profiling reports.

This script intentionally does not invent preprocessing labels. It emits an
unlabelled feature dataset and a separate annotation template. A Random Forest
must not be trained until target actions are validated by policy, controlled
corruption experiments, or qualified review.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


IDENTITY_FIELDS = {
    "table_name",
    "column_name",
    "profile_report",
    "semantic_type",
    "pandas_dtype",
    "validation_type",
    "candidate_primary_key",
    "candidate_foreign_key",
    "semantic_role",
    "target_action",
    "label_status",
    "label_source",
}


def _flatten(prefix: str, value: Any, output: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            _flatten(f"{prefix}_{key}" if prefix else str(key), nested, output)
    elif isinstance(value, (list, tuple, set)):
        # Keep lists out of the tabular feature matrix. Store stable counts and
        # JSON summaries for audit; these can be feature-engineered later.
        output[f"{prefix}_count"] = len(value)
        output[f"{prefix}_json"] = json.dumps(
            list(value), ensure_ascii=False, sort_keys=True, default=str
        )
    elif value is None:
        output[prefix] = None
    elif isinstance(value, (str, bool, int, float)):
        output[prefix] = value
    else:
        output[prefix] = str(value)


def _load_json(path: Path) -> dict | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def build_feature_rows(reports_dir: Path, context_path: Path | None = None) -> list[dict]:
    reports = sorted(reports_dir.glob("*_profile.json"))
    if not reports:
        raise FileNotFoundError(
            f"No *_profile.json reports found in {reports_dir}. Run the profiler first."
        )

    context = _load_json(context_path) if context_path and context_path.exists() else {}
    context_tables = (context or {}).get("tables", {})
    rows: list[dict] = []

    for report_path in reports:
        report = _load_json(report_path)
        if not report:
            continue
        dataset = report.get("dataset", {})
        columns = report.get("columns", {})
        keys = report.get("keys", {})
        primary_keys = set(keys.get("primary_keys", []))
        foreign_keys = set(keys.get("foreign_keys", []))
        table_name = str(dataset.get("table_name") or report_path.name.removesuffix("_profile"))
        table_context = context_tables.get(table_name, {})
        context_columns = table_context.get("columns", {})

        dataset_flat: dict[str, Any] = {}
        _flatten("table", {
            key: value for key, value in dataset.items()
            if key not in {"table_name", "empty_columns", "constant_columns", "shape"}
        }, dataset_flat)

        for column_name, column in columns.items():
            if not isinstance(column, dict):
                continue
            row: dict[str, Any] = {
                "table_name": table_name,
                "column_name": str(column_name),
                "profile_report": report_path.name,
                "semantic_type": column.get("semantic_type"),
                "pandas_dtype": column.get("pandas_dtype"),
                "validation_type": column.get("validation_type"),
                "candidate_primary_key": column_name in primary_keys,
                "candidate_foreign_key": column_name in foreign_keys,
                "target_action": "",
                "label_status": "unlabeled",
                "label_source": "",
            }

            # Column fields plus nested statistics and newly added extended
            # metadata are retained in the feature export.
            for key, value in column.items():
                if key in {"name", "semantic_type", "pandas_dtype", "validation_type"}:
                    continue
                _flatten(key, value, row)

            # Table-level context is repeated per column so models can use
            # dataset size/completeness without needing raw table identifiers.
            row.update(dataset_flat)

            semantic = context_columns.get(str(column_name), {})
            row["semantic_role"] = semantic.get("role")
            row["semantic_context_confidence"] = semantic.get("confidence")
            roles = semantic.get("roles", [])
            row["semantic_roles_count"] = len(roles) if isinstance(roles, list) else None
            row["semantic_roles_json"] = json.dumps(roles, ensure_ascii=False, default=str)

            # Features requiring external evidence remain explicit nulls rather
            # than being silently filled with zero.
            row.setdefault("extended_schema_violation_ratio", None)
            row.setdefault("extended_trusted_range_violation_ratio", None)
            row.setdefault("extended_intended_use", None)
            row.setdefault("extended_source_similarity_score", None)
            rows.append(row)

    if not rows:
        raise ValueError(f"Reports were found in {reports_dir}, but no column profiles were usable.")
    return rows


def write_outputs(rows: list[dict], output_dir: Path) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    features_path = output_dir / "preprocessing_features_unlabeled.csv"
    labels_path = output_dir / "preprocessing_labels_template.csv"
    schema_path = output_dir / "preprocessing_feature_schema.json"

    frame = pd.DataFrame(rows)
    frame.to_csv(features_path, index=False, quoting=csv.QUOTE_MINIMAL)

    annotation_columns = [
        "table_name", "column_name", "profile_report",
        "issue_type", "target_action", "label_status", "label_source",
        "reviewer_notes",
    ]
    annotations = frame[["table_name", "column_name", "profile_report"]].copy()
    annotations["issue_type"] = ""
    annotations["target_action"] = ""
    annotations["label_status"] = "needs_review"
    annotations["label_source"] = ""
    annotations["reviewer_notes"] = ""
    annotations.to_csv(labels_path, index=False, columns=annotation_columns)

    excluded = sorted(IDENTITY_FIELDS)
    feature_columns = [column for column in frame.columns if column not in excluded]
    schema = {
        "schema_version": "1.0",
        "source": "TrippinnI profiling JSON reports",
        "row_grain": "one row per profiled column",
        "row_count": int(len(frame)),
        "feature_columns": feature_columns,
        "excluded_from_model_features": excluded,
        "target_column": "target_action",
        "label_status": "unlabelled; do not train until target_action values are validated",
        "null_handling": "Null means unavailable/not applicable; do not blindly replace all nulls with zero.",
        "notes": [
            "Some extended statistics are estimates from a bounded sample.",
            "Table and column names are retained for lineage but excluded from default model features.",
            "Categorical features require encoding in a scikit-learn preprocessing pipeline.",
            "Schema violations and intended use require external schema or workflow metadata.",
            "The feature table does not itself constitute labelled training data.",
        ],
    }
    schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    return features_path, labels_path, schema_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", type=Path, default=Path("outputs/reports/profiling"))
    parser.add_argument(
        "--context-path",
        type=Path,
        default=Path("outputs/reports/context/healthcare_context.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/training"))
    args = parser.parse_args()

    context_path = args.context_path if args.context_path.exists() else None
    rows = build_feature_rows(args.reports_dir, context_path)
    feature_path, label_path, schema_path = write_outputs(rows, args.output_dir)
    print(f"Profiled-column rows exported: {len(rows)}")
    print(f"Unlabelled features: {feature_path}")
    print(f"Label annotation template: {label_path}")
    print(f"Feature schema: {schema_path}")
    print("No model was trained: target actions still require validated labels.")


if __name__ == "__main__":
    main()
