"""Extended, bounded-memory metadata used by preprocessing recommendation models.

Estimates derived from samples are explicitly marked as sampled. This module does
not infer clinical truth, missingness mechanisms, or schema rules without metadata.
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd


DATE_NAME_TOKENS = (
    "date", "time", "timestamp", "datetime", "admit", "discharge",
    "birth", "death", "start", "stop", "recorded",
)
UNIT_NAME_TOKENS = ("unit", "uom", "units", "valueuom")
DATE_FORMAT_PATTERNS = (
    (re.compile(r"^\d{4}-\d{2}-\d{2}$"), "YYYY-MM-DD"),
    (re.compile(r"^\d{4}/\d{2}/\d{2}$"), "YYYY/MM/DD"),
    (re.compile(r"^\d{2}/\d{2}/\d{4}$"), "MM/DD/YYYY_or_DD/MM/YYYY"),
    (re.compile(r"^\d{2}-\d{2}-\d{4}$"), "DD-MM-YYYY_or_MM-DD-YYYY"),
    (re.compile(r"^\d{4}-\d{2}-\d{2}[T ]"), "ISO_datetime"),
)


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if np.isfinite(result) else None


def _string_lengths(values: pd.Series) -> dict:
    strings = values.dropna().astype(str)
    if strings.empty:
        return {
            "string_length_count": 0,
            "string_length_mean": None,
            "string_length_median": None,
            "string_length_std": None,
            "string_length_min": None,
            "string_length_max": None,
        }
    lengths = strings.str.len()
    return {
        "string_length_count": int(len(lengths)),
        "string_length_mean": _safe_float(lengths.mean()),
        "string_length_median": _safe_float(lengths.median()),
        "string_length_std": _safe_float(lengths.std(ddof=1)) if len(lengths) > 1 else 0.0,
        "string_length_min": int(lengths.min()),
        "string_length_max": int(lengths.max()),
    }


def _date_format_consistency(values: pd.Series) -> dict:
    strings = values.dropna().astype(str).str.strip()
    if strings.empty:
        return {
            "date_format_candidate": False,
            "date_format_consistency": None,
            "date_format_pattern_count": None,
            "dominant_date_format": None,
            "date_parse_success_ratio": None,
            "date_parse_failure_ratio": None,
        }

    sample = strings.head(5000)
    name_like_date = False
    # Date candidates are identified from observed values; column-name context
    # is added by profile_series below.
    labels = []
    for value in sample:
        label = "other"
        for pattern, pattern_label in DATE_FORMAT_PATTERNS:
            if pattern.search(value):
                label = pattern_label
                break
        labels.append(label)

    counts = pd.Series(labels).value_counts()
    recognized = int(len(labels) - counts.get("other", 0))
    dominant = str(counts.index[0]) if len(counts) else None
    consistency = float(counts.iloc[0] / len(labels)) if len(counts) else None

    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    parse_success = float(parsed.notna().mean()) if len(parsed) else None
    return {
        "date_format_candidate": bool(recognized > 0 or (parse_success or 0) >= 0.5),
        "date_format_consistency": _safe_float(consistency),
        "date_format_pattern_count": int(sum(label != "other" for label in counts.index)),
        "dominant_date_format": dominant,
        "date_parse_success_ratio": _safe_float(parse_success),
        "date_parse_failure_ratio": _safe_float(1 - parse_success) if parse_success is not None else None,
    }


def profile_extended_series(column_name: str, series: pd.Series, sampled: bool = False) -> dict:
    """Return extra statistical, parseability, format and availability metadata."""
    non_null = series.dropna()
    name = str(column_name).casefold()
    output: dict[str, Any] = {
        "metadata_version": "1.0",
        "statistics_sampled": bool(sampled),
        "sample_size": int(len(series)),
        "non_null_sample_size": int(len(non_null)),
    }

    # Statistical shape and IQR outliers. Outliers are only statistical flags;
    # they are not automatically errors or safe candidates for removal.
    numeric = pd.to_numeric(non_null, errors="coerce")
    numeric_success = int(numeric.notna().sum())
    numeric_candidate = pd.api.types.is_numeric_dtype(series) or (
        len(non_null) > 0 and numeric_success / len(non_null) >= 0.8
    )
    if numeric_candidate and numeric_success:
        numeric = numeric.dropna().astype(float)
        q1, q3 = numeric.quantile([0.25, 0.75])
        iqr = float(q3 - q1)
        low, high = float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)
        outlier_mask = (numeric < low) | (numeric > high)
        output.update({
            "skewness": _safe_float(numeric.skew()) if len(numeric) >= 3 else None,
            "outlier_method": "IQR_1.5x",
            "outlier_lower_bound": _safe_float(low),
            "outlier_upper_bound": _safe_float(high),
            "outlier_count": int(outlier_mask.sum()),
            "outlier_ratio": _safe_float(outlier_mask.mean()),
            "numeric_parse_success_ratio": _safe_float(numeric_success / max(len(non_null), 1)),
            "numeric_parse_failure_ratio": _safe_float(1 - numeric_success / max(len(non_null), 1)),
        })
    else:
        output.update({
            "skewness": None,
            "outlier_method": None,
            "outlier_lower_bound": None,
            "outlier_upper_bound": None,
            "outlier_count": None,
            "outlier_ratio": None,
            "numeric_parse_success_ratio": _safe_float(numeric_success / max(len(non_null), 1)) if len(non_null) else None,
            "numeric_parse_failure_ratio": _safe_float(1 - numeric_success / max(len(non_null), 1)) if len(non_null) else None,
        })

    output.update(_string_lengths(non_null))

    # Only treat date parsing as a meaningful validation signal when the field
    # name or observed values suggest a date/time field.
    date_context = any(token in name for token in DATE_NAME_TOKENS)
    date_meta = _date_format_consistency(non_null)
    if date_context or date_meta["date_format_candidate"]:
        output.update(date_meta)
        output["date_parse_applicable"] = True
    else:
        output.update({
            "date_format_candidate": False,
            "date_format_consistency": None,
            "date_format_pattern_count": None,
            "dominant_date_format": None,
            "date_parse_success_ratio": None,
            "date_parse_failure_ratio": None,
            "date_parse_applicable": False,
        })

    # Unit values can be described only when a unit/UOM field is present.
    # We do not infer measurement units from numeric values or column names.
    if any(token in name for token in UNIT_NAME_TOKENS):
        unit_values = non_null.astype(str).str.strip().str.casefold()
        counts = unit_values.value_counts()
        output.update({
            "unit_metadata_available": True,
            "unit_unique_count": int(unit_values.nunique()),
            "unit_dominant_share": _safe_float(counts.iloc[0] / len(unit_values)) if len(unit_values) else None,
            "unit_consistency": _safe_float(counts.iloc[0] / len(unit_values)) if len(unit_values) else None,
            "unit_values_sample": [str(v) for v in counts.index[:20]],
        })
    else:
        output.update({
            "unit_metadata_available": False,
            "unit_unique_count": None,
            "unit_dominant_share": None,
            "unit_consistency": None,
            "unit_values_sample": [],
        })

    # These require external schema/use-case evidence. Null means unavailable,
    # not zero and not a claim that the column is valid.
    output.update({
        "schema_available": False,
        "schema_violation_count": None,
        "schema_violation_ratio": None,
        "trusted_range_available": False,
        "trusted_range_violation_ratio": None,
        "intended_use": None,
        "missingness_mechanism": "not_inferred",
        "source_similarity_evaluated": False,
        "source_similarity_score": None,
    })
    return output
