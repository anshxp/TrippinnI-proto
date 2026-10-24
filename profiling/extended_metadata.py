"""Extended, bounded-memory metadata used by preprocessing recommendation models.

Sample-derived metrics are explicitly marked. This module does not infer clinical
truth, missingness mechanisms, or schema rules without external evidence.
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


def _string_lengths(values: pd.Series, applicable: bool) -> dict:
    if not applicable:
        return {
            "string_length_applicable": False,
            "string_length_count": None,
            "string_length_mean": None,
            "string_length_median": None,
            "string_length_std": None,
            "string_length_min": None,
            "string_length_max": None,
        }
    strings = values.dropna().astype(str)
    if strings.empty:
        return {
            "string_length_applicable": True,
            "string_length_count": 0,
            "string_length_mean": None,
            "string_length_median": None,
            "string_length_std": None,
            "string_length_min": None,
            "string_length_max": None,
        }
    lengths = strings.str.len()
    return {
        "string_length_applicable": True,
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
    labels = []
    for value in sample:
        label = "other"
        for pattern, pattern_label in DATE_FORMAT_PATTERNS:
            if pattern.search(value):
                label = pattern_label
                break
        labels.append(label)

    counts = pd.Series(labels).value_counts()
    recognized_counts = counts.drop(labels=["other"], errors="ignore")
    recognized = int(recognized_counts.sum())
    dominant = str(recognized_counts.idxmax()) if not recognized_counts.empty else None
    consistency = (
        float(recognized_counts.max() / len(labels))
        if not recognized_counts.empty and len(labels)
        else None
    )

    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    parse_success = float(parsed.notna().mean()) if len(parsed) else None
    return {
        "date_format_candidate": bool(recognized > 0 or (parse_success or 0) >= 0.5),
        "date_format_consistency": _safe_float(consistency),
        "date_format_pattern_count": int(len(recognized_counts)),
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

    # Statistical outliers are flags, not proof of errors or safe deletion.
    numeric = pd.to_numeric(non_null, errors="coerce")
    numeric_success = int(numeric.notna().sum())
    numeric_ratio = numeric_success / max(len(non_null), 1) if len(non_null) else None
    numeric_candidate = pd.api.types.is_numeric_dtype(series) or (
        len(non_null) > 0 and numeric_ratio is not None and numeric_ratio >= 0.8
    )
    if numeric_candidate and numeric_success:
        numeric_values = numeric.dropna().astype(float)
        q1, q3 = numeric_values.quantile([0.25, 0.75])
        iqr = float(q3 - q1)
        low, high = float(q1 - 1.5 * iqr), float(q3 + 1.5 * iqr)
        outlier_mask = (numeric_values < low) | (numeric_values > high)
        output.update({
            "skewness": _safe_float(numeric_values.skew()) if len(numeric_values) >= 3 else None,
            "outlier_method": "IQR_1.5x",
            "outlier_lower_bound": _safe_float(low),
            "outlier_upper_bound": _safe_float(high),
            "outlier_count": int(outlier_mask.sum()),
            "outlier_ratio": _safe_float(outlier_mask.mean()),
            "numeric_parse_applicable": not pd.api.types.is_numeric_dtype(series),
            "numeric_parse_success_ratio": _safe_float(numeric_ratio),
            "numeric_parse_failure_ratio": _safe_float(1 - numeric_ratio),
        })
    else:
        numeric_parse_applicable = (
            not pd.api.types.is_numeric_dtype(series)
            and any(token in name for token in ("value", "result", "amount", "count", "score", "rate"))
        )
        output.update({
            "skewness": None,
            "outlier_method": None,
            "outlier_lower_bound": None,
            "outlier_upper_bound": None,
            "outlier_count": None,
            "outlier_ratio": None,
            "numeric_parse_applicable": numeric_parse_applicable,
            "numeric_parse_success_ratio": _safe_float(numeric_ratio) if numeric_parse_applicable else None,
            "numeric_parse_failure_ratio": _safe_float(1 - numeric_ratio) if numeric_parse_applicable and numeric_ratio is not None else None,
        })

    is_string_like = (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
    )
    output.update(_string_lengths(non_null, is_string_like))

    date_context = any(token in name for token in DATE_NAME_TOKENS)
    date_meta = _date_format_consistency(non_null) if is_string_like else {
        "date_format_candidate": False,
        "date_format_consistency": None,
        "date_format_pattern_count": None,
        "dominant_date_format": None,
        "date_parse_success_ratio": None,
        "date_parse_failure_ratio": None,
    }
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

    # Dominant unit share is descriptive; true consistency requires measurement linkage.
    if any(token in name for token in UNIT_NAME_TOKENS):
        unit_values = non_null.astype(str).str.strip().str.casefold()
        counts = unit_values.value_counts()
        output.update({
            "unit_metadata_available": True,
            "unit_unique_count": int(unit_values.nunique()),
            "unit_dominant_share": _safe_float(counts.iloc[0] / len(unit_values)) if len(unit_values) else None,
            "unit_consistency_assessment": "requires_measurement_linkage",
            "unit_values_sample": [str(v) for v in counts.index[:20]],
        })
    else:
        output.update({
            "unit_metadata_available": False,
            "unit_unique_count": None,
            "unit_dominant_share": None,
            "unit_consistency_assessment": "unavailable",
            "unit_values_sample": [],
        })

    # External evidence is required. Null means unavailable, not zero/valid.
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
