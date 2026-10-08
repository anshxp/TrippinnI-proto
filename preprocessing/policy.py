"""Policy definitions for safe, context-aware data remediation.

The initial prototype intentionally permits only conservative transformations.
Raw source data is never modified by this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RemediationPolicy:
    """Controls which deterministic transformations are allowed."""

    enabled: bool = True
    impute_numeric: bool = True
    impute_categorical: bool = True
    categorical_missing_token: str = "UNKNOWN"
    numeric_strategy: str = "median"
    remove_exact_duplicates: bool = True
    normalize_datatypes: bool = True
    normalize_whitespace: bool = True
    normalize_units: bool = True
    treat_outliers: str = "retain"
    encode_categoricals: bool = False
    protect_identifiers: bool = True
    protect_datetime_columns: bool = True
    max_numeric_imputation_missing_fraction: float = 0.50
    explicit_unit_map: dict[str, dict[str, str]] = field(default_factory=dict)
    protected_issue_types: set[str] = field(
        default_factory=lambda: {"clinical_range", "plausibility", "rule_violation"}
    )

    def allows(self, action: str) -> bool:
        return {
            "numeric_imputation": self.impute_numeric,
            "categorical_imputation": self.impute_categorical,
            "duplicate_removal": self.remove_exact_duplicates,
            "datatype_normalization": self.normalize_datatypes,
            "unit_normalization": self.normalize_units,
            "encoding": self.encode_categoricals,
        }.get(action, False)


def policy_from_config(config: Any) -> RemediationPolicy:
    """Build the policy from config.py without requiring new config fields."""
    return RemediationPolicy(
        enabled=bool(getattr(config, "REMEDIATION_ENABLED", True)),
        impute_numeric=bool(getattr(config, "REMEDIATION_IMPUTE_NUMERIC", True)),
        impute_categorical=bool(getattr(config, "REMEDIATION_IMPUTE_CATEGORICAL", True)),
        remove_exact_duplicates=bool(getattr(config, "REMEDIATION_REMOVE_EXACT_DUPLICATES", True)),
        encode_categoricals=bool(getattr(config, "REMEDIATION_ENABLE_ENCODING", False)),
    )
