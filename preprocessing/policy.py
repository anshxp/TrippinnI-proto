"""Policy controls for conservative, auditable remediation."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

@dataclass(slots=True)
class RemediationPolicy:
    enabled: bool = False
    impute_numeric: bool = False
    impute_categorical: bool = False
    categorical_missing_token: str = "UNKNOWN"
    remove_exact_duplicates: bool = False
    normalize_datatypes: bool = True
    normalize_whitespace: bool = True
    normalize_units: bool = False
    encode_categoricals: bool = False
    protect_identifiers: bool = True
    protect_datetime_columns: bool = True
    max_numeric_imputation_missing_fraction: float = 0.50
    explicit_unit_map: dict[str, dict[str, str]] = field(default_factory=dict)
    protected_issue_types: set[str] = field(default_factory=lambda: {"clinical_range", "plausibility", "rule_violation"})

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
    return RemediationPolicy(
        enabled=bool(getattr(config, "REMEDIATION_ENABLED", False)),
        impute_numeric=bool(getattr(config, "REMEDIATION_IMPUTE_NUMERIC", False)),
        impute_categorical=bool(getattr(config, "REMEDIATION_IMPUTE_CATEGORICAL", False)),
        remove_exact_duplicates=bool(getattr(config, "REMEDIATION_REMOVE_EXACT_DUPLICATES", False)),
        encode_categoricals=bool(getattr(config, "REMEDIATION_ENABLE_ENCODING", False)),
    )
