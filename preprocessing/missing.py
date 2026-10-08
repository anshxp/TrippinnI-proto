"""Conservative missing-value remediation."""
from __future__ import annotations
import pandas as pd
from preprocessing.policy import RemediationPolicy
from preprocessing.provenance import TransformationLog, TransformationRecord

class MissingValueRemediator:
    name = "MissingValueRemediator"
    def apply(self, table: str, df: pd.DataFrame, profile: dict, log: TransformationLog, policy: RemediationPolicy) -> pd.DataFrame:
        if not policy.enabled:
            return df
        columns = profile.get("columns", {}) if isinstance(profile, dict) else {}
        protected = {c for c,m in columns.items() if isinstance(m,dict) and m.get("semantic_type")=="identifier"}
        result = df.copy()
        for column in result.columns:
            if column in protected and policy.protect_identifiers:
                continue
            missing = result[column].isna()
            if not missing.any():
                continue
            meta = columns.get(column,{}) or {}
            semantic = str(meta.get("semantic_type","")).lower()
            dtype = result[column].dtype
            if policy.protect_datetime_columns and ("datetime" in semantic or pd.api.types.is_datetime64_any_dtype(dtype)):
                continue
            if float(missing.mean()) > policy.max_numeric_imputation_missing_fraction:
                continue
            if pd.api.types.is_numeric_dtype(dtype):
                if not policy.allows("numeric_imputation"):
                    continue
                value = result[column].median()
                if pd.isna(value):
                    continue
                for idx in result.index[missing]:
                    log.add(TransformationRecord(table=table,row_index=int(idx),column=str(column),
                        action="impute",reason="missing_numeric",new_value=value,method="median",
                        detector="MissingDetector"))
                result.loc[missing,column]=value
            elif pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
                if not policy.allows("categorical_imputation"):
                    continue
                value=policy.categorical_missing_token
                for idx in result.index[missing]:
                    log.add(TransformationRecord(table=table,row_index=int(idx),column=str(column),
                        action="impute",reason="missing_categorical",new_value=value,
                        method="explicit_missing_category",detector="MissingDetector"))
                result.loc[missing,column]=value
        return result
