"""Confirmed duplicate remediation."""
from __future__ import annotations
import pandas as pd
from preprocessing.policy import RemediationPolicy
from preprocessing.provenance import TransformationLog, TransformationRecord

class DuplicateRemediator:
    name = "DuplicateRemediator"
    def apply(self, table: str, df: pd.DataFrame, profile: dict, log: TransformationLog, policy: RemediationPolicy) -> pd.DataFrame:
        if not policy.enabled or not policy.allows("duplicate_removal"):
            return df
        mask = df.duplicated(keep="first")
        if not mask.any():
            return df
        for idx in df.index[mask]:
            log.add(TransformationRecord(table=table,row_index=int(idx),column=None,action="remove_row",
                reason="exact_duplicate",original_value="row",method="exact_row_match",
                detector="DuplicateDetector"))
        return df.loc[~mask].copy()
