"""Categorical encoding placeholder; task-specific encoding is not enabled."""
from __future__ import annotations
import pandas as pd
from preprocessing.policy import RemediationPolicy
from preprocessing.provenance import TransformationLog

class EncodingRemediator:
    name = "EncodingRemediator"
    def apply(self, table: str, df: pd.DataFrame, profile: dict, log: TransformationLog, policy: RemediationPolicy) -> pd.DataFrame:
        # Intentionally a no-op until downstream task requirements define encoding.
        return df
