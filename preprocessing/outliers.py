"""Outliers are retained unless a separately validated policy is added."""
from __future__ import annotations
import pandas as pd
from preprocessing.policy import RemediationPolicy
from preprocessing.provenance import TransformationLog

class OutlierRemediator:
    name = "OutlierRemediator"
    def apply(self, table: str, df: pd.DataFrame, issues, log: TransformationLog, policy: RemediationPolicy) -> pd.DataFrame:
        # Statistical abnormality alone is not sufficient evidence to change a clinical value.
        return df
