"""Module 3 remediation orchestration for bounded in-memory datasets."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import pandas as pd
from preprocessing.datatype import DatatypeRemediator
from preprocessing.duplicates import DuplicateRemediator
from preprocessing.encoding import EncodingRemediator
from preprocessing.missing import MissingValueRemediator
from preprocessing.normalization import NormalizationRemediator
from preprocessing.outliers import OutlierRemediator
from preprocessing.policy import RemediationPolicy
from preprocessing.provenance import TransformationLog

@dataclass(slots=True)
class RemediationResult:
    dataset: dict[str, pd.DataFrame]
    logs: dict[str, TransformationLog] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    def to_dict(self):
        return {"summary": self.summary, "logs": {t: log.to_dict() for t, log in self.logs.items()}}

class RemediationExecutor:
    def __init__(self, policy: RemediationPolicy | None = None):
        self.policy = policy or RemediationPolicy()
        self.missing = MissingValueRemediator()
        self.duplicates = DuplicateRemediator()
        self.datatype = DatatypeRemediator()
        self.normalization = NormalizationRemediator()
        self.outliers = OutlierRemediator()
        self.encoding = EncodingRemediator()

    def run(self, dataset, profiles, quality_results=None):
        if not self.policy.enabled:
            return RemediationResult({k: v.copy() for k, v in dataset.items() if isinstance(v, pd.DataFrame)}, summary={"enabled": False, "transformations": 0})
        refined, logs, tables = {}, {}, {}
        for table, frame in dataset.items():
            if not isinstance(frame, pd.DataFrame):
                continue
            log = TransformationLog()
            profile = profiles.get(table, {}) or {}
            quality = (quality_results or {}).get(table)
            issues = getattr(quality, "issues", []) if quality else []
            current = frame.copy(deep=True)
            rows_before, cells_before = len(current), current.size
            current = self.normalization.apply(table, current, profile, log, self.policy)
            current = self.datatype.apply(table, current, profile, log, self.policy)
            current = self.missing.apply(table, current, profile, log, self.policy)
            current = self.duplicates.apply(table, current, profile, log, self.policy)
            current = self.outliers.apply(table, current, issues, log, self.policy)
            current = self.encoding.apply(table, current, profile, log, self.policy)
            refined[table], logs[table] = current, log
            tables[table] = {"rows_before": rows_before, "rows_after": len(current), "cells_before": cells_before, "cells_after": current.size, "transformations": log.count}
        return RemediationResult(refined, logs, {"enabled": True, "tables": tables, "transformations": sum(log.count for log in logs.values()), "scope": "bounded in-memory remediation dataset"})
